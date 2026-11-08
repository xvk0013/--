"""Retained V1 model, packaged independently of historical experiment modules.

P25/P50 change only the runner's waveform/noise distribution. This module keeps
all original acoustic transforms, Train normalization and classifier operations.
Project classes were extracted without changing their arithmetic from the retained
V1 dependency chain; third-party algorithms are unchanged local vendor copies.
See vendor/README.md for exact source files and licenses.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from tabm import TabM, LinearEnsemble
from kymatio.torch import Scattering1D, Scattering2D

ARMS = ('P25_noise75', 'P50_noise50')
COUNT_MAP = (0, 1, 1, 1, 2, 2, 2)
SET_NAMES = ('E', 'PF', 'MB', 'RV', 'PF+MB', 'PF+RV', 'MB+RV')
SCATTER_DIM = 3730
SPECTRAL_DIM = 3200


def count_probabilities(p):
    return torch.stack((p[..., 0], p[..., 1:4].sum(-1), p[..., 4:7].sum(-1)), -1)


class CalibratedInteraction(nn.Module):
    def __init__(self,old,mode):
        super().__init__()
        self.base,self.split=old.base,old.split
        self.left,self.right,self.project=old.left,old.right,old.project
        self.combine_views = None
        self.mode=mode
        if mode=='constant':self.gate_logit=nn.Parameter(torch.zeros(()))
        elif mode=='adaptive':
            self.gater=nn.Sequential(nn.Linear(3*old.left.out_features+4,16),nn.GELU(),nn.Linear(16,1))
            nn.init.zeros_(self.gater[-1].weight);nn.init.zeros_(self.gater[-1].bias)

    def forward(self,x):
        adapted=x*self.base.r
        u=self.left(adapted[...,:self.split]);v=self.right(adapted[...,self.split:])
        fused = u*v if self.combine_views is None else self.combine_views(u, v)
        direct=self.base(x);cross=self.project(fused)*self.base.s
        if self.mode=='constant':gate=2*torch.sigmoid(self.gate_logit)
        elif self.mode=='adaptive':
            normalize=lambda t:F.layer_norm(t,(t.shape[-1],))
            rms=lambda t:t.float().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
            strength=torch.cat((torch.log1p(rms(u)),torch.log1p(rms(v)),
                torch.log1p(rms(fused)),torch.log1p(rms(cross))-torch.log1p(rms(direct))),-1)
            evidence=torch.cat((normalize(u),normalize(v),normalize(fused),strength),-1)
            gate=2*torch.sigmoid(self.gater(evidence))
        else:gate=1.
        return direct+gate*cross

class WaveformInputs(nn.Module):
    def __init__(self,config,device='cuda'):
        super().__init__();self.sample_rate=config['sample_rate'];self.representation=config['representation']
        self.scattering=Scattering1D(shape=config['sample_rate']*config['duration_seconds'],**config['scattering'])
        self.register_buffer('positive_order',torch.as_tensor(self.scattering.meta()['order']>0),persistent=False)
        self.register_buffer('long_window',torch.hann_window(self.representation['long_fft'],device=device),persistent=False)
        self.register_buffer('short_window',torch.hann_window(self.representation['short_fft'],device=device),persistent=False)
        self.to(device).eval().requires_grad_(False)
    @torch.no_grad()
    def forward(self,waves):
        x=torch.as_tensor(np.ascontiguousarray(waves),device=self.positive_order.device,dtype=torch.float32)
        with torch.autocast(x.device.type,enabled=False):
            p=self.representation;upper=p['frequency_range_hz'][1]
            long=torch.stft(x,p['long_fft'],p['long_hop'],p['long_fft'],self.long_window,center=True,return_complex=True)
            short=torch.stft(x,p['short_fft'],p['short_hop'],p['short_fft'],self.short_window,center=True,return_complex=True)
            long=torch.log(long[:,:int(upper*p['long_fft']/self.sample_rate)+1].abs().square()+1e-6).unsqueeze(1)
            short=torch.log(short[:,:int(upper*p['short_fft']/self.sample_rate)+1].abs().square()+1e-6).unsqueeze(1)
            values=self.scattering(x)[:,self.positive_order].clamp_min(1e-12).log()
            sections=(values,*torch.tensor_split(values,4,dim=-1))
            scatter=torch.cat([v for s in sections for v in (s.mean(-1),s.std(-1,unbiased=False))],-1)
        return scatter,long.half(),short.half()


class FixedViews(nn.Module):
    """Unchanged fixed 1D summary and native long/short 2D summaries."""

    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        self.fixed = WaveformInputs(config, device)
        p = config['representation']
        n = config['sample_rate'] * config['duration_seconds']
        upper = p['frequency_range_hz'][1]
        self.shapes = tuple((int(upper * p[name + '_fft'] / config['sample_rate']) + 1,
                             1 + n // p[name + '_hop']) for name in ('long', 'short'))
        s = config['spectral_scattering']
        self.grid = tuple(s['summary_grid'])
        self.subbatch = int(s['subbatch'])
        self.spectral = nn.ModuleList([
            Scattering2D(J=s['J'], shape=shape, L=s['L'], max_order=s['max_order'])
            for shape in self.shapes
        ])
        self.to(device).requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def _spectral_summary(self, image, transform):
        parts = []
        for start in range(0, len(image), self.subbatch):
            coefficients = transform(image[start:start + self.subbatch, 0].float().contiguous())
            mean = F.adaptive_avg_pool2d(coefficients, self.grid)
            second = F.adaptive_avg_pool2d(coefficients.square(), self.grid)
            std = (second - mean.square()).clamp_min(0).sqrt()
            parts.append(torch.cat((mean, std), 1).flatten(1))
        return torch.cat(parts)

    @torch.no_grad()
    def forward(self, waves):
        scatter, long, short = self.fixed(waves)
        with torch.autocast(long.device.type, enabled=False):
            spectral = torch.cat((self._spectral_summary(long, self.spectral[0]),
                                  self._spectral_summary(short, self.spectral[1])), -1)
        return scatter.float(), spectral.float()


FixedInputs = FixedViews


class JointModel(nn.Module):
    """Complete retained V1; the arm changes no model computation."""

    def __init__(self, config, arm, normalization):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown final noise-distribution arm: ' + str(arm))
        self.config, self.arm, self.view = config, arm, 'F1'
        stats = normalization[self.view]
        self.register_buffer('feature_mean', torch.as_tensor(stats['mean'], dtype=torch.float32).clone())
        self.register_buffer('feature_scale', torch.as_tensor(stats['scale'], dtype=torch.float32).clone())
        n_features = SCATTER_DIM + SPECTRAL_DIM
        if self.feature_mean.numel() != n_features or self.feature_scale.numel() != n_features:
            raise ValueError('Normalization dimension does not match the retained 6930 features.')
        self.tabm = TabM.make(n_num_features=n_features, d_out=7,
                              arch_type='tabm', **config['model'])
        self.tabm.count_readout = LinearEnsemble(config['model']['d_block'], 3,
                                                k=config['model']['k'])
        nn.init.zeros_(self.tabm.count_readout.weight)
        nn.init.zeros_(self.tabm.count_readout.bias)
        old = SimpleNamespace(base=self.tabm.backbone.blocks[0][0], split=SCATTER_DIM,
                              left=nn.Linear(SCATTER_DIM, 64, bias=False),
                              right=nn.Linear(SPECTRAL_DIM, 64, bias=False),
                              project=nn.Linear(64, config['model']['d_block'], bias=False))
        nn.init.zeros_(old.project.weight)
        self.tabm.backbone.blocks[0][0] = CalibratedInteraction(old, 'adaptive')

    def forward(self, scatter, spectral):
        x = torch.cat((scatter.float(), spectral.float()), -1)
        x = (x - self.feature_mean) / self.feature_scale
        h = self.tabm.backbone(self.tabm.ensemble_view(x))
        return self.tabm.output(h) + self.tabm.count_readout(h)[..., COUNT_MAP]


def make_models(config, normalization, device='cuda'):
    """Both arms inherit exactly the same retained V1 and Train normalization."""
    checkpoint = Path(config['initial_checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = HERE / checkpoint
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('arm') != 'V1_stored_single':
        raise ValueError('Final noise-distribution training must start from retained V1.')
    models = {}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(config.get('seed', 13)))
            model = JointModel(config, arm, normalization)
        model.load_state_dict(saved['model'], strict=True)
        models[arm] = model.to(device)
    return models


class Recognizer:
    """Waveforms only, with source-equivalent 32/64 numerical batch paths."""

    def __init__(self, checkpoint, device='cuda'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.config = saved['config']
        self.device = torch.device(device)
        self.feature_batch_size = int(self.config.get('feature_batch_size', self.config.get('batch_size', 32)))
        self.classifier_batch_size = int(self.config.get('classifier_batch_size', self.config.get('evaluation_batch_size', 64)))
        if self.feature_batch_size != 32 or self.classifier_batch_size != 64:
            raise ValueError('Retained V1 inference uses feature32 and classifier64 batches.')
        self.fixed = FixedViews(self.config, self.device).eval().requires_grad_(False)
        # The V1 reference and both distribution arms have the same state layout.
        arm = 'P50_noise50' if saved['arm'] == 'V1_stored_single' else saved['arm']
        self.model = JointModel(self.config, arm, saved['normalization']).to(self.device)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, waves):
        waves = np.atleast_2d(np.asarray(waves, dtype=np.float32))
        expected_samples = self.config['sample_rate'] * self.config['duration_seconds']
        if waves.ndim != 2 or waves.shape[1] != expected_samples:
            raise ValueError(f'Expected waveforms with shape [batch,{expected_samples}].')
        if not len(waves):
            return dict(probabilities=np.empty((0, len(SET_NAMES)), dtype=np.float32),
                        predictions=np.empty(0, dtype=np.int64),
                        counts=np.empty(0, dtype=np.int64), sets=[])
        features = [[], []]
        for start in range(0, len(waves), self.feature_batch_size):
            values = self.fixed(waves[start:start + self.feature_batch_size])
            for destination, value in zip(features, values):
                destination.append(value.detach().float().cpu())
        features = tuple(torch.cat(parts) for parts in features)
        parts = []
        for start in range(0, len(waves), self.classifier_batch_size):
            batch = tuple(value[start:start + self.classifier_batch_size].to(self.device)
                          for value in features)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == 'cuda'):
                logits = self.model(*batch)
            parts.append(logits.float().softmax(-1).cpu().numpy().mean(1))
        probability = np.concatenate(parts)
        prediction = probability.argmax(-1)
        return dict(probabilities=probability, predictions=prediction,
                    counts=np.asarray(COUNT_MAP)[prediction],
                    sets=[SET_NAMES[int(i)] for i in prediction])

