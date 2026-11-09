"""Fixed dual scattering and direct seven-set TabM readout for Q0."""
from __future__ import annotations

import sys
from pathlib import Path

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

COUNT_MAP = (0, 1, 1, 1, 2, 2, 2)
SET_NAMES = ('E', 'PF', 'MB', 'RV', 'PF+MB', 'PF+RV', 'MB+RV')
SCATTER_DIM = 3730
SPECTRAL_DIM = 3200


class CalibratedInteraction(nn.Module):
    """Project both scattering views and gate their product into the first block."""

    def __init__(self, base, width):
        super().__init__()
        self.base = base
        self.left = nn.Linear(SCATTER_DIM, 64, bias=False)
        self.right = nn.Linear(SPECTRAL_DIM, 64, bias=False)
        self.project = nn.Linear(64, width, bias=False)
        nn.init.zeros_(self.project.weight)
        self.gater = nn.Sequential(
            nn.Linear(3 * self.left.out_features + 4, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.gater[-1].weight)
        nn.init.zeros_(self.gater[-1].bias)

    def forward(self, x):
        def rms(values):
            return values.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()

        adapted = x * self.base.r
        left = self.left(adapted[..., :SCATTER_DIM])
        right = self.right(adapted[..., SCATTER_DIM:])
        product = left * right
        direct = self.base(x)
        cross = self.project(product) * self.base.s
        strength = torch.cat((
            torch.log1p(rms(left)),
            torch.log1p(rms(right)),
            torch.log1p(rms(product)),
            torch.log1p(rms(cross)) - torch.log1p(rms(direct)),
        ), dim=-1)
        shape = (left.shape[-1],)
        evidence = torch.cat((
            F.layer_norm(left, shape),
            F.layer_norm(right, shape),
            F.layer_norm(product, shape),
            strength,
        ), dim=-1)
        gate = 2 * torch.sigmoid(self.gater(evidence))
        return direct + gate * cross

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


class DirectSetModel(nn.Module):
    """Q0 ensemble with a direct seven-set output."""

    def __init__(self, config, normalization):
        super().__init__()
        stats = normalization['F1']
        self.register_buffer('feature_mean', torch.as_tensor(stats['mean'], dtype=torch.float32).clone())
        self.register_buffer('feature_scale', torch.as_tensor(stats['scale'], dtype=torch.float32).clone())
        n_features = SCATTER_DIM + SPECTRAL_DIM
        if self.feature_mean.numel() != n_features or self.feature_scale.numel() != n_features:
            raise ValueError('Normalization dimension does not match the 6930 Q0 features.')
        self.tabm = TabM.make(n_num_features=n_features, d_out=7,
                              arch_type='tabm', **config['model'])
        # Match the training constructor's discarded count head to preserve the RNG order.
        LinearEnsemble(config['model']['d_block'], 3, k=config['model']['k'])
        first_block = self.tabm.backbone.blocks[0]
        first_block[0] = CalibratedInteraction(first_block[0], config['model']['d_block'])

    def forward(self, scatter, spectral):
        values = torch.cat((scatter.float(), spectral.float()), -1)
        return self.tabm((values - self.feature_mean) / self.feature_scale)


class Recognizer:
    """Waveforms only, with source-equivalent 32/64 numerical batch paths."""

    def __init__(self, checkpoint, device='cuda'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('experiment_id') != 'SE-Q0-MBTrainValSwap-13':
            raise ValueError('Expected the sealed ShipsEar Q0 MB Train/Val checkpoint.')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.config = saved['config']
        self.device = torch.device(device)
        self.feature_batch_size = int(self.config.get('feature_batch_size', self.config.get('batch_size', 32)))
        self.classifier_batch_size = int(self.config.get('classifier_batch_size', self.config.get('evaluation_batch_size', 64)))
        if self.feature_batch_size != 32 or self.classifier_batch_size != 64:
            raise ValueError('Q0 inference uses feature32 and classifier64 batches.')
        self.fixed = FixedViews(self.config, self.device).eval().requires_grad_(False)
        self.model = DirectSetModel(self.config, saved['normalization']).to(self.device)
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

