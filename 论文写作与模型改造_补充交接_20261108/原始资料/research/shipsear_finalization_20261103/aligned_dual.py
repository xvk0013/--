"""Matched global and time-aligned waveform-scattering / ShuffleFAC models."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from model import COUNT_MAP, SCATTER_DIM, CalibratedInteraction, LinearEnsemble, Scattering1D, TabM


_AUTHOR_PATH = (
    Path(__file__).resolve().parent.parent
    / 'external_baselines_20261026' / '05_ShuffleFACAuthor' / 'model.py'
)
_spec = importlib.util.spec_from_file_location('aligned_author_shufflefac', _AUTHOR_PATH)
_author = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_author)

_PARTS = 5
_TOKEN_WIDTH = 128
_VIEW_WIDTH = _PARTS * _TOKEN_WIDTH
_MEL_HOP = 2048
_TIME_POOL_FACTOR = 4
_FREQUENCY_REGIONS = 4


def _validate_recording(config):
    if (config['sample_rate'], config['duration_seconds']) != (16000, 5):
        raise ValueError('Aligned ShuffleFAC expects 5 s at 16 kHz.')


class AlignedViews(nn.Module):
    """Fixed waveform-scattering statistics and the author's unresized log-Mel."""

    def __init__(self, config, device='cuda'):
        super().__init__()
        _validate_recording(config)
        self.scattering = Scattering1D(
            shape=config['sample_rate'] * config['duration_seconds'], **config['scattering']
        )
        self.register_buffer(
            'positive_order', torch.as_tensor(self.scattering.meta()['order'] > 0), persistent=False
        )
        self.mel = _author.FrontEnd()
        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, waves):
        x = torch.as_tensor(waves, device=self.positive_order.device, dtype=torch.float32)
        with torch.autocast(x.device.type, enabled=False):
            values = self.scattering(x.contiguous())[:, self.positive_order].clamp_min(1e-12).log()
            sections = (values, *torch.tensor_split(values, 4, dim=-1))
            scatter = torch.cat([
                statistic for section in sections
                for statistic in (section.mean(-1), section.std(-1, unbiased=False))
            ], dim=-1)
            return scatter, self.mel(x)


def _spectral_encoder():
    # Reuse the local author-configured implementation. Frequency 128 -> 4
    # takes five pools; keeping all later convolutions leaves four output bands.
    layers = []
    frequency_bins = 128
    for layer in _author.Model().encoder[:-2]:
        if isinstance(layer, _author.FrequencyPosition) and layer.position.numel() != frequency_bins:
            layer = _author.FrequencyPosition(frequency_bins)
        if isinstance(layer, nn.AvgPool2d):
            if frequency_bins == _FREQUENCY_REGIONS:
                continue
            frequency_bins //= layer.kernel_size[1]
        layers.append(layer)
    return nn.Sequential(*layers)


class AlignedModel(nn.Module):
    """Vary local Mel descriptors only, with a common 640 + 640 TabM interface."""

    def __init__(self, config, normalization):
        super().__init__()
        _validate_recording(config)
        self.alignment_arm = config['alignment_arm']
        if self.alignment_arm not in ('global', 'aligned'):
            raise ValueError(f'Unknown alignment arm: {self.alignment_arm}')

        stats = normalization['F1']
        self.register_buffer('feature_mean', torch.as_tensor(stats['mean'], dtype=torch.float32).clone())
        self.register_buffer('feature_scale', torch.as_tensor(stats['scale'], dtype=torch.float32).clone())
        if self.feature_mean.shape != (SCATTER_DIM,) or self.feature_scale.shape != (SCATTER_DIM,):
            raise ValueError('Alignment expects normalization of the 3730 waveform-scattering features.')

        # Centered Mel frame j sits at j * hop. Two 2x time pools group
        # four frames, so output k has nominal center (4*k + 1.5) * hop.
        # Assign centers to [0,1.25), [1.25,2.5), [2.5,3.75), [3.75,5] s.
        # Local convolution and frequency attention can cross these boundaries.
        samples = config['sample_rate'] * config['duration_seconds']
        mel_frames = 1 + samples // _MEL_HOP
        lengths = [0] * 4
        for k in range(mel_frames // _TIME_POOL_FACTOR):
            center_sample = (_TIME_POOL_FACTOR * k + (_TIME_POOL_FACTOR - 1) / 2) * _MEL_HOP
            lengths[min(3, int(4 * center_sample / samples))] += 1
        self.segment_lengths = tuple(lengths)

        self.spectral_encoder = _spectral_encoder()
        self.scatter_projection = nn.Sequential(
            nn.Linear(SCATTER_DIM // _PARTS, _TOKEN_WIDTH), nn.LayerNorm(_TOKEN_WIDTH)
        )
        self.spectral_projection = nn.Sequential(
            nn.Linear(256 * _FREQUENCY_REGIONS, _TOKEN_WIDTH), nn.LayerNorm(_TOKEN_WIDTH)
        )
        width = config['model']['d_block']
        self.tabm = TabM.make(
            n_num_features=2 * _VIEW_WIDTH, d_out=7, arch_type='tabm', **config['model']
        )
        self.tabm.count_readout = LinearEnsemble(width, 3, k=config['model']['k'])
        nn.init.zeros_(self.tabm.count_readout.weight)
        nn.init.zeros_(self.tabm.count_readout.bias)
        interaction = SimpleNamespace(
            base=self.tabm.backbone.blocks[0][0],
            split=_VIEW_WIDTH,
            left=nn.Linear(_VIEW_WIDTH, 64, bias=False),
            right=nn.Linear(_VIEW_WIDTH, 64, bias=False),
            project=nn.Linear(64, width, bias=False),
        )
        nn.init.zeros_(interaction.project.weight)
        self.tabm.backbone.blocks[0][0] = CalibratedInteraction(interaction, 'adaptive')

    def forward(self, scatter, mel):
        waveform = (scatter.float() - self.feature_mean) / self.feature_scale
        waveform = self.scatter_projection(waveform.reshape(len(waveform), _PARTS, -1))

        encoded = self.spectral_encoder(mel.transpose(2, 3))
        whole = encoded.mean(dim=2).flatten(1)
        if self.alignment_arm == 'aligned':
            quarters = encoded.split(self.segment_lengths, dim=2)
            spectra = torch.stack((whole, *(part.mean(dim=2).flatten(1) for part in quarters)), dim=1)
        else:
            spectra = whole[:, None, :].expand(-1, _PARTS, -1)
        spectra = self.spectral_projection(spectra)

        # Each view uses the same [whole, q1, q2, q3, q4] order. Preserve
        # the branch boundary required by the existing calibrated interaction.
        features = torch.cat((waveform.flatten(1), spectra.flatten(1)), dim=-1).float()
        h = self.tabm.backbone(self.tabm.ensemble_view(features))
        return self.tabm.output(h) + self.tabm.count_readout(h)[..., COUNT_MAP]
