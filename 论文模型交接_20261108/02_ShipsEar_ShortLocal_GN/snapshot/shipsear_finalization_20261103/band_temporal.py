"""Matched local temporal processing with and without frequency position."""

import torch
from torch import nn
from torch.nn import functional as F

from model import FixedViews, JointModel


def spectral_summary(coefficients, grid):
    mean = F.adaptive_avg_pool2d(coefficients, grid)
    second = F.adaptive_avg_pool2d(coefficients.square(), grid)
    variance = (second - mean.square()).clamp_min(0)
    zero = variance == 0
    std = torch.where(zero, 1., variance).sqrt().masked_fill(zero, 0.)
    return torch.cat((mean, std), dim=1).flatten(1)


def baseline_spectral(position, other, coeff, grid):
    """Construct the unchanged long-then-short representation."""
    selected = spectral_summary(coeff, grid)
    return torch.cat((selected, other) if position == 'long' else (other, selected), dim=1)


class BandViews(FixedViews):
    """Return the selected spatial map and unchanged summaries of the other views."""

    def __init__(self, config, position, device='cuda'):
        super().__init__(config, device=device)
        if position not in ('long', 'short'):
            raise ValueError(f'Unknown temporal placement: {position}')
        self.selected = 0 if position == 'long' else 1

    @torch.no_grad()
    def forward(self, waves):
        scatter, long, short = self.fixed(waves)
        images = (long, short)
        other = 1 - self.selected
        with torch.autocast(long.device.type, enabled=False):
            other_summary = self._spectral_summary(images[other], self.spectral[other])
            transform = self.spectral[self.selected]
            coefficients = torch.cat([
                transform(image[:, 0].float().contiguous())
                for image in images[self.selected].split(self.subbatch)
            ])
        return scatter.float(), other_summary, coefficients


class BandTemporalModel(JointModel):
    """Share a zero-sum 3-tap residual over paths; vary only the gate input."""

    def __init__(self, config, arm, normalization):
        super().__init__(config, 'P50_noise50', normalization)
        if arm not in ('B', 'C'):
            raise ValueError(f'Unknown frequency-condition arm: {arm}')
        self.band_arm = arm
        self.position = config['band_position']
        self.grid = tuple(config['spectral_scattering']['summary_grid'])
        representation = config['representation']
        self.frequency_range = representation['frequency_range_hz']
        self.frequency_step = (
            config['sample_rate'] * 2 ** config['spectral_scattering']['J']
            / representation[f'{self.position}_fft']
        )
        self.temporal_taps = nn.Parameter(torch.zeros(2))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config['seed'] + 8731)
            self.band_gate = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))

    def local_temporal(self, coeff):
        # Per-row content has no position identifier in the shared control.
        content = coeff.square().mean(dim=(1, 3)).log1p()
        frequency = torch.arange(content.shape[1], device=coeff.device, dtype=coeff.dtype)
        lower, upper = self.frequency_range
        frequency = 2 * (frequency * self.frequency_step - lower) / (upper - lower) - 1
        if self.band_arm == 'B':
            frequency = torch.zeros_like(frequency)
        gate_inputs = torch.stack((content, frequency.expand_as(content)), dim=-1)
        strength = self.band_gate(gate_inputs).sigmoid()[:, None, :, :]

        padded = F.pad(coeff, (1, 1, 0, 0), mode='replicate')
        previous = padded[..., :-2] - coeff
        following = padded[..., 2:] - coeff
        residual = self.temporal_taps[0] * previous + self.temporal_taps[1] * following
        return coeff + strength * residual

    def forward(self, scatter, other, coeff):
        with torch.autocast(coeff.device.type, enabled=False):
            spectral = baseline_spectral(self.position, other.float(), self.local_temporal(coeff.float()), self.grid)
        return super().forward(scatter, spectral)
