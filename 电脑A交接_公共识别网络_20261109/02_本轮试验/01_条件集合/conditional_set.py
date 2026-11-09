"""Class-conditional reuse of local evidence for the seven ShipsEar sets."""

import math

import torch
from torch import nn

from model import SCATTER_DIM, SPECTRAL_DIM
from short_local import ShortLocalEncoder


class ConditionalSetReadout(nn.Module):
    """Normalize all legal stop/add paths, then sum the two orders of each pair."""

    def __init__(self, mode, strength):
        super().__init__()
        if mode not in ('local', 'uniform') or not 0 <= strength < 1:
            raise ValueError('Invalid evidence discount.')
        self.mode = mode
        self.strength = strength
        self.stop_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer('repeat_ship', torch.eye(3, dtype=torch.bool), persistent=False)
        self.register_buffer('pairs', torch.triu_indices(3, 3, offset=1), persistent=False)

    def continuation_scores(self, scores):
        ships = scores[..., :3]
        background = scores[..., 3:]
        support = (ships - background).sigmoid()
        if self.mode == 'uniform':
            support = support.mean(dim=1, keepdim=True)
        log_weight = torch.log1p(-self.strength * support)
        # The two class axes represent the anchor and the proposed added ship.
        weighted = ships.unsqueeze(2) + log_weight.unsqueeze(3)
        return weighted.logsumexp(dim=1) - math.log(scores.shape[1])

    def forward(self, scores):
        scores = scores.float()
        pooled = scores.logsumexp(dim=1) - math.log(scores.shape[1])
        background = pooled[:, 3:]
        first = torch.cat((background, pooled[:, :3]), dim=1).log_softmax(dim=1)

        continuation = self.continuation_scores(scores).masked_fill(self.repeat_ship, -torch.inf)
        stop = (background + self.stop_bias).unsqueeze(1).expand(-1, 3, -1)
        second = torch.cat((stop, continuation), dim=2).log_softmax(dim=2)
        singles = first[:, 1:] + second[:, :, 0]
        ordered_pairs = first[:, 1:].unsqueeze(2) + second[:, :, 1:]
        left, right = self.pairs
        doubles = torch.logaddexp(ordered_pairs[:, left, right], ordered_pairs[:, right, left])
        return torch.cat((first[:, :1], singles, doubles), dim=1)


class ConditionalSetModel(nn.Module):
    """Read ShortLocal-GN's existing statistics as local cells."""

    def __init__(self, config, normalization):
        super().__init__()
        self.short_local = ShortLocalEncoder(config)
        self.grid = tuple(config['spectral_scattering']['summary_grid'])
        cells_per_spectrum = math.prod(self.grid)
        spectral_width = SPECTRAL_DIM // 2 // cells_per_spectrum
        width = config['model']['width']
        self.waveform = nn.Linear(SCATTER_DIM // 5, width)
        self.long_view = nn.Linear(spectral_width, width)
        self.short_view = nn.Linear(spectral_width, width)
        self.cell_position = nn.Parameter(torch.zeros(1, 4 + 2 * cells_per_spectrum, width))
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(config['model']['dropout']), nn.Linear(width, 4),
        )
        self.readout = ConditionalSetReadout(config['model']['mode'], config['model']['strength'])

        stats = normalization['F1']
        self.register_buffer('feature_mean', stats['mean'].float().clone())
        self.register_buffer('feature_scale', stats['scale'].float().clone())

    def spectral_cells(self, summary):
        return summary.reshape(len(summary), -1, math.prod(self.grid)).transpose(1, 2)

    def forward(self, scatter, long_summary, short_mel):
        wave = (scatter.float() - self.feature_mean[:SCATTER_DIM]) / self.feature_scale[:SCATTER_DIM]
        long = (long_summary.float() - self.feature_mean[SCATTER_DIM:]) / self.feature_scale[SCATTER_DIM:]
        wave = self.waveform(wave.reshape(len(wave), 5, -1))
        long = self.long_view(self.spectral_cells(long))
        short = self.short_view(self.spectral_cells(self.short_local(short_mel)))
        cells = torch.cat((wave[:, 1:], long, short), dim=1)
        scores = self.score(cells + wave[:, :1] + self.cell_position)
        # The formal trainer accepts one or more members on the middle axis.
        return self.readout(scores).unsqueeze(1)
