"""Read class evidence from the long- and short-window scattering grids."""

import torch
from torch import nn
from torch.nn import functional as F


class LocalShipEvidence(nn.Module):
    def __init__(self, spectral_width, grid, num_classes, hidden_dim):
        super().__init__()
        self.grid = tuple(grid)
        num_tiles = 2 * self.grid[0] * self.grid[1]
        self.stats_per_tile = spectral_width // num_tiles
        self.embed = nn.Linear(self.stats_per_tile, hidden_dim)
        self.position = nn.Parameter(torch.zeros(num_tiles, hidden_dim))
        self.score = nn.Linear(hidden_dim, num_classes)
        self.project = nn.Linear(hidden_dim, self.stats_per_tile, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, spectral):
        # Q0 stores [window, statistic, frequency, time] in a flattened vector.
        batch = len(spectral)
        grid = spectral.reshape(batch, 2, self.stats_per_tile, *self.grid)
        tiles = grid.permute(0, 1, 3, 4, 2).flatten(1, 3)
        features = F.gelu(self.embed(tiles) + self.position)
        scores = self.score(features).float()
        # Each ship type pools over locations independently of the other types.
        weights = scores.softmax(dim=1)
        evidence = weights.transpose(1, 2) @ features.float()
        presence = (weights * scores).sum(dim=1)
        # Allow more than one ship type to contribute to the same time-frequency tile.
        update = (scores.sigmoid() @ self.project(evidence)) / scores.shape[-1]
        refined = (tiles + update).reshape(batch, 2, *self.grid, self.stats_per_tile)
        return refined.permute(0, 1, 4, 2, 3).reshape_as(spectral), presence
