"""Score whole-clip ship sets using the union of four local evidence states."""

import torch
from torch import nn

from model import SCATTER_DIM, SPECTRAL_DIM


class FrozenCatFeatures(nn.Module):
    """Expose the standard GN-CAT member states and same-segment summaries."""

    def __init__(self, front, cat_model, grid):
        super().__init__()
        self.front = front
        self.cat_model = cat_model
        self.grid = tuple(grid)
        if self.grid != (8, 4) or cat_model.tabm.num_module is not None:
            raise ValueError('Expected the frozen GN-CAT 8-by-4 numerical representation.')

    @torch.no_grad()
    def encode(self, scatter, long_summary, short_mel):
        cat = self.cat_model
        with torch.autocast('cuda', dtype=torch.bfloat16):
            short_summary = cat.short_local(short_mel)
            features = torch.cat((scatter.float(), long_summary.float(), short_summary.float()), dim=-1)
            values = (features - cat.feature_mean) / cat.feature_scale
            hidden = cat.tabm.backbone(cat.tabm.ensemble_view(values))
            logits = cat.tabm.output(hidden)

        wave, long, short = values.split((SCATTER_DIM, SPECTRAL_DIM // 2, SPECTRAL_DIM // 2), dim=-1)
        wave = wave.reshape(len(wave), 5, -1)[:, 1:]
        long = self.spectral_segments(long)
        short = self.spectral_segments(short)
        return torch.cat((wave, long, short), dim=-1), hidden, logits

    def spectral_segments(self, summary):
        # Upstream flattening orders statistic channels, frequency bands, then time.
        return summary.reshape(len(summary), -1, *self.grid).movedim(-1, 1).flatten(2)

    def forward(self, waves):
        return self.encode(*self.front(waves))


class SetPathScores(nn.Module):
    """Use a normalized path prior separately for each legal whole-clip set."""

    def __init__(self, change_cost):
        super().__init__()
        masks = torch.tensor([0, 1, 2, 4, 3, 5, 6])
        cardinalities = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3])
        states = torch.arange(len(masks))
        paths = torch.cartesian_prod(states, states, states, states)
        members = masks[paths]
        unions = members[:, 0] | members[:, 1] | members[:, 2] | members[:, 3]
        paths = torch.cat([paths[unions == mask] for mask in masks], dim=0)
        self.group_sizes = tuple(int((unions == mask).sum()) for mask in masks)

        members = masks[paths]
        changes = cardinalities[members[:, 1:] ^ members[:, :-1]].sum(dim=-1)
        log_prior = -change_cost * changes.float()
        offsets = torch.arange(4) * len(masks)
        self.register_buffer('paths', paths, persistent=False)
        self.register_buffer('indices', (paths + offsets).flatten(), persistent=False)
        self.register_buffer('log_prior', log_prior, persistent=False)

    def forward(self, state_scores):
        evidence = state_scores.float() - state_scores[..., :1].float()
        scores = evidence.flatten(-2).index_select(-1, self.indices)
        scores = scores.unflatten(-1, (-1, 4)).sum(dim=-1)
        path_groups = (scores + self.log_prior).split(self.group_sizes, dim=-1)
        prior_groups = self.log_prior.split(self.group_sizes)
        return torch.stack([paths.logsumexp(dim=-1) - prior.logsumexp(dim=-1)
                            for paths, prior in zip(path_groups, prior_groups)], dim=-1)


class TemporalPaths(nn.Module):
    """Add normalized local path evidence to the full TabM member scores."""

    def __init__(self, config, change_cost):
        super().__init__()
        width = config['temporal_paths']['width']
        self.segment_encoder = nn.Sequential(nn.Linear(1546, width), nn.LayerNorm(width), nn.GELU())
        self.member_context = nn.Linear(config['model']['d_block'], width)
        self.state_scores = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                          nn.Dropout(config['model']['dropout']), nn.Linear(width, 7))
        nn.init.zeros_(self.state_scores[-1].weight)
        nn.init.zeros_(self.state_scores[-1].bias)
        self.paths = SetPathScores(change_cost)

    def forward(self, segments, hidden, base_logits):
        local = self.segment_encoder(segments).unsqueeze(1)
        context = self.member_context(hidden).unsqueeze(2)
        return base_logits.float() + self.paths(self.state_scores(local + context))
