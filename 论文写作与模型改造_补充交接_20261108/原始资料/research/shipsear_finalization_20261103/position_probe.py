"""Screen two pre-pooling temporal insertion points using Train and Val only."""

import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_formal import (
    CONDITIONS, HERE, ROOT, FixedDataset, FixedViews, evaluation_views,
    fit_normalization, ranking, recognition_metrics, training_views, write_json,
)

POSITIONS = ('baseline', 'short', 'long')
SCREEN_CONDITIONS = ('5dB', '0dB')
TRAIN_VIEWS = (1, 2)
PROBE_EPOCHS = 24


class PositionViews(FixedViews):
    """Share each scattering map across the original and filtered summaries."""

    def summarize(self, coefficients):
        mean = F.adaptive_avg_pool2d(coefficients, self.grid)
        second = F.adaptive_avg_pool2d(coefficients.square(), self.grid)
        std = (second - mean.square()).clamp_min(0).sqrt()
        return torch.cat((mean, std), dim=1).flatten(1)

    def compare_summaries(self, image, transform):
        baseline, modified = [], []
        for part in image.split(self.subbatch):
            coefficients = transform(part[:, 0].float().contiguous())
            padded = F.pad(coefficients, (1, 1, 0, 0), mode='replicate')
            blended = .5 * coefficients + .25 * (padded[..., :-2] + padded[..., 2:])
            baseline.append(self.summarize(coefficients))
            modified.append(self.summarize(blended))
        return torch.cat(baseline), torch.cat(modified)

    @torch.no_grad()
    def forward(self, waves):
        one_d, long_image, short_image = self.fixed(waves)
        with torch.autocast(long_image.device.type, enabled=False):
            long, long_blended = self.compare_summaries(long_image, self.spectral[0])
            short, short_blended = self.compare_summaries(short_image, self.spectral[1])
        return (
            (one_d, long, short),
            (one_d, long, short_blended),
            (one_d, long_blended, short),
        )


class FixedLongBlendViews(PositionViews):
    """Use the exact long-window probe representation in the full model."""

    @torch.no_grad()
    def forward(self, waves):
        one_d, long_image, short_image = self.fixed(waves)
        with torch.autocast(long_image.device.type, enabled=False):
            _, long = self.compare_summaries(long_image, self.spectral[0])
            short = self._spectral_summary(short_image, self.spectral[1])
        return one_d, torch.cat((long, short), dim=1)


@torch.no_grad()
def extract(extractor, dataset, batch_size, view):
    batches = [[] for _ in POSITIONS]
    for start in range(0, len(dataset), batch_size):
        ids = np.arange(start, min(start + batch_size, len(dataset)))
        for batch, outputs in zip(batches, extractor(view(ids))):
            batch.append(torch.cat(outputs, dim=1).cpu())
    return [torch.cat(batch, dim=0) for batch in batches]


def probe_features(features, normalization):
    return ((features - normalization['mean']) / normalization['scale']).cuda()


def fit_probe(train, features, normalization, seed):
    torch.manual_seed(seed)
    linear = nn.Linear(features.shape[1], 7).cuda()
    x = probe_features(features, normalization)
    target = torch.as_tensor(np.tile(train.labels, len(TRAIN_VIEWS)), device='cuda')
    weights = torch.as_tensor(np.tile(train.weights * len(train), len(TRAIN_VIEWS)), device='cuda')
    optimizer = torch.optim.AdamW(linear.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(PROBE_EPOCHS):
        order = np.random.default_rng([seed, 760, epoch]).permutation(len(x))
        for batch in torch.as_tensor(order, device='cuda').split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = (weights[batch] * F.cross_entropy(linear(x[batch]), target[batch], reduction='none')).mean()
            loss.backward()
            optimizer.step()
    return linear.eval()


def focus_metrics(metrics):
    result = {
        key: float(np.mean([metrics[c][key] for c in SCREEN_CONDITIONS]))
        for key in ('acc', 'single_acc', 'double_acc')
    }
    for key in ('double_pair_errors', 'single_to_double'):
        result[key] = sum(metrics[c][key] for c in SCREEN_CONDITIONS)
    return result


def select_position(metrics):
    focus = {position: focus_metrics(values) for position, values in metrics.items()}
    baseline = focus['baseline']
    gates = {}
    for position in ('short', 'long'):
        delta = {key: value - baseline[key] for key, value in focus[position].items()}
        checks = {
            'acc_gain_at_least_1pp': delta['acc'] >= 1 - 1e-10,
            'single_drop_no_more_than_1pp': delta['single_acc'] >= -1 - 1e-10,
            'fewer_combo_errors': delta['double_pair_errors'] < 0,
            'no_more_single_to_double': delta['single_to_double'] <= 0,
        }
        gates[position] = dict(delta_vs_baseline=delta, checks=checks, passed=all(checks.values()))
    eligible = [position for position, row in gates.items() if row['passed']]
    winner = max(eligible, key=lambda p: (focus[p]['acc'], ranking(metrics[p]))) if eligible else None
    return winner, focus, gates


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads((HERE / '正式实验配置.json').read_text(encoding='utf-8'))
    seed = 13
    batch_size = config['batch_size']
    started = time.monotonic()
    train = FixedDataset(config['data_root'], 'Train')
    val = FixedDataset(config['data_root'], 'Val')
    fixed = PositionViews(config)

    clean = extract(fixed, train, batch_size, train.batch)
    stats = [fit_normalization(train, (matrix,), config)['F1'] for matrix in clean]
    del clean
    print('Computed per-arm normalization using original Train.', flush=True)

    train_blocks = []
    for epoch in TRAIN_VIEWS:
        train_blocks.append(extract(
            fixed, train, batch_size,
            lambda ids: training_views(train, ids, epoch, config['noise_probabilities'], seed)[0],
        ))
        print(f'Extracted fixed Train noise view {epoch}.', flush=True)
    probes = []
    for blocks, normalization in zip(zip(*train_blocks), stats):
        probes.append(fit_probe(train, torch.cat(blocks), normalization, seed))
    del train_blocks
    print('Fit three linear probes.', flush=True)

    results = {position: {} for position in POSITIONS}
    for condition in CONDITIONS:
        matrices = extract(fixed, val, batch_size, lambda ids: evaluation_views(
            val, ids, condition, config['evaluation_noise_seed']))
        for position, matrix, normalization, probe in zip(POSITIONS, matrices, stats, probes):
            with torch.no_grad():
                pred = probe(probe_features(matrix, normalization)).argmax(dim=-1).cpu().numpy()
            results[position][condition] = recognition_metrics(val.labels, pred, val.counts, val.contexts)
        print(f'Evaluated Val {condition}.', flush=True)

    winner, focus, gates = select_position(results)
    j = config['spectral_scattering']['J']
    spacings = {p: 1000 * config['representation'][f'{p}_hop'] * 2**j / config['sample_rate']
                for p in ('short', 'long')}
    ROOT.mkdir(exist_ok=True)
    write_json(ROOT / 'position_probe.json', dict(
        seed=seed, evaluation_noise_seed=config['evaluation_noise_seed'],
        dataset=config['dataset_version'], train=len(train), val=len(val),
        train_view_epochs=TRAIN_VIEWS, train_noise_probabilities=config['noise_probabilities'],
        normalization='separate per position, original Train only',
        probe='linear seven-set, weighted CE, AdamW 3e-4, batch 128',
        probe_epochs=PROBE_EPOCHS, features=6930, filter_weights=[.25, .5, .25],
        temporal_spacing_ms=spacings, temporal_span_ms={p: 2 * d for p, d in spacings.items()},
        Val=results, focus=focus, gates=gates, winner=winner, elapsed_seconds=time.monotonic() - started,
    ))
    print(json.dumps(dict(focus=focus, gates=gates, winner=winner), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
