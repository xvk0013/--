"""Read the frozen ShipsEar mixtures and replay the formal evaluation protocol."""
import csv
import json
from pathlib import Path

import numpy as np

SET_NAMES = ('E', 'PF', 'MB', 'RV', 'PF+MB', 'PF+RV', 'MB+RV')
SET_COUNTS = np.asarray((0, 1, 1, 1, 2, 2, 2), dtype=np.int64)
CONDITIONS = ('original', '10dB', '5dB', '0dB', '-5dB', '-10dB')
SNRS = (10, 5, 0, -5, -10)


class FixedDataset:
    def __init__(self, root, split):
        if split not in ('Train', 'Val', 'Test'):
            raise ValueError(split)
        self.split = split
        root = Path(root).resolve()
        config = json.loads((root / 'CONFIG.json').read_text(encoding='utf-8'))
        with (root / 'DATA_INDEX.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = sorted(
                (row for row in csv.DictReader(f) if row['split'] == split),
                key=lambda row: int(row['array_index']),
            )
        self.audio = np.load(root / 'data' / split / 'audio.npy', mmap_mode='r')
        samples = int(config['sample_rate'] * config['duration_seconds'])
        if self.audio.shape != (len(rows), samples):
            raise ValueError('Stored waveform shape differs from the formal data index.')
        self.labels = np.asarray([int(row['label_index']) for row in rows], dtype=np.int64)
        self.counts = np.asarray([int(row['count']) for row in rows], dtype=np.int64)
        self.ids = np.asarray([row['sample_id'] for row in rows])
        self.contexts = np.asarray([row['context_condition'] for row in rows])

    def __len__(self):
        return len(self.labels)

    def batch(self, indices):
        return np.array(self.audio[np.atleast_1d(indices).astype(np.int64)], copy=True)


def add_noise(wave, count, snr, random):
    x = np.asarray(wave, dtype=np.float64)
    noise = random.standard_normal(len(x))
    noise -= noise.mean()
    noise /= np.sqrt(np.mean(noise * noise))
    reference = float(np.mean(x * x)) if count > 0 else .05 ** 2
    mixed = x + noise * np.sqrt(reference / 10 ** (snr / 10))
    mixed *= .05 / np.sqrt(np.mean(mixed * mixed))
    return mixed.astype(np.float32)


def evaluation_views(data, indices, condition, seed):
    indices = np.asarray(indices)
    waves = data.batch(indices)
    if condition == 'original':
        return waves
    snr = SNRS[CONDITIONS.index(condition) - 1]
    code = 1 if data.split == 'Val' else 2
    for j, i in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, code, int(i)]))
        waves[j] = add_noise(waves[j], data.counts[i], snr, rng)
    return waves


def _accuracy(correct, mask):
    return float(correct[mask].mean() * 100) if mask.any() else None


def recognition_metrics(labels, predictions, counts, contexts):
    correct = labels == predictions
    predicted_count = SET_COUNTS[predictions]
    confusion, count_confusion = np.zeros((7, 7), dtype=np.int64), np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    np.add.at(count_confusion, (counts, predicted_count), 1)
    result = dict(n=len(labels), acc=float(correct.mean() * 100),
        single_acc=_accuracy(correct, counts == 1), double_acc=_accuracy(correct, counts == 2),
        noise_acc=_accuracy(correct, counts == 0), count_acc=float((predicted_count == counts).mean() * 100),
        single_to_double=int(((counts == 1) & (predicted_count == 2)).sum()),
        double_to_single=int(((counts == 2) & (predicted_count == 1)).sum()),
        single_to_background=int(((counts == 1) & (predicted_count == 0)).sum()),
        double_to_background=int(((counts == 2) & (predicted_count == 0)).sum()),
        single_type_errors=int(((counts == 1) & (predicted_count == 1) & ~correct).sum()),
        double_pair_errors=int(((counts == 2) & (predicted_count == 2) & ~correct).sum()),
        confusion=confusion.tolist(), final_count_confusion=count_confusion.tolist())
    for context in ('C0', 'C1', 'C2'):
        result['double_' + context + '_acc'] = _accuracy(correct, (counts == 2) & (contexts == context))
    result['per_set_acc'] = {name: _accuracy(correct, labels == k) for k, name in enumerate(SET_NAMES)}
    return result
