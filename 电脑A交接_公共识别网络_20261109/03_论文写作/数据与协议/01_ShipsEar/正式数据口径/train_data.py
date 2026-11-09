"""Formal stored mixtures, original loss budgets and unchanged AWGN protocol.

Extracted from fixed_data.py/noise_data.py in the existing project. There is no
artificial Single envelope, source loading or new recording arrangement.
"""
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
        self.root, self.split = Path(root).resolve(), split
        self.cfg = json.loads((self.root / 'CONFIG.json').read_text(encoding='utf-8'))
        with (self.root / 'DATA_INDEX.csv').open(encoding='utf-8-sig', newline='') as f:
            self.rows = sorted((r for r in csv.DictReader(f) if r['split'] == split),
                key=lambda r: int(r['array_index']))
        self.audio = np.load(self.root / 'data' / split / 'audio.npy', mmap_mode='r')
        if self.audio.shape != (len(self.rows), int(self.cfg['sample_rate'] * self.cfg['duration_seconds'])):
            raise ValueError('Stored waveform shape differs from the formal data index.')
        self.labels = np.asarray([int(r['label_index']) for r in self.rows], dtype=np.int64)
        self.counts = np.asarray([int(r['count']) for r in self.rows], dtype=np.int64)
        self.ids = np.asarray([r['sample_id'] for r in self.rows])
        self.contexts = np.asarray([r['context_condition'] for r in self.rows])
        weights = np.zeros(len(self.rows), dtype=np.float64)
        for mask, mass in [(self.counts == 0, .1)] + [(self.labels == k, .1) for k in (1, 2, 3)]:
            if mask.any():
                weights[mask] = mass / int(mask.sum())
        for k in (4, 5, 6):
            mask = self.labels == k
            if mask.any():
                weights[mask] = .2 / int(mask.sum())
        self.weights = (weights / weights.sum()).astype(np.float32)

    def __len__(self):
        return len(self.rows)

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


def training_views(data, indices, epoch, probabilities, seed):
    indices = np.asarray(indices)
    choices = np.array([np.random.default_rng(np.random.SeedSequence(
        [seed, 100, epoch, int(i)])).choice(6, p=probabilities) for i in indices])
    waves = data.batch(indices)
    for j, (i, condition) in enumerate(zip(indices, choices)):
        if condition:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 101, epoch, int(i)]))
            waves[j] = add_noise(waves[j], data.counts[i], SNRS[condition - 1], rng)
    return waves, choices


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


def recording_results(data, predictions):
    groups = {}
    for i, row in enumerate(data.rows):
        for side in (1, 2):
            key = row.get(f'source{side}_recording_id', '')
            if key:
                item = groups.setdefault(key, dict(class_name=row.get(f'source{side}_class_name', ''), indices=set()))
                item['indices'].add(i)
    correct = predictions == data.labels
    count_correct = SET_COUNTS[predictions] == data.counts
    result = []
    for key, item in sorted(groups.items()):
        mask = np.zeros(len(data), dtype=bool)
        mask[list(item['indices'])] = True
        row = dict(recording_id=key, class_name=item['class_name'], n=int(mask.sum()),
            acc=_accuracy(correct, mask), final_count_acc=_accuracy(count_correct, mask))
        for name, count in (('single', 1), ('double', 2), ('noise', 0)):
            subset = mask & (data.counts == count)
            row[name + '_n'], row[name + '_acc'] = int(subset.sum()), _accuracy(correct, subset)
        result.append(row)
    return result
