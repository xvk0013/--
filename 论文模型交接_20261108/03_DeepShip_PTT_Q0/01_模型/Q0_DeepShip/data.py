"""Adapt the screened DeepShip reader to the Q0 training interface."""

from pathlib import Path
import sys

import numpy as np

# Resolve the package first so the legacy Q0 directory link uses the same reader.
generator = Path(__file__).resolve().parents[2] / '02_数据生成与数据集'
sys.path.insert(0, str(generator))
from dataset import MixedDataset, recognition_metrics, set_schema


class Q0Dataset:
    def __init__(self, config, split):
        source = MixedDataset(config['data_root'], split)
        source_version = config.get('source_dataset_version', config['dataset_version'])
        self.class_names = tuple(config['class_names'])
        self.set_names, _, set_counts = set_schema(self.class_names)
        selected_classes = tuple(name for name in source.class_names if name in self.class_names)
        if (source.cfg['version'] != source_version
                or source.cfg['sample_rate'] != config['sample_rate']
                or source.cfg['duration_seconds'] != config['duration_seconds']
                or selected_classes != self.class_names
                or self.set_names != tuple(config['set_names'])
                or set_counts.tolist() != config['count_map']):
            raise ValueError(f'{split} does not match the configured DeepShip dataset.')
        if any(row['label_name'] != source.set_names[label] for row, label in zip(source.rows, source.labels)):
            raise ValueError(f'{split} has an index label whose name does not match the set schema.')
        self.source = source
        self.split = split
        target_labels = {name: i for i, name in enumerate(self.set_names)}
        mapping = np.asarray([target_labels.get(name, -1) for name in source.set_names])
        labels = mapping[source.labels]
        # Keep parent array rows so stored waveforms and noise seeds stay aligned.
        self.indices = np.flatnonzero(labels >= 0)
        if len(self.indices) != config['split_sizes'][split]:
            raise ValueError(f'{split} does not contain the configured number of samples.')
        if config['smoke']:
            self.indices = np.sort(np.concatenate([
                np.flatnonzero(labels == label)[:3] for label in range(len(self.set_names))
            ]))
        self.labels = labels[self.indices]
        self.counts = source.counts[self.indices]
        self.ids = source.ids[self.indices]
        self.contexts = source.contexts[self.indices]
        sizes = np.bincount(self.labels, minlength=len(self.set_names))
        if not np.all(sizes):
            raise ValueError(f'{split} must contain all configured target sets.')
        if split == 'Train':
            budgets = np.asarray(config['class_loss_budgets'])
            self.weights = ((budgets / budgets.sum()) / sizes)[self.labels].astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def batch(self, indices):
        waves, _, _, _ = self.source.batch(self.indices[indices], augment_single=False)
        return waves
