"""Read one arrangement index; return only waveforms and task labels."""
from pathlib import Path
import numpy as np
from common import read, rows, normalize, SPLITS
from generate import single_envelope


class MixedDataset:
    def __init__(self, root, split, augment=None):
        self.root = Path(root)
        assert split in SPLITS
        assert not (self.root/'.generating').exists(), 'Generation is incomplete.'
        self.split = split
        self.cfg = read(self.root/'CONFIG.json')
        self.augment = (split == 'Train' and self.cfg['single_train_augmentation']['enabled']) if augment is None else bool(augment)
        if split != 'Train' and self.augment:
            raise ValueError('Single envelope augmentation is Train-only.')
        index = self.root/'DATA_INDEX.csv'
        self.records = [r for r in rows(index) if r['split'] == split]
        assert [int(r['array_index']) for r in self.records] == list(range(len(self.records)))
        path = self.root/'data'/split/'audio.npy'
        self.audio = np.load(path, mmap_mode='r')
        n = self.cfg['sample_rate']*self.cfg['duration_seconds']
        assert self.audio.shape == (len(self.records), n) and self.audio.dtype == np.float32
        self.target = np.asarray([[int(r[f'target_{c}']) for c in self.cfg['classes']] for r in self.records], dtype=np.float32)
        self.label = np.asarray([int(r['label_index']) for r in self.records], dtype=np.int64)
        self.count = self.target.sum(1).astype(np.int64)
        # Cheap cache invalidation hint, not an integrity check or a file scan.
        self.cache_fingerprint = f"index-v1:{self.cfg['version']}:{split}:{int(self.augment)}:{index.stat().st_mtime_ns}:{path.stat().st_mtime_ns}"

    def __len__(self):
        return len(self.records)

    def batch(self, indices):
        indices = np.atleast_1d(indices).astype(int)
        audio = np.array(self.audio[indices], copy=True)
        assert np.isfinite(audio).all(), 'Non-finite audio in batch.'
        if self.augment:
            for j, i in enumerate(indices):
                r = self.records[int(i)]
                if r['kind'] == 'single' and r['augmentation_type'] != 'flat':
                    audio[j] = normalize(audio[j].astype(np.float64)*single_envelope(r, self.cfg), self.cfg['output_rms'])
        return audio, self.target[indices], self.label[indices], self.count[indices]
