"""Small I/O helpers and stable random substreams; no file hash validation."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('Train', 'Val', 'Test')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def rng(seed, *keys):
    # Only derives the same random substream seeds as before; never hashes files.
    payload = json.dumps([seed, *keys], sort_keys=True, ensure_ascii=False).encode('utf-8')
    return np.random.default_rng(int(hashlib.sha256(payload).hexdigest()[:16], 16))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def normalize(x, level):
    value = rms(x)
    assert np.isfinite(value) and value > 0, 'Invalid or silent audio.'
    return (np.asarray(x, dtype=np.float64)*(level/value)).astype(np.float32)


def config(path=None):
    cfg = read(path or ROOT/'config.json')
    # This component bank and MATLAB preprocessing use 5-second, 16-kHz cores.
    assert cfg['sample_rate'] == 16000 and cfg['duration_seconds'] == 5
    assert len(cfg['classes']) == len(set(cfg['classes'])) == 3
    for weights in (cfg['approximate_type_ratio'], cfg['context_ratio']):
        values = np.asarray(list(weights.values()), dtype=float)
        assert np.isfinite(values).all() and (values >= 0).all() and values.sum() > 0
    assert cfg['approximate_type_ratio']['single'] > 0
    aug = cfg['single_train_augmentation']
    assert all(np.isfinite(aug[k]) and aug[k] >= 0 for k in ('flat','edge','middle'))
    assert sum(aug[k] for k in ('flat','edge','middle')) > 0
    for key in ('context_seconds','fade_seconds'):
        lo, hi = cfg[key]
        assert 0 < lo <= hi < cfg['duration_seconds']
    assert 2*(max(cfg['context_seconds'])+max(cfg['fade_seconds'])) < cfg['duration_seconds']
    assert np.isfinite(cfg['sir_db']) and 0 < cfg['epsilon'] <= 1 and cfg['output_rms'] > 0
    assert not cfg['additional_noise'], 'Additional noise injection is not implemented.'
    dest = (ROOT/cfg['output_directory']).resolve()
    assert dest.is_relative_to(ROOT/'datasets'), 'Output must stay inside datasets/.'
    return cfg, dest
