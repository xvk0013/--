"""Replay the retained waveform inference entry against the archived Val outputs."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'shipsear_final_noise_distribution_20261102'
RUN = PACKAGE / 'runs' / 'seed13'
sys.path.insert(0, str(PACKAGE))
from data import CONDITIONS, SET_COUNTS, SET_NAMES, FixedDataset, evaluation_views
from model import Recognizer


def choose_rows(labels, batch_size):
    """Cover each class while preserving the evaluation batch boundaries."""
    blocks = [np.arange(start, min(start + batch_size, len(labels)))
              for start in range(0, len(labels), batch_size)]
    missing = set(range(len(SET_NAMES)))
    selected = []
    while missing:
        block = max(blocks, key=lambda ids: len(set(labels[ids]) & missing))
        covered = set(labels[block]) & missing
        if not covered:
            raise ValueError(f'Val lacks these classes: {sorted(missing)}')
        selected.append(block)
        missing -= covered
    return np.concatenate(sorted(selected, key=lambda ids: ids[0]))


def compare(output, archive, ids):
    expected = archive['probabilities'][ids]
    np.testing.assert_array_equal(output['predictions'], archive['predictions'][ids])
    np.testing.assert_array_equal(output['counts'], SET_COUNTS[output['predictions']])
    np.testing.assert_array_equal(output['sets'], np.asarray(SET_NAMES)[output['predictions']])
    np.testing.assert_allclose(output['probabilities'], expected, rtol=1e-6, atol=1e-7)
    return float(np.max(np.abs(output['probabilities'] - expected)))


def main():
    torch.set_num_threads(4)
    config = json.loads((PACKAGE / 'config.json').read_text(encoding='utf-8'))
    split_sizes = {name: len(FixedDataset(config['data_root'], name))
                   for name in ('Train', 'Val', 'Test')}
    val = FixedDataset(config['data_root'], 'Val')
    ids = choose_rows(val.labels, config['evaluation_batch_size'])
    report = {
        'verified_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'environment': {'python': sys.executable, 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'cuda': torch.version.cuda,
                        'device': torch.cuda.get_device_name()},
        'data_root': config['data_root'], 'split_sizes': split_sizes,
        'split': 'Val', 'evaluation_seed': config['seed'],
        'array_indices': ids.tolist(), 'sample_ids': val.ids[ids].tolist(),
        'class_counts': dict(zip(SET_NAMES, np.bincount(val.labels[ids], minlength=len(SET_NAMES)).tolist())),
        'sample_count_per_condition': int(len(ids)), 'checks': {},
    }
    baselines = {
        'P50': (RUN / 'P50_noise50_best.pt', RUN, 'P50_noise50'),
        'V1': (PACKAGE / 'weights' / 'V1_reference.pt', PACKAGE / 'reference', 'V1_stored_single'),
    }
    for name, (checkpoint, archive_dir, arm) in baselines.items():
        metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)
        recognizer = Recognizer(checkpoint, 'cuda')
        row = {'checkpoint': str(checkpoint), 'arm': metadata['arm'],
               'checkpoint_epoch': metadata['epoch'], 'global_epoch': metadata['global_epoch'],
               'trainable_parameters_in_architecture': sum(p.numel() for p in recognizer.model.parameters()),
               'conditions': {}}
        for condition in CONDITIONS:
            waves = evaluation_views(val, ids, condition, config['seed'])
            path = archive_dir / f'{arm}_Val_{condition}.npz'
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive['ids'][ids], val.ids[ids])
                error = compare(recognizer.predict(waves), archive, ids)
            row['conditions'][condition] = {'archive': str(path), 'changed_predictions': 0,
                                            'max_absolute_probability_error': error}
        report['checks'][name] = row
        print(f'{name}: all six Val views reproduced for {len(ids)} samples', flush=True)
    waves = evaluation_views(val, ids, '-10dB', config['seed'])
    with tempfile.TemporaryDirectory(prefix='waveform_replay_', dir=HERE) as folder:
        sample_path, output_path = Path(folder) / 'input.npy', Path(folder) / 'predictions.npz'
        np.save(sample_path, waves, allow_pickle=False)
        command = [sys.executable, str(PACKAGE / 'predict.py'), '--checkpoint', str(baselines['P50'][0]),
                   '--input', str(sample_path), '--output', str(output_path), '--device', 'cuda']
        cli = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        with np.load(output_path) as result, np.load(RUN / 'P50_noise50_Val_-10dB.npz') as archive:
            error = compare(result, archive, ids)
        report['cli'] = {'condition': '-10dB', 'entry': str(PACKAGE / 'predict.py'),
                         'stdout': cli.stdout.strip(), 'changed_predictions': 0,
                         'max_absolute_probability_error': error}
    (HERE / '验收回放.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Standalone prediction entry reproduced the archived P50 -10 dB Val outputs.', flush=True)


if __name__ == '__main__':
    main()
