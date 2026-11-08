"""Replay both archived splits and verify the sealed Q0 probabilities."""

import argparse
import filecmp
import json
from pathlib import Path

import numpy as np

from data import CONDITIONS, FixedDataset, evaluation_views, recognition_metrics
from model import Recognizer


HERE = Path(__file__).resolve().parent
RECORDS = HERE / 'records'


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Location of the frozen MB-swap data.')
    args = parser.parse_args()
    config = load_json(HERE / 'config.json')
    data_root = args.data_root or Path(config['data_root'])
    if not filecmp.cmp(data_root / 'DATA_INDEX.csv', RECORDS / 'dataset/DATA_INDEX.csv', shallow=False):
        raise ValueError('Waveform index differs from the archived MB-swap index.')

    model = Recognizer(HERE / 'weights/best.pt')
    results = load_json(RECORDS / 'results.json')
    if config != model.config or config != results['config']:
        raise ValueError('Checkpoint, config and results disagree.')

    replay = {}
    for split in ('Val', 'Test'):
        data = FixedDataset(data_root, split)
        indices = np.arange(len(data))
        for condition in CONDITIONS:
            waves = evaluation_views(data, indices, condition, config['evaluation_noise_seed'])
            output = model.predict(waves)
            key = f'{split}_{condition}'
            with np.load(RECORDS / f'{key}.npz') as archived:
                for actual, field in ((data.ids, 'ids'), (data.labels, 'labels'), (data.contexts, 'contexts'),
                                      (output['predictions'], 'predictions'), (output['probabilities'], 'probabilities')):
                    np.testing.assert_array_equal(actual, archived[field], err_msg=f'{key} {field}')
            metrics = recognition_metrics(data.labels, output['predictions'], data.counts, data.contexts)
            if metrics != results[split][condition]:
                raise ValueError(f'{key} metrics differ from the sealed result.')
            replay[key] = {'n': len(data), 'acc': metrics['acc'], 'probability_max_abs_error': 0.0}
            print(f'{key}: {len(data)} exact probability and prediction matches', flush=True)

    (RECORDS / 'release_verification.json').write_text(
        json.dumps(replay, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
