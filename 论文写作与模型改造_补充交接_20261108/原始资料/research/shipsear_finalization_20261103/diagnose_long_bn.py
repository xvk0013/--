"""Check long-window normalization on Train and Val, then choose one L4 arm."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from train_formal import (CONDITIONS, FixedDataset, create_model, evaluate,
                          feature_cache, model_state, ranking, save_state, write_json)
from short_local import ShortLocalViews
from long_bn_calibration import calibration_cache, recalibrate_long_bn


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'runs/SE-Q0-ShortLocal-L3-13'
OUT = HERE / 'runs/SE-Q0-LongBN-Diagnostic-13'


def mean_metric(metrics, key, conditions=CONDITIONS):
    return float(np.mean([metrics[condition][key] for condition in conditions]))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').is_file():
        print((OUT / 'results.json').read_text(encoding='utf-8'), flush=True)
        return
    saved = torch.load(SOURCE / 'best.pt', map_location='cpu', weights_only=False)
    config = saved['config']
    train = FixedDataset(config['data_root'], 'Train')
    val = FixedDataset(config['data_root'], 'Val')
    fixed = ShortLocalViews(config).cuda().eval()
    images, counts = calibration_cache(fixed, train, config)
    cache = {condition: feature_cache(fixed, val, config, condition) for condition in CONDITIONS}
    model, optimizer = create_model(config, saved['normalization'])
    del optimizer
    model.load_state_dict(saved['model'], strict=True)
    before_dir, after_dir = OUT / 'before', OUT / 'after'
    before_dir.mkdir(exist_ok=True)
    after_dir.mkdir(exist_ok=True)
    before = evaluate(model, val, cache, config, before_dir)
    reference = json.loads((SOURCE / 'results.json').read_text(encoding='utf-8'))
    if before != reference['Val']:
        raise RuntimeError('The source L3 validation metrics did not replay.')
    for condition in CONDITIONS:
        with np.load(SOURCE / f'Val_{condition}.npz') as old:
            with np.load(before_dir / f'Val_{condition}.npz') as replayed:
                for key in ('ids', 'labels', 'contexts', 'predictions'):
                    np.testing.assert_array_equal(old[key], replayed[key])
    parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    changes = recalibrate_long_bn(model, images, config['batch_size'])
    if any(not torch.equal(value.detach().cpu(), parameters[name])
           for name, value in model.named_parameters()):
        raise RuntimeError('Calibration changed learned parameters.')
    for name, value in model.named_buffers():
        changed_stat = name.startswith('long_local.') and name.endswith(('.running_mean', '.running_var'))
        if not changed_stat and not torch.equal(value.cpu(), buffers[name]):
            raise RuntimeError(f'Calibration changed an unrelated buffer {name}.')
    after = evaluate(model, val, cache, config, after_dir)
    background_gain = (mean_metric(after, 'noise_acc', ('10dB', '5dB', '0dB'))
                       - mean_metric(before, 'noise_acc', ('10dB', '5dB', '0dB')))
    single_change = mean_metric(after, 'single_acc') - mean_metric(before, 'single_acc')
    double_change = mean_metric(after, 'double_acc') - mean_metric(before, 'double_acc')
    passed = background_gain >= 5. and single_change >= -1. and double_change >= -1.
    method = 'precise' if passed else 'record'
    result = {
        'experiment_id': OUT.name, 'operation': 'Train-only layerwise long BatchNorm recalibration',
        'source_checkpoint': str(SOURCE / 'best.pt'), 'source_selected_epoch': saved['epoch'],
        'optimizer_updates': 0, 'learned_parameters_unchanged': True,
        'unrelated_buffers_unchanged': True, 'calibration_split': 'Train',
        'calibration_view_epoch': 1001, 'calibration_condition_counts': counts,
        'calibration_layers': changes, 'Val_before': before, 'Val_after': after,
        'rank_before': ranking(before), 'rank_after': ranking(after),
        'gate': {'background_gain_min_pp': 5., 'single_mean_loss_max_pp': 1.,
                 'double_mean_loss_max_pp': 1., 'background_gain_pp': background_gain,
                 'single_change_pp': single_change, 'double_change_pp': double_change,
                 'passed': passed, 'selected_full_training_method': method},
    }
    save_state(OUT / 'calibrated.pt', {
        **saved, 'experiment_id': OUT.name, 'model': model_state(model),
        'diagnostic_only': True, 'operation': result['operation'],
    })
    write_json(OUT / 'results.json', result)
    print(json.dumps({'event': 'diagnostic_completed', 'gate': result['gate']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
