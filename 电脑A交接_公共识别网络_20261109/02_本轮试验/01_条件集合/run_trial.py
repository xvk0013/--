"""Train the conditional set model using the frozen ShortLocal-GN protocol."""

import argparse
import json
import math
import msvcrt
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FROZEN = (HERE.parents[3] / 'final' / 'data_gen_v8_single_physical' / 'research'
          / 'shipsear_shortlocal_gn_freeze_20261108' / 'snapshot' / 'shipsear_finalization_20261103')
REFERENCE = FROZEN / 'runs' / 'SE-Q0-ShortLocal-GN-13'
sys.path.insert(0, str(FROZEN))
from train_formal import (CONDITIONS, FixedDataset, evaluate, feature_cache, fit_normalization,
                          model_state, now, ranking, resume_state, save_state, train_step,
                          training_features, write_json)
from short_local import ShortLocalViews
from conditional_set import ConditionalSetModel, ConditionalSetReadout


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(out, event, **details):
    row = dict(time=now(), event=event, **details)
    write_json(out / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def create_model(config, normalization):
    torch.manual_seed(config['seed'])
    return ConditionalSetModel(config, normalization).cuda()


def smoke(config):
    # With PF as the anchor, the local weights can reverse the MB/RV order.
    example = torch.tensor([[[10., 0., math.log(6), 0.], [-10., math.log(5), 0., 0.]]])
    local = ConditionalSetReadout('local', config['model']['strength']).continuation_scores(example)
    uniform = ConditionalSetReadout('uniform', config['model']['strength']).continuation_scores(example)
    if not (local[0, 0, 1] > local[0, 0, 2] and uniform[0, 0, 1] < uniform[0, 0, 2]):
        raise RuntimeError('The controlled evidence example does not distinguish local and uniform discount.')

    data = FixedDataset(config['data_root'], 'Train')
    ids = np.random.default_rng(config['seed']).permutation(len(data))[:config['batch_size']]
    fixed = ShortLocalViews(config).cuda().eval()
    features = fixed(data.batch(ids))
    normalization = fit_normalization(data, [value.cpu() for value in features[:2]], config)
    model = create_model(config, normalization)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        log_probabilities = model(*features)
    if log_probabilities.shape != (len(ids), 1, 7) or not log_probabilities.isfinite().all():
        raise RuntimeError('Unexpected seven-set output.')
    torch.testing.assert_close(log_probabilities.exp().sum(dim=-1), torch.ones_like(log_probabilities[..., 0]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'], weight_decay=config['weight_decay'])
    before = model.score[-1].weight.detach().clone()
    losses = train_step(model, optimizer, features, data, ids, config, 1, 0)
    if torch.equal(before, model.score[-1].weight.detach()):
        raise RuntimeError('The evidence scorer did not update on a real batch.')
    print(json.dumps(dict(output_shape=list(log_probabilities.shape),
                          parameters=sum(p.numel() for p in model.parameters()),
                          pf_anchor_mb_rv_local=local[0, 0, 1:].tolist(),
                          pf_anchor_mb_rv_uniform=uniform[0, 0, 1:].tolist(), **losses)), flush=True)


def train(config, out, fixed):
    train_data = FixedDataset(config['data_root'], 'Train')
    val_data = FixedDataset(config['data_root'], 'Val')
    if [len(train_data), len(val_data)] != [config['split_sizes'][split] for split in ('Train', 'Val')]:
        raise ValueError('Train or Val does not match the frozen data protocol.')
    status(out, 'preparing', train=len(train_data), val=len(val_data))
    originals = feature_cache(fixed, train_data, config)
    normalization = fit_normalization(train_data, originals[:2], config)
    val_cache = {condition: feature_cache(fixed, val_data, config, condition) for condition in CONDITIONS}

    model = create_model(config, normalization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'], weight_decay=config['weight_decay'])
    history, best, clock, elapsed, stop_reason = [], None, None, 0., None
    resume_path = out / 'resume.pt'
    if resume_path.is_file():
        saved = resume_state(resume_path, config, model, optimizer)
        for name in ('mean', 'scale'):
            torch.testing.assert_close(saved['normalization']['F1'][name], normalization['F1'][name], rtol=0, atol=0)
        history, best, clock = saved['history'], saved['best'], saved['clock']
        elapsed, stop_reason = saved['elapsed_seconds'], saved['stop_reason']
        del saved

    status(out, 'training_started', next_epoch=len(history) + 1)
    while stop_reason is None:
        begun = time.perf_counter()
        epoch = len(history) + 1
        order = np.random.default_rng([config['seed'], 700, epoch]).permutation(len(train_data))
        counts = np.zeros(len(CONDITIONS), dtype=np.int64)
        losses = {'loss': 0., 'set_loss': 0., 'count_loss': 0.}
        for index, start in enumerate(range(0, len(order), config['batch_size'])):
            ids = order[start:start + config['batch_size']]
            features, choices = training_features(fixed, train_data, originals, config, ids, epoch)
            counts += np.bincount(choices, minlength=len(CONDITIONS))
            for name, value in train_step(model, optimizer, features, train_data, ids, config, epoch, index).items():
                losses[name] += value * len(ids) / len(train_data)

        metrics = evaluate(model, val_data, val_cache, config)
        score = ranking(metrics)
        state = model_state(model)
        if best is None or score > best['score']:
            best = dict(config=config, epoch=epoch, score=score, normalization=normalization, model=state)
        if clock is None or score[0] >= clock['anchor'] + config['early_stop_min_delta_pp'] - 1e-10:
            clock = dict(anchor=score[0], last_epoch=epoch)
        if epoch >= config['max_epochs']:
            stop_reason = 'max_epochs'
        elif epoch >= config['min_epochs'] and epoch - clock['last_epoch'] >= config['patience_epochs']:
            stop_reason = 'patience'

        seconds = time.perf_counter() - begun
        elapsed += seconds
        history.append(dict(epoch=epoch, **losses, condition_counts=dict(zip(CONDITIONS, counts.tolist())),
                            Val=metrics, score=score, best_epoch=best['epoch'], seconds=seconds))
        save_state(resume_path, dict(config=config, model=state, optimizer=optimizer.state_dict(),
                                     normalization=normalization, history=history, best=best, clock=clock,
                                     elapsed_seconds=elapsed, stop_reason=stop_reason))
        write_json(out / 'history.json', history)
        status(out, 'epoch_finished', epoch=epoch, best_epoch=best['epoch'], score=score,
               seconds=round(seconds, 2), stop_reason=stop_reason)

    save_state(out / 'best.pt', best)
    write_json(out / 'history.json', history)
    model.load_state_dict(best['model'], strict=True)
    val_metrics = evaluate(model, val_data, val_cache, config, out)
    if ranking(val_metrics) != best['score']:
        raise RuntimeError('Restored weights did not reproduce the selected Val score.')
    selection = dict(experiment_id=config['experiment_id'], epoch=best['epoch'], score=best['score'],
                     sealed_at=now(), epochs_trained=len(history), elapsed_seconds=elapsed,
                     stop_reason=stop_reason, parameters=sum(p.numel() for p in model.parameters()), Val=val_metrics)
    write_json(out / 'selection_sealed.json', selection)
    status(out, 'selection_sealed', epoch=best['epoch'], score=best['score'])


def run(config, out):
    fixed = ShortLocalViews(config).cuda().eval()
    selection_path = out / 'selection_sealed.json'
    if not selection_path.is_file():
        train(config, out, fixed)
    selection = load_json(selection_path)
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    if saved['config'] != config or (saved['epoch'], saved['score']) != (selection['epoch'], selection['score']):
        raise RuntimeError('Checkpoint differs from the sealed selection.')
    model = create_model(config, saved['normalization'])
    model.load_state_dict(saved['model'], strict=True)
    status(out, 'test_started', epoch=selection['epoch'])
    test_data = FixedDataset(config['data_root'], 'Test')
    if len(test_data) != config['split_sizes']['Test']:
        raise ValueError('Test does not match the frozen data protocol.')
    test_cache = {condition: feature_cache(fixed, test_data, config, condition) for condition in CONDITIONS}
    metrics = evaluate(model, test_data, test_cache, config, out)
    result = dict(experiment_id=config['experiment_id'], config=config, selection=selection,
                  Val=selection['Val'], Test=metrics, parameters=selection['parameters'])
    write_json(out / 'results.json', result)
    status(out, 'completed', epoch=selection['epoch'], test_acc={c: metrics[c]['acc'] for c in CONDITIONS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('local', 'uniform'), default='local')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--smoke', action='store_true', help='Check the path probabilities and one real training batch.')
    actions.add_argument('--resume', action='store_true', help='Resume the selected run after an interruption.')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The frozen protocol requires CUDA.')
    config = load_json(REFERENCE / 'config_resolved.json')
    config['experiment_id'] = f'SE-ConditionalSet-{args.mode.title()}-{config["seed"]}'
    config['model'] = dict(width=256, dropout=.15, mode=args.mode, strength=.5)
    config['decision_arm'] = 'conditional_set'
    config['reference_only'] = {'ShortLocal-GN': str(REFERENCE)}
    config['training_seeds'] = [config['seed']]
    config['runtime'] = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                             numpy=np.__version__, python=sys.executable)
    if args.smoke:
        smoke(config)
        return

    out = HERE / 'runs' / config['experiment_id']
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        saved_config = out / 'config_resolved.json'
        if saved_config.is_file() and load_json(saved_config) != config:
            raise ValueError('Stored configuration differs from this run.')
        if (out / 'results.json').is_file():
            print(f'Completed results already exist at {out}.', flush=True)
            return
        write_json(saved_config, config)
        run(config, out)


if __name__ == '__main__':
    main()
