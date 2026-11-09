"""Screen temporal path scores on a frozen GN-CAT backbone with a matched control."""

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
RESEARCH = HERE.parents[3] / 'final' / 'data_gen_v8_single_physical' / 'research'
FROZEN = RESEARCH / 'shipsear_shortlocal_gn_freeze_20261108' / 'snapshot' / 'shipsear_finalization_20261103'
CAT_RUN = RESEARCH / 'shipsear_finalization_20261103' / 'runs' / 'SE-ABL-ShortLocal-GN-CAT-13'
sys.path.insert(0, str(FROZEN))
from train_formal import (CONDITIONS, FixedDataset, ShortLocalSetModel, evaluate, feature_cache,
                          model_state, now, ranking, save_state, train_step, training_features, write_json)
from short_local import ShortLocalViews
from temporal_paths import FrozenCatFeatures, TemporalPaths


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(out, event, **details):
    row = dict(time=now(), event=event, **details)
    write_json(out / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def create_head(config, arm):
    torch.manual_seed(config['seed'])
    return TemporalPaths(config, config['temporal_paths']['change_cost'][arm]).cuda()


def make_frozen(config):
    saved = torch.load(CAT_RUN / 'best.pt', map_location='cpu', weights_only=False)
    seal = load_json(CAT_RUN / 'selection_sealed.json')
    if (saved['experiment_id'], saved['epoch'], saved['score']) != (
            config['reference_id'], seal['epoch'], seal['score']):
        raise ValueError('The GN-CAT checkpoint differs from the archived selection.')
    if saved['config']['short_local_fusion'] != 'concat':
        raise ValueError('The reference must use the standard TabM first layer.')
    cat = ShortLocalSetModel(saved['config'], saved['normalization'])
    cat.load_state_dict(saved['model'], strict=True)
    return FrozenCatFeatures(ShortLocalViews(config), cat,
                             config['spectral_scattering']['summary_grid']).cuda().eval().requires_grad_(False)


def context_cache(frozen, dataset, config, condition='original'):
    raw = feature_cache(frozen.front, dataset, config, condition)
    parts = []
    batch_size = config['evaluation_batch_size']
    for start in range(0, len(dataset), batch_size):
        features = (value[start:start + batch_size].cuda() for value in raw)
        parts.append(tuple(value.cpu() for value in frozen.encode(*features)))
    return tuple(torch.cat(values) for values in zip(*parts))


def check_reference(data, cache):
    for condition, features in cache.items():
        predicted = features[-1].cuda().float().softmax(-1).cpu().numpy().mean(1)
        with np.load(CAT_RUN / f'{data.split}_{condition}.npz') as reference:
            for field in ('ids', 'labels', 'contexts'):
                np.testing.assert_array_equal(getattr(data, field), reference[field])
            np.testing.assert_array_equal(predicted.argmax(axis=1), reference['predictions'])
            np.testing.assert_allclose(predicted, reference['probabilities'], rtol=1e-5, atol=1e-6)


def smoke(config, frozen):
    val = FixedDataset(config['data_root'], 'Val')
    ids = np.arange(config['evaluation_batch_size'])
    # Match the original fixed extractor's batch size and the readout's batch size.
    raw = [frozen.front(val.batch(part)) for part in np.split(ids, len(ids) // config['batch_size'])]
    features = frozen.encode(*(torch.cat(values) for values in zip(*raw)))
    with np.load(CAT_RUN / 'Val_original.npz') as reference:
        probabilities = features[-1].float().softmax(-1).cpu().numpy().mean(1)
        np.testing.assert_array_equal(val.ids[ids], reference['ids'][ids])
        np.testing.assert_allclose(probabilities, reference['probabilities'][ids], rtol=1e-5, atol=1e-6)

    model = create_head(config, 'transition').eval()
    if model.paths.group_sizes != (1, 15, 15, 15, 225, 225, 225):
        raise RuntimeError('The path groups do not cover the seven legal unions.')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(model(*features), features[-1].float(), atol=1e-6, rtol=0)

    independent = create_head(config, 'independent').eval()
    toy = torch.zeros(2, 1, 4, 7, device='cuda')
    toy[0, 0, :2, 1] = toy[0, 0, 2:, 2] = 4.
    toy[1, 0, 0::2, 1] = toy[1, 0, 1::2, 2] = 4.
    without_order = independent.paths(toy)[:, 0, 4]
    with_order = model.paths(toy)[:, 0, 4]
    torch.testing.assert_close(without_order[0], without_order[1])
    if not with_order[0] > with_order[1]:
        raise RuntimeError('The transition prior did not favor the sustained evidence path.')

    train_data = FixedDataset(config['data_root'], 'Train')
    train_ids = np.arange(config['batch_size'])
    train_features = frozen(train_data.batch(train_ids))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'])
    before = model.state_scores[-1].weight.detach().clone()
    loss = train_step(model, optimizer, train_features, train_data, train_ids, config, 1, 0)
    if torch.equal(before, model.state_scores[-1].weight.detach()):
        raise RuntimeError('No gradient update reached the state scorer.')
    print(json.dumps(dict(groups=model.paths.group_sizes, state_inputs=[list(value.shape) for value in features],
                          prior_example=with_order.tolist(), parameters=sum(p.numel() for p in model.parameters()),
                          **loss)), flush=True)


def train(config, out, frozen):
    train_data = FixedDataset(config['data_root'], 'Train')
    val = FixedDataset(config['data_root'], 'Val')
    arms = tuple(config['temporal_paths']['change_cost'])
    models = {arm: create_head(config, arm) for arm in arms}
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'],
                                         weight_decay=config['weight_decay']) for arm, model in models.items()}
    history, best = [], {}
    resume_path = out / 'resume.pt'
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise ValueError('Stored training configuration differs.')
        history, best = saved['history'], saved['best']
        for arm in arms:
            models[arm].load_state_dict(saved['models'][arm], strict=True)
            optimizers[arm].load_state_dict(saved['optimizers'][arm])

    status(out, 'preparing', train=len(train_data), val=len(val))
    originals = context_cache(frozen, train_data, config)
    val_cache = {condition: context_cache(frozen, val, config, condition) for condition in CONDITIONS}
    check_reference(val, val_cache)
    status(out, 'training_started', next_epoch=len(history) + 1, reference_replay='all_val_views')
    for epoch in range(len(history) + 1, config['temporal_paths']['epochs'] + 1):
        start = time.perf_counter()
        order = np.random.default_rng([config['seed'], 700, epoch]).permutation(len(train_data))
        counts = np.zeros(len(CONDITIONS), dtype=np.int64)
        losses = dict.fromkeys(arms, 0.)
        for index, offset in enumerate(range(0, len(order), config['batch_size'])):
            ids = order[offset:offset + config['batch_size']]
            features, choices = training_features(frozen, train_data, originals, config, ids, epoch)
            counts += np.bincount(choices, minlength=len(CONDITIONS))
            for arm in arms:
                metrics = train_step(models[arm], optimizers[arm], features, train_data, ids, config, epoch, index)
                losses[arm] += metrics['loss'] * len(ids) / len(train_data)

        current = {arm: model_state(model) for arm, model in models.items()}
        epoch_results = {}
        for arm, model in models.items():
            metrics = evaluate(model, val, val_cache, config)
            score = ranking(metrics)
            if arm not in best or score > best[arm]['score']:
                best[arm] = dict(epoch=epoch, score=score, model=current[arm])
            epoch_results[arm] = dict(loss=losses[arm], Val=metrics, score=score, best_epoch=best[arm]['epoch'])
        seconds = time.perf_counter() - start
        history.append(dict(epoch=epoch, arms=epoch_results, seconds=seconds,
                            condition_counts=dict(zip(CONDITIONS, counts.tolist()))))
        save_state(resume_path, dict(config=config, history=history, best=best, models=current,
                                     optimizers={arm: opt.state_dict() for arm, opt in optimizers.items()}))
        write_json(out / 'history.json', history)
        status(out, 'epoch_finished', epoch=epoch, seconds=round(seconds, 2),
               arms={arm: dict(best=best[arm]['epoch'], score=epoch_results[arm]['score']) for arm in arms})

    for arm, model in models.items():
        directory = out / arm
        directory.mkdir(exist_ok=True)
        picked = best[arm]
        model.load_state_dict(picked['model'], strict=True)
        metrics = evaluate(model, val, val_cache, config, directory)
        if ranking(metrics) != picked['score']:
            raise RuntimeError('Restored path weights did not reproduce the Val selection.')
        save_state(directory / 'best.pt', dict(config=config, arm=arm, **picked))
        write_json(directory / 'selection_sealed.json', dict(arm=arm, epoch=picked['epoch'],
                   score=picked['score'], epochs_trained=len(history), sealed_at=now(), Val=metrics,
                   parameters=sum(p.numel() for p in model.parameters())))
    status(out, 'selection_sealed', epochs={arm: best[arm]['epoch'] for arm in arms})


def test(config, out, frozen):
    selections = {arm: load_json(out / arm / 'selection_sealed.json')
                  for arm in config['temporal_paths']['change_cost']}
    status(out, 'test_started', epochs={arm: selection['epoch'] for arm, selection in selections.items()})
    data = FixedDataset(config['data_root'], 'Test')
    cache = {condition: context_cache(frozen, data, config, condition) for condition in CONDITIONS}
    check_reference(data, cache)
    results = {}
    for arm, selection in selections.items():
        directory = out / arm
        saved = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['arm'] != arm or (saved['epoch'], saved['score']) != (
                selection['epoch'], selection['score']):
            raise RuntimeError('Checkpoint differs from its sealed selection.')
        model = create_head(config, arm)
        model.load_state_dict(saved['model'], strict=True)
        metrics = evaluate(model, data, cache, config, directory)
        results[arm] = dict(selection=selection, Test=metrics)
    write_json(out / 'results.json', dict(config=config, arms=results))
    status(out, 'completed', test_acc={arm: {c: row['Test'][c]['acc'] for c in CONDITIONS}
                                      for arm, row in results.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--smoke', action='store_true', help='Check CAT replay, path normalization and gradients.')
    actions.add_argument('--resume', action='store_true', help='Resume a partially completed trial.')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The frozen protocol requires CUDA.')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = load_json(CAT_RUN / 'config_resolved.json')
    config['reference_id'] = config['experiment_id']
    config['reference_only'] = {'GN-CAT': str(CAT_RUN)}
    config['experiment_id'] = f'SE-TemporalPaths-CAT-{config["seed"]}'
    config['initialization'] = 'frozen_gn_cat_with_zero_output_state_scorer'
    config['temporal_paths'] = dict(width=128, epochs=20,
                                   change_cost={'transition': math.log(2), 'independent': 0.})
    frozen = make_frozen(config)
    if args.smoke:
        smoke(config, frozen)
        return

    out = HERE / 'runs' / config['experiment_id']
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        config_path = out / 'config_resolved.json'
        if config_path.is_file() and load_json(config_path) != config:
            raise ValueError('Stored experiment configuration differs.')
        if (out / 'results.json').is_file():
            print(f'Completed results already exist at {out}.', flush=True)
            return
        write_json(config_path, config)
        if not all((out / arm / 'selection_sealed.json').is_file()
                   for arm in config['temporal_paths']['change_cost']):
            train(config, out, frozen)
        test(config, out, frozen)


if __name__ == '__main__':
    main()
