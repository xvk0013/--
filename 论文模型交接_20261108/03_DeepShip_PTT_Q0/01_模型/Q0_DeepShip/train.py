"""Train the DeepShip Q0 model from random initialization."""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from data import Q0Dataset, recognition_metrics
from dataset import set_schema
from model import DirectSetModel, FixedViews, configure_runtime
from noise_views import CONDITIONS, evaluation_views, training_views


HERE = Path(__file__).resolve().parent
COUNT_GROUP_SIZES = (1, 4, 6)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_state(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def feature_cache(fixed, dataset, config, condition='original'):
    parts = None
    for start in range(0, len(dataset), config['batch_size']):
        ids = np.arange(start, min(start + config['batch_size'], len(dataset)))
        waves = dataset.batch(ids) if condition == 'original' else evaluation_views(
            dataset, ids, condition, config['evaluation_noise_seed'])
        values = fixed(waves)
        if parts is None:
            parts = [[] for _ in values]
        for part, feature in zip(parts, values):
            part.append(feature.cpu())
    return tuple(torch.cat(part, dim=0) for part in parts)


def fit_normalization(train, features, config):
    if train.split != 'Train':
        raise ValueError('Feature normalization requires Train.')
    values = torch.cat(features, dim=1).double()
    mean = values.mean(0)
    scale = (values.square().mean(0) - mean.square()).clamp_min(0).sqrt()
    scale.clamp_min_(config['normalization']['std_floor'])
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError('Nonfinite Train feature normalization.')
    return {'F1': {'mean': mean.float(), 'scale': scale.float()}}


def training_features(fixed, train, originals, config, ids, epoch):
    waves, choices = training_views(train, ids, epoch, config['noise_probabilities'], config['seed'])
    features = tuple(values[ids].cuda() for values in originals)
    noisy = np.flatnonzero(choices)
    if len(noisy):
        for base, replacement in zip(features, fixed(waves[noisy])):
            base[noisy] = replacement
    return features, choices


def train_step(model, optimizer, features, data, ids, config, epoch, index, presence_membership=None):
    torch.manual_seed(config['seed'] + epoch * 100000 + index)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    y = torch.as_tensor(data.labels[ids], device='cuda')
    n = torch.as_tensor(data.counts[ids], device='cuda')
    weights = torch.as_tensor(data.weights[ids] * len(data), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(*features)
    log_probs = logits.float().log_softmax(-1)
    members = logits.shape[1]
    set_loss = -log_probs.gather(-1, y[:, None, None].expand(-1, members, 1)).squeeze(-1)
    probabilities = log_probs.exp()
    group_sizes = tuple(config['count_map'].count(n) for n in range(3))
    counts = torch.stack([group.sum(-1) for group in probabilities.split(group_sizes, dim=-1)], dim=-1)
    counts = counts.clamp_min(1e-12)
    count_loss = -counts.log().gather(-1, n[:, None, None].expand(-1, members, 1)).squeeze(-1)
    weighted_set = (set_loss.mean(1) * weights).mean()
    weighted_count = (count_loss.mean(1) * weights).mean()
    loss = weighted_set + config['count_marginal_weight'] * weighted_count
    weighted_presence = None
    if presence_membership is not None:
        # The log-odds are exactly the sum of set probabilities containing
        # the ship type versus the sum of the remaining set probabilities.
        by_class = log_probs.unsqueeze(-2)
        has_ship = presence_membership.T
        log_present = by_class.masked_fill(~has_ship, -torch.inf).logsumexp(-1)
        log_absent = by_class.masked_fill(has_ship, -torch.inf).logsumexp(-1)
        presence_logits = log_present - log_absent
        target = presence_membership[y].unsqueeze(1).expand_as(presence_logits)
        presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            presence_logits, target.to(presence_logits.dtype), reduction='none'
        )
        weighted_presence = (presence_loss.mean(dim=(1, 2)) * weights).mean()
        loss = loss + config['presence_marginal_weight'] * weighted_presence
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss.')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    metrics = {'loss': float(loss.detach()), 'set_loss': float(weighted_set.detach()),
               'count_loss': float(weighted_count.detach())}
    if weighted_presence is not None:
        metrics['presence_loss'] = float(weighted_presence.detach())
    return metrics


@torch.no_grad()
def evaluate(model, data, cache, config, out=None):
    model.eval()
    metrics = {}
    for condition, features in cache.items():
        probabilities = []
        for start in range(0, len(data), config['evaluation_batch_size']):
            batch = tuple(feature[start:start + config['evaluation_batch_size']].cuda() for feature in features)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(*batch)
            probabilities.append(logits.float().softmax(-1).cpu().numpy().mean(1))
        p = np.concatenate(probabilities)
        predictions = p.argmax(-1)
        metrics[condition] = recognition_metrics(data.labels, predictions, data.contexts, data.class_names)
        if out is not None:
            stem = f'{data.split}_{condition}'
            np.savez_compressed(out / (stem + '.npz'), ids=data.ids, labels=data.labels,
                                contexts=data.contexts, predictions=predictions, probabilities=p)
    return metrics


def ranking(metrics):
    single = float(np.mean([metrics[c]['single_acc'] for c in CONDITIONS]))
    double = float(np.mean([metrics[c]['double_acc'] for c in CONDITIONS]))
    return [(single + double) / 2, float(np.mean([metrics[c]['acc'] for c in CONDITIONS])),
            min(single, double), metrics['original']['acc']]


def log(out, event, **fields):
    status = {'time': now(), 'event': event, **fields}
    write_json(out / 'status.json', status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


def train(config, out, fixed, train_data, originals, val_data, val_cache, normalization, resume):
    torch.manual_seed(config['seed'])
    model = DirectSetModel(config, normalization).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'],
                                  weight_decay=config['weight_decay'])
    resume_path = out / 'resume.pt'
    history, best, clock, elapsed, stop_reason = [], None, None, 0.0, None
    if resume and resume_path.is_file():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('Resume checkpoint was produced under a different configuration.')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history, best, clock = saved['history'], saved['best'], saved['clock']
        elapsed, stop_reason = saved['elapsed_seconds'], saved['stop_reason']
        for key in ('mean', 'scale'):
            if not torch.equal(saved['model']['feature_' + key], normalization['F1'][key]):
                raise ValueError('Train normalization differs on resume.')
        del saved
    presence_membership = None
    if config.get('presence_marginal_weight', 0) > 0:
        _, membership, _ = set_schema(config['class_names'])
        presence_membership = torch.as_tensor(membership, dtype=torch.bool, device='cuda')
    log(out, 'training_started', next_epoch=len(history) + 1, rows=len(train_data))
    while stop_reason is None:
        begun = time.perf_counter()
        epoch = len(history) + 1
        order = np.random.default_rng([config['seed'], 700, epoch]).permutation(len(train_data))
        counts = np.zeros(len(CONDITIONS), dtype=np.int64)
        losses = {'loss': 0.0, 'set_loss': 0.0, 'count_loss': 0.0}
        for index, start in enumerate(range(0, len(order), config['batch_size'])):
            ids = order[start:start + config['batch_size']]
            features, choices = training_features(fixed, train_data, originals, config, ids, epoch)
            counts += np.bincount(choices, minlength=len(CONDITIONS))
            step = train_step(model, optimizer, features, train_data, ids, config, epoch, index, presence_membership)
            for key, loss in step.items():
                losses[key] = losses.get(key, 0.0) + loss * len(ids) / len(train_data)
        metrics = evaluate(model, val_data, val_cache, config)
        score = ranking(metrics)
        if best is None or score > best['score']:
            best = {'experiment_id': config['experiment_id'], 'config': config, 'epoch': epoch,
                    'score': score, 'normalization': normalization, 'model': model_state(model)}
        if clock is None or score[0] >= clock['anchor'] + config['early_stop_min_delta_pp'] - 1e-10:
            clock = {'anchor': score[0], 'last_epoch': epoch}
        if epoch >= config['max_epochs']:
            stop_reason = 'max_epochs'
        elif epoch >= config['min_epochs'] and epoch - clock['last_epoch'] >= config['patience_epochs']:
            stop_reason = 'patience'
        seconds = time.perf_counter() - begun
        elapsed += seconds
        history.append(dict(epoch=epoch, optimizer_updates=epoch * (index + 1), **losses,
                            condition_counts=dict(zip(CONDITIONS, counts.tolist())), Val=metrics,
                            score=score, best_epoch=best['epoch'], clock=clock, seconds=seconds,
                            stop_reason=stop_reason))
        save_state(resume_path, {'config': config, 'model': model_state(model),
                   'optimizer': optimizer.state_dict(), 'history': history, 'best': best,
                   'clock': clock, 'elapsed_seconds': elapsed, 'stop_reason': stop_reason})
        write_json(out / 'history.json', history)
        log(out, 'epoch_finished', epoch=epoch, best_epoch=best['epoch'], score=score, stop_reason=stop_reason)

    write_json(out / 'history.json', history)
    model.load_state_dict(best['model'], strict=True)
    val = evaluate(model, val_data, val_cache, config, out)
    if ranking(val) != best['score']:
        raise RuntimeError('Restored best checkpoint differs from its recorded Val rank.')
    selection = dict(experiment_id=config['experiment_id'], epoch=best['epoch'], score=best['score'],
                     sealed_at=now(), epochs_trained=len(history), stop_reason=stop_reason,
                     parameters=sum(parameter.numel() for parameter in model.parameters()), Val=val)
    save_state(out / 'best.pt', best)
    write_json(out / 'selection_sealed.json', selection)
    log(out, 'selection_sealed', epoch=best['epoch'])
    return selection


def run(config, out, resume):
    resolved = out / 'config_resolved.json'
    if resume:
        if json.loads(resolved.read_text(encoding='utf-8')) != config:
            raise ValueError('The requested run configuration differs from the stored one.')
        if (out / 'results.json').is_file():
            log(out, 'completed', results=str(out / 'results.json'))
            return
    else:
        write_json(resolved, config)
    fixed = FixedViews(config).cuda().eval()
    sealed = out / 'selection_sealed.json'
    if sealed.is_file():
        selection = json.loads(sealed.read_text(encoding='utf-8'))
    else:
        log(out, 'preparing_train_and_val')
        train_data = Q0Dataset(config, 'Train')
        val_data = Q0Dataset(config, 'Val')
        originals = feature_cache(fixed, train_data, config)
        normalization = fit_normalization(train_data, originals, config)
        val_cache = {condition: feature_cache(fixed, val_data, config, condition) for condition in CONDITIONS}
        selection = train(config, out, fixed, train_data, originals, val_data, val_cache, normalization, resume)
        del val_cache, originals

    log(out, 'test_started', epoch=selection['epoch'])
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    if saved['config'] != config or saved['epoch'] != selection['epoch'] or saved['score'] != selection['score']:
        raise ValueError('best.pt differs from the sealed Val selection.')
    model = DirectSetModel(config, saved['normalization']).cuda()
    model.load_state_dict(saved['model'], strict=True)
    test_data = Q0Dataset(config, 'Test')
    test_cache = {condition: feature_cache(fixed, test_data, config, condition) for condition in CONDITIONS}
    test = evaluate(model, test_data, test_cache, config, out)
    write_json(out / 'results.json', {'config': config, 'selection': selection, 'Test': test})
    log(out, 'completed', results=str(out / 'results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json', help='Training protocol JSON.')
    parser.add_argument('--data-root', type=Path, help='Location of the screened DeepShip dataset.')
    parser.add_argument('--out', type=Path, help='Run directory. Use a new path for each formal run.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--smoke', action='store_true', help='One epoch with three samples per set and split.')
    parser.add_argument('--resume', action='store_true', help='Resume the same interrupted run.')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    config['data_root'] = str(Path(args.data_root or config['data_root']).resolve())
    config['seed'] = args.seed
    config['smoke'] = args.smoke
    prefix = config.get('experiment_prefix', 'DS-Q0')
    config['experiment_id'] = f"{prefix}-{'smoke-' if args.smoke else ''}{args.seed}"
    if args.smoke:
        config.update(min_epochs=1, max_epochs=1)
    configure_runtime()
    config['runtime'] = dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                             cuda=torch.version.cuda, numpy=np.__version__, python=sys.executable)
    out = (args.out or HERE / 'runs' / config['experiment_id']).resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = out / '.running'
    lock.mkdir()
    try:
        run(config, out, args.resume)
    finally:
        lock.rmdir()


if __name__ == '__main__':
    main()
