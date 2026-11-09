"""Compare rank-matched projection readouts on paired swaps and the formal protocol."""

from dataclasses import asdict
import msvcrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from screen import (ARMS, ATTENUATION, HERE, LEVELS, REFERENCE, FrozenGn, Panel, configure,
                    load_json, recording_means, status)
from train_formal import (CONDITIONS, FixedDataset, evaluate, evaluation_views, feature_cache,
                          model_state, ranking, save_state, write_json)

EPOCHS = 20
PAIR_WEIGHT = 0.5
SET_INDEX = {(0, 1): 4, (0, 2): 5, (1, 2): 6}


def attenuated(x, basis):
    x = x.float()
    coord = torch.einsum('bmd,amdr->bamr', x, basis)
    return x.unsqueeze(1) - ATTENUATION * torch.einsum('bamr,amdr->bamd', coord, basis)


class Readout(nn.Module):
    def __init__(self, bases):
        super().__init__()
        self.register_buffer('waveform_basis', bases['waveform'])
        self.register_buffer('spectral_basis', bases['spectral'])
        self.score = nn.Linear(64, 3, bias=False)
        nn.init.zeros_(self.score.weight)

    def margins(self, waveform, spectral):
        with torch.autocast('cuda', enabled=False):
            features = attenuated(waveform, self.waveform_basis) * attenuated(spectral, self.spectral_basis)
            scores = self.score(features)
            pf = scores[:, 0, :, 1] - scores[:, 0, :, 2]
            mb = scores[:, 1, :, 0] - scores[:, 1, :, 2]
            rv = scores[:, 2, :, 0] - scores[:, 2, :, 1]
            return torch.stack((pf, mb, rv), dim=-1)

    def adjust(self, base, margins):
        pf, mb, rv = margins.unbind(-1)
        correction = 0.5 * torch.stack((pf + mb, -pf + rv, -mb - rv), dim=-1)
        double = base[..., 4:].float()
        corrected = double + correction
        offset = double.logsumexp(-1, keepdim=True) - corrected.logsumexp(-1, keepdim=True)
        return torch.cat((base[..., :4].float(), corrected + offset), dim=-1)

    def forward(self, waveform, spectral, base):
        return self.adjust(base, self.margins(waveform, spectral))


def formal_cache(frozen, data, config, condition, indices=None):
    if indices is None:
        raw = feature_cache(frozen.front, data, config, condition)
    else:
        batches = []
        for start in range(0, len(indices), config['batch_size']):
            ids = indices[start:start + config['batch_size']]
            waves = evaluation_views(data, ids, condition, config['evaluation_noise_seed'])
            batches.append(tuple(item.cpu() for item in frozen.front(waves)))
        raw = tuple(torch.cat(parts) for parts in zip(*batches))
    encoded = []
    for start in range(0, len(raw[0]), config['evaluation_batch_size']):
        batch = (item[start:start + config['evaluation_batch_size']].cuda() for item in raw)
        encoded.append(tuple(item.cpu() for item in frozen.encode(*batch)))
    return tuple(torch.cat(parts) for parts in zip(*encoded))


def check_reference(data, cache):
    for condition, values in cache.items():
        probability = values[-1].cuda().float().softmax(-1).cpu().numpy().mean(1)
        with np.load(REFERENCE / f'{data.split}_{condition}.npz') as expected:
            for key in ('ids', 'labels', 'contexts'):
                np.testing.assert_array_equal(getattr(data, key), expected[key])
            np.testing.assert_array_equal(probability.argmax(-1), expected['predictions'])
            np.testing.assert_allclose(probability, expected['probabilities'], rtol=1e-5, atol=1e-6)


def train_batch(model, optimizer, formal, targets, pair, anchor, config):
    optimizer.zero_grad(set_to_none=True)
    logits = model(*formal)[..., 4:]
    conditional = F.cross_entropy(logits.flatten(0, 1), targets[:, None].expand(logits.shape[:2]).reshape(-1))
    margins = model.margins(*pair).unflatten(0, (-1, 2))
    by_anchor = margins.gather(-1, anchor[:, None, None, None].expand(*margins.shape[:3], 1)).squeeze(-1)
    contrast = (F.softplus(-by_anchor[:, 0]).mean() + F.softplus(by_anchor[:, 1]).mean()) / 2
    loss = conditional + PAIR_WEIGHT * contrast
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def fit(models, frozen, config, out, train, val, train_panel, panel_cache):
    indices = np.flatnonzero(train.counts == 2)
    targets = torch.from_numpy(train.labels[indices] - 4).cuda()
    formal = []
    for condition in CONDITIONS:
        formal.append(formal_cache(frozen, train, config, condition, indices))
        status(out, 'train_cached', condition=condition, double=len(indices))
    formal = tuple(torch.stack(parts) for parts in zip(*formal))
    val_cache = {condition: formal_cache(frozen, val, config, condition) for condition in CONDITIONS}
    check_reference(val, val_cache)
    sample = tuple(item[:config['evaluation_batch_size']].cuda() for item in val_cache['original'])
    with torch.no_grad():
        for model in models.values():
            torch.testing.assert_close(model(*sample), sample[-1].float(), rtol=0, atol=0)
    pair = tuple(torch.cat([panel_cache[level][view][:, (3, 5)] for level in LEVELS])
                 for view in (0, 1))
    anchor = torch.from_numpy(np.tile(train_panel.labels, len(LEVELS))).cuda()
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'],
                                         weight_decay=config['weight_decay']) for arm, model in models.items()}
    history, best = [], {}
    batch_size = config['evaluation_batch_size']
    batches = int(np.ceil(len(indices) / batch_size))
    status(out, 'training_started', epochs=EPOCHS, replay='all_val_views_and_zero_readout')
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng([config['seed'], 851, epoch])
        order = rng.permutation(len(indices))
        noise = rng.choice(len(CONDITIONS), len(indices), p=config['noise_probabilities'])
        swap_batches = np.array_split(rng.permutation(len(anchor)), batches)
        losses = dict.fromkeys(ARMS, 0.)
        for i, pair_ids in enumerate(swap_batches):
            ids = order[i * batch_size:(i + 1) * batch_size]
            formal_batch = tuple(item[noise[ids], ids].cuda() for item in formal)
            pair_batch = tuple(item[pair_ids].flatten(0, 1).cuda() for item in pair)
            for arm, model in models.items():
                losses[arm] += train_batch(model, optimizers[arm], formal_batch, targets[ids],
                                            pair_batch, anchor[pair_ids], config) / batches

        rows = {}
        for arm, model in models.items():
            metrics = evaluate(model, val, val_cache, config)
            score = ranking(metrics)
            if arm not in best or score > best[arm]['score']:
                best[arm] = dict(epoch=epoch, score=score, model=model_state(model))
            rows[arm] = dict(loss=losses[arm], score=score, best_epoch=best[arm]['epoch'], Val=metrics)
        history.append(dict(epoch=epoch, arms=rows))
        status(out, 'epoch_finished', epoch=epoch,
               best={arm: row['best_epoch'] for arm, row in rows.items()})
    write_json(out / 'history.json', history)

    selected = {}
    for arm, model in models.items():
        directory = out / arm
        directory.mkdir(exist_ok=True)
        picked = best[arm]
        model.load_state_dict(picked['model'], strict=True)
        metrics = evaluate(model, val, val_cache, config, directory)
        if ranking(metrics) != picked['score']:
            raise RuntimeError('The selected readout failed to reproduce its validation score.')
        selected[arm] = dict(epoch=picked['epoch'], score=picked['score'], Val=metrics,
                             parameters=sum(p.numel() for p in model.parameters()))
        save_state(directory / 'best.pt', dict(arm=arm, **picked))
        write_json(directory / 'selection_sealed.json', selected[arm])
    status(out, 'selection_sealed', epochs={arm: selected[arm]['epoch'] for arm in ARMS})
    return selected


def pair_correctness(margin):
    return (margin[:, (1, 3, 4)] > 0).all(1) & (margin[:, (2, 5, 6)] < 0).all(1)


def strength_drift(margin):
    weaker = np.abs(margin[:, (3, 5)] - margin[:, (1, 2)])
    stronger = np.abs(margin[:, (4, 6)] - margin[:, (1, 2)])
    return (weaker + stronger).mean(1) / 2


@torch.no_grad()
def panel_results(models, panel, cache, config):
    first = np.array([SET_INDEX[tuple(sorted((t.anchor.label, t.first.label)))] for t in panel.triplets])
    second = np.array([SET_INDEX[tuple(sorted((t.anchor.label, t.second.label)))] for t in panel.triplets])
    rows = np.arange(len(first))
    results = {}
    for level, features in cache.items():
        pieces = {arm: [] for arm in ('GN', *ARMS)}
        branch = {arm: [] for arm in ARMS}
        flat = tuple(item.flatten(0, 1) for item in features)
        for start in range(0, len(flat[0]), config['evaluation_batch_size']):
            batch = tuple(item[start:start + config['evaluation_batch_size']].cuda() for item in flat)
            pieces['GN'].append(batch[-1].float().softmax(-1).cpu().numpy().mean(1))
            for arm, model in models.items():
                margins = model.margins(*batch[:2])
                logits = model.adjust(batch[-1], margins)
                pieces[arm].append(logits.softmax(-1).cpu().numpy().mean(1))
                branch[arm].append(margins.mean(1).cpu().numpy())
        results[f'{level}dB'] = {}
        for arm, parts in pieces.items():
            p = np.concatenate(parts).reshape(len(rows), -1, 7)
            margin = np.log(p[rows, :, first].clip(1e-30)) - np.log(p[rows, :, second].clip(1e-30))
            pred = p.argmax(-1)
            swap = (margin[:, 1] > 0) & (margin[:, 2] < 0)
            both_full = (pred[:, 1] == first) & (pred[:, 2] == second)
            metrics = dict(swap_sign_both=recording_means(swap, panel),
                           both_double_sets=recording_means(both_full, panel),
                           all_three_sets=recording_means(both_full & (pred[:, 0] == panel.labels + 1), panel),
                           all_strengths_correct=recording_means(pair_correctness(margin), panel),
                           margin_drift=float(strength_drift(margin).mean()))
            if arm in branch:
                by_anchor = np.concatenate(branch[arm]).reshape(len(rows), -1, 3)[rows, :, panel.labels]
                metrics['branch_swap_sign_both'] = recording_means(
                    (by_anchor[:, 1] > 0) & (by_anchor[:, 2] < 0), panel)
                metrics['branch_all_strengths_correct'] = recording_means(pair_correctness(by_anchor), panel)
                metrics['branch_margin_drift'] = float(strength_drift(by_anchor).mean())
            results[f'{level}dB'][arm] = metrics
    return results


def check_mass(data, out):
    for condition in CONDITIONS:
        stem = f'{data.split}_{condition}.npz'
        with np.load(REFERENCE / stem) as original:
            baseline = original['probabilities']
        for arm in ARMS:
            with np.load(out / arm / stem) as adjusted:
                p = adjusted['probabilities']
                np.testing.assert_allclose(p[:, :4], baseline[:, :4], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(p[:, 4:].sum(-1), baseline[:, 4:].sum(-1), rtol=1e-5, atol=1e-6)


def run():
    config = configure()
    root = HERE / 'runs' / f'SE-AnchorProjection-GN-{config["seed"]}'
    with (root / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        screened = load_json(root / 'screen.json')
        if not screened['passed']:
            raise ValueError('The prespecified direction screen did not pass.')
        out = root / 'readout'
        out.mkdir(exist_ok=True)
        if (out / 'results.json').is_file():
            print(f'Readout results already exist at {out}.', flush=True)
            return
        settings = dict(epochs=EPOCHS, pair_weight=PAIR_WEIGHT, formal_train='fixed_six_views_of_double',
                        formal_noise_probabilities=config['noise_probabilities'],
                        selection=config['validation_rank'])
        write_json(out / 'protocol.json', settings)
        prepared = torch.load(root / 'screen_features.pt', map_location='cpu', weights_only=True)
        train, val = (FixedDataset(config['data_root'], split) for split in ('Train', 'Val'))
        train_panel, val_panel = (Panel(data, config['seed']) for data in (train, val))
        expected = load_json(root / 'panel_index.json')
        for panel in (train_panel, val_panel):
            if [asdict(t) for t in panel.triplets] != expected[panel.split]:
                raise RuntimeError('The paired cache differs from its source manifest.')
        models = {arm: Readout(prepared['bases'][arm]).cuda() for arm in ARMS}
        frozen = FrozenGn(config)
        selected = fit(models, frozen, config, out, train, val, train_panel, prepared['caches']['Train'])
        check_mass(val, out)
        mechanism = panel_results(models, val_panel, prepared['caches']['Val'], config)
        write_json(out / 'paired_val.json', mechanism)

        test = FixedDataset(config['data_root'], 'Test')
        status(out, 'test_started', examples=len(test))
        cache = {condition: formal_cache(frozen, test, config, condition) for condition in CONDITIONS}
        check_reference(test, cache)
        metrics = {arm: evaluate(model, test, cache, config, out / arm) for arm, model in models.items()}
        check_mass(test, out)
        write_json(out / 'results.json', dict(protocol=settings, selection=selected, Test=metrics))
        status(out, 'completed', test_acc={arm: {c: row[c]['acc'] for c in CONDITIONS}
                                          for arm, row in metrics.items()})


if __name__ == '__main__':
    run()
