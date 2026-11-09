"""Generate all three splits from local Bellhop components and raw-E caches.

No old experiment arrays, model weights, predictions, or recognition scores are read.
Source ownership is immutable. A complete output is never silently overwritten.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow
from common import ROOT, SPLITS, config, read, write, table, rows, rng, rms, normalize


def parameters(cfg, random):
    return dict(left=float(random.uniform(*cfg['context_seconds'])),
                right=float(random.uniform(*cfg['context_seconds'])),
                fade_left=float(random.uniform(*cfg['fade_seconds'])),
                fade_right=float(random.uniform(*cfg['fade_seconds'])))


def envelopes(row, cfg):
    n = cfg['sample_rate'] * cfg['duration_seconds']
    e = [np.ones(n), np.ones(n)]
    if row['context_condition'] == 'C0':
        return e
    t = np.arange(n) / cfg['sample_rate']
    left, right, fl, fr = (float(row[k]) for k in ('left', 'right', 'fade_left', 'fade_right'))
    eps = cfg['epsilon']
    rise = eps + (1-eps) * .5 * (1-np.cos(np.pi*np.clip((t-left)/fl, 0, 1)))
    fall = eps + (1-eps) * .5 * (1+np.cos(np.pi*np.clip((t-(5-right-fr))/fr, 0, 1)))
    first, side = int(row['first']), int(row['side'])
    if row['context_condition'] == 'C2' or side == 0:
        e[1-first] = rise
    if row['context_condition'] == 'C2' or side == 1:
        e[first] = fall
    return e


def single_envelope(row, cfg):
    if row['augmentation_type'] == 'flat':
        return np.ones(cfg['sample_rate'] * cfg['duration_seconds'])
    p = dict(row, context_condition='C2')
    a, b = envelopes(p, cfg)
    return a*b if row['augmentation_type'] == 'middle' else (a, b)[int(row['side'])]


def apportion(total, weights, minimum=0):
    w = np.asarray(weights, dtype=float)
    assert total >= minimum*len(w) and np.all(w >= 0) and w.sum() > 0
    x = (total-minimum*len(w))*w/w.sum()
    out = np.floor(x).astype(int) + minimum
    for i in sorted(range(len(w)), key=lambda i: (-(x[i]-np.floor(x[i])), i))[:total-int(out.sum())]:
        out[i] += 1
    return out.tolist()


def core_schedule(pool, total, random):
    """Full core cycles; cover different recordings early within every cycle."""
    by = defaultdict(list)
    for i, row in enumerate(pool):
        by[row['recording_id']].append(i)
    result = []
    while len(result) < total:
        order = list(random.permutation(sorted(by)))
        queues = {r: random.permutation(by[r]).tolist() for r in order}
        cycle = []
        while any(queues.values()):
            for rec in order:
                if queues[rec]:
                    cycle.append(queues[rec].pop())
        result.extend(cycle)
    return result[:total]


def unique_edges(a, b, random):
    """Realize exact per-core degree budgets as a simple bipartite graph."""
    ca, cb = Counter(a), Counter(b)
    ai = random.permutation(sorted(ca)).tolist()
    bi = random.permutation(sorted(cb)).tolist()
    na, nb = len(ai), len(bi)
    sink = na+nb+1
    ii = np.r_[np.zeros(na, dtype=int), np.repeat(np.arange(1, na+1), nb), np.arange(na+1, sink)]
    jj = np.r_[np.arange(1, na+1), np.tile(np.arange(na+1, sink), na), np.full(nb, sink)]
    vv = np.r_[[ca[i] for i in ai], np.ones(na*nb, dtype=int), [cb[i] for i in bi]].astype(np.int32)
    graph = coo_matrix((vv, (ii, jj)), shape=(sink+1, sink+1)).tocsr()
    result = maximum_flow(graph, 0, sink)
    assert result.flow_value == len(a) == len(b), 'Unique-pair capacity conflicts with the core reuse budget.'
    block = result.flow[1:na+1, na+1:sink].tocoo()
    edges = [(ai[int(i)], bi[int(j)]) for i, j, v in zip(block.row, block.col, block.data) if v > 0]
    random.shuffle(edges)
    assert len(set(edges)) == len(a)
    return edges


def noise_candidates(cfg, recordings):
    n = cfg['sample_rate'] * cfg['duration_seconds']
    step = round(cfg['noise_stride_seconds'] * cfg['sample_rate'])
    result = []
    for rec in recordings:
        if rec['class_name'] != 'E':
            continue
        path = ROOT / rec['background_path']
        x, fs = sf.read(path, dtype='float64')
        assert fs == cfg['sample_rate'] and x.ndim == 1 and np.isfinite(x).all()
        for start in range(0, len(x)-n+1, step):
            assert rms(x[start:start+n]) > 0, 'Silent E candidate requires explicit review.'
            result.append(dict(source_id=f"{rec['recording_id']}_E_{start:09d}",
                               parent_core_id=f"{rec['recording_id']}_E_{start:09d}",
                               recording_id=rec['recording_id'], split=rec['split'], class_name='E',
                               start_sample=start, stop_sample=start+n, channel_id=0,
                               component_path=rec['background_path'], source_kind='background_window'))
    return result


def select_noise(candidates, count):
    by = defaultdict(list)
    for r in candidates:
        by[r['recording_id']].append(r)
    names = sorted(by)
    count = min(max(count, len(names)), len(candidates))
    capacities = [len(by[k]) for k in names]
    q = np.minimum(apportion(count, capacities, minimum=1), capacities).tolist()
    while sum(q) < count:
        for i, capacity in enumerate(capacities):
            if q[i] < capacity and sum(q) < count:
                q[i] += 1
    result = []
    for name, quota in zip(names, q):
        pool = sorted(by[name], key=lambda r: r['start_sample'])
        assert quota <= len(pool)
        indices = np.rint(np.linspace(0, len(pool)-1, quota)).astype(int)
        assert len(set(indices.tolist())) == quota
        result.extend(pool[int(i)] for i in indices)
    return result


def approximate_items(total, names, ratios):
    weights = np.asarray([ratios[k] for k in names], dtype=float)
    assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
    # Integer rounding, with the remainder in the last group. No ratio equality check.
    edges = np.r_[0, np.floor(np.cumsum(weights/weights.sum())*total).astype(int)]
    edges[-1] = total
    return [name for name, n in zip(names, np.diff(edges)) for _ in range(int(n))]


def make_plan(cfg, sources, recordings):
    classes = cfg['classes']
    pairs = list(combinations(classes, 2))
    sets = ['E', *classes, *['+'.join(p) for p in pairs]]
    noise = noise_candidates(cfg, recordings)
    selected_noise, plans = [], {}
    for split in SPLITS:
        part = sorted((r for r in sources if r['split'] == split), key=lambda r: r['source_id'])
        by = {c: [r for r in part if r['class_name'] == c] for c in classes}
        assert all(by.values()), 'Every split needs each target class.'
        ratio = cfg['approximate_type_ratio']
        requested = round(len(part)*ratio['double']/ratio['single']/len(pairs))
        coverage_floor = int(np.ceil(max(map(len, by.values()))/(len(classes)-1)))
        capacity = min(len(by[a])*len(by[b]) for a,b in pairs)
        q = min(max(requested, coverage_floor), capacity)
        assert q >= coverage_floor, 'Source capacity cannot cover all cores with equal pair quotas.'
        records = []
        def item(a, b=None, **kw):
            label = a['class_name'] if b is None else '+'.join((a['class_name'], b['class_name']))
            kind = 'noise' if label == 'E' else 'single' if b is None else 'double'
            r = dict(sample_id=f'{split}_{len(records):06d}', array_index=len(records), split=split,
                     kind=kind, label_name=label, label_index=sets.index(label), count=0 if label=='E' else 1 if b is None else 2,
                     source1_id=a['source_id'], source2_id='' if b is None else b['source_id'],
                     context_condition='NA', augmentation_type='none', first=0, side=0,
                     left=1., right=1., fade_left=.25, fade_right=.25,
                     nominal_sir_db=cfg['sir_db'] if b is not None else '',
                     wav_path=f"data/{split}/wav/{split}_{len(records):06d}.wav")
            for k, source in ((1,a), (2,b)):
                for f in ('recording_id','parent_core_id','start_sample','stop_sample','channel_id','class_name'):
                    r[f'source{k}_{f}'] = source[f] if source else ''
            r.update({f'target_{c}': int(c in label.split('+')) for c in classes})
            r.update(kw)
            records.append(r)
        for r in part:
            item(r)
        if split == 'Train':
            for c in classes:
                group = [r for r in records if r['label_name'] == c]
                random = rng(cfg['generation_seed'], split, c, 'single-augmentation')
                kinds = approximate_items(len(group), ('flat','edge','middle'), cfg['single_train_augmentation'])
                random.shuffle(kinds)
                for j, (r, kind) in enumerate(zip(group, kinds)):
                    r.update(augmentation_type=kind, first=int(random.integers(2)), side=j%2, **parameters(cfg, random))
        schedules = {c: core_schedule(by[c], (len(classes)-1)*q, rng(cfg['generation_seed'], split, c, 'core-cycles')) for c in classes}
        cursors = Counter()
        for ca, cb in pairs:
            random = rng(cfg['generation_seed'], split, ca, cb, 'pairing')
            edges = unique_edges(schedules[ca][cursors[ca]:cursors[ca]+q], schedules[cb][cursors[cb]:cursors[cb]+q], random)
            cursors[ca] += q
            cursors[cb] += q
            conditions = approximate_items(q, ('C0','C1','C2'), cfg['context_ratio'])
            random.shuffle(conditions)
            orientations = {}
            for cond, n in Counter(conditions).items():
                allowed = [(0,0),(0,1),(1,0),(1,1)] if cond == 'C1' else [(0,0),(1,0)]
                values = [allowed[j%len(allowed)] for j in range(n)]
                random.shuffle(values)
                orientations[cond] = values
            for (ia, ib), cond in zip(edges, conditions):
                first, side = orientations[cond].pop()
                item(by[ca][ia], by[cb][ib], context_condition=cond, first=first, side=side, **parameters(cfg, random))
        selected = select_noise([r for r in noise if r['split']==split], round(len(part)*ratio['noise']/ratio['single']))
        selected_noise.extend(selected)
        for r in selected:
            item(r)
        plans[split] = records
    return plans, selected_noise, noise


def load_component(source):
    x, fs = sf.read(ROOT/source['component_path'], dtype='float64')
    assert fs == 16000 and x.ndim == 1
    if source.get('source_kind') == 'background_window':
        x = x[int(source['start_sample']):int(source['stop_sample'])]
    assert x.shape == (80000,) and np.isfinite(x).all() and rms(x)>0
    return x


def render(row, source_map, cfg, cache):
    def get(key):
        if key not in cache:
            cache[key] = load_component(source_map[key])
        return cache[key]
    a = get(row['source1_id'])
    g1, g2 = 1/rms(a), 0.
    post = ''
    if row['kind'] == 'double':
        b = get(row['source2_id'])
        g2 = 10**(-float(row['nominal_sir_db'])/20)/rms(b)
        ea, eb = envelopes(row, cfg)
        u, v = a*g1*ea, b*g2*eb
        post = float(20*np.log10(rms(u)/rms(v)))
        raw = u+v
    else:
        raw = a*g1
    common = cfg['output_rms']/rms(raw)
    out = (raw*common).astype(np.float32)
    return out, dict(source1_gain=g1, source2_gain=g2, common_gain=common,
                     post_envelope_sir_db=post, output_rms=rms(out), peak=float(np.max(np.abs(out))))


def source_map_from_index(index):
    result = {}
    for row in index:
        for k in (1,2):
            key = row[f'source{k}_id']
            if not key:
                continue
            source = dict(source_id=key, split=row['split'],
                recording_id=row[f'source{k}_recording_id'], class_name=row[f'source{k}_class_name'],
                component_path=row[f'source{k}_component_path'],
                source_kind='background_window' if row['kind']=='noise' else 'bellhop_component',
                start_sample=int(row[f'source{k}_start_sample']), stop_sample=int(row[f'source{k}_stop_sample']))
            assert key not in result or result[key] == source, 'Inconsistent source in index.'
            result[key] = source
    return result


def run(config_path=None):
    cfg, dest = config(config_path)
    index_path = dest/'DATA_INDEX.csv'
    existing = [(dest/'data'/split/'audio.npy').exists() for split in SPLITS]
    if any(existing):
        assert all(existing) and index_path.exists(), 'Incomplete output; use a new directory.'
        from audit import run as check
        check(dest)
        print('Existing indexed data kept unchanged. New ratios require a new output_directory.', flush=True)
        return dest
    if index_path.exists():
        # Exact reproduction of the accepted ordering uses the saved index and its configuration.
        cfg = read(dest/'CONFIG.json')
        index = rows(index_path)
        plans = {split:[r for r in index if r['split']==split] for split in SPLITS}
        source_map = source_map_from_index(index)
    else:
        sources = read(ROOT/'metadata/sources.json')
        recordings = read(ROOT/'metadata/recordings.json')
        plans, noise, _ = make_plan(cfg, sources, recordings)
        source_map = {r['source_id']:r for r in sources+noise}
        for records in plans.values():
            for row in records:
                for k in (1,2):
                    row[f'source{k}_component_path'] = source_map[row[f'source{k}_id']]['component_path'] if row[f'source{k}_id'] else ''
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest/'.generating'
    marker.write_text('Generation in progress', encoding='utf-8')
    write(dest/'CONFIG.json', cfg)
    final_index = []
    for split, records in plans.items():
        folder = dest/'data'/split
        (folder/'wav').mkdir(parents=True, exist_ok=True)
        audio = np.lib.format.open_memmap(folder/'audio.npy', mode='w+', dtype=np.float32, shape=(len(records),80000))
        cache = {}
        for i, row in enumerate(records):
            wave, detail = render(row, source_map, cfg, cache)
            audio[i] = wave
            sf.write(dest/row['wav_path'], wave, cfg['sample_rate'], subtype='FLOAT')
            row.update(detail)
            final_index.append(row)
            if i % 500 == 0:
                print(split, i+1, '/', len(records), flush=True)
        audio.flush()
        del audio
    table(index_path, final_index)
    marker.unlink()
    from audit import run as check
    check(dest)
    print('Generated:', dest, flush=True)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--triple-only', action='store_true', help='Generate only the separate triple extension')
    parser.add_argument('--plan', action='store_true', help='Preview triple counts without writing audio')
    args = parser.parse_args()
    selected_cfg = read(args.config or ROOT/'config.json')
    if args.triple_only or selected_cfg.get('triple', {}).get('enabled', False):
        from generate_triple import run as run_triple
        run_triple(args.config, plan_only=args.plan)
    else:
        if args.plan:
            parser.error('--plan requires --triple-only or triple.enabled=true')
        run(args.config)
