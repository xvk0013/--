"""Generate an independent PF+MB+RV extension from the accepted source split."""
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from common import ROOT, SPLITS, config, read, rows, write, table, rng, rms
from generate import approximate_items, core_schedule, parameters, load_component


def accepted_pools(base_index, sources, recordings, classes):
    """Use only existing indexed components, keeping recording ownership fixed."""
    owners = {r['recording_id']: (r['split'], r['class_name']) for r in recordings}
    indexed = {}
    for row in base_index:
        for k in (1, 2):
            key = row[f'source{k}_id']
            if not key or row[f'source{k}_class_name'] not in classes:
                continue
            identity = (row['split'], row[f'source{k}_recording_id'], row[f'source{k}_class_name'])
            if key in indexed and indexed[key] != identity:
                raise ValueError(f'Conflicting indexed source ownership: {key}')
            indexed[key] = identity
    pools = {s: {c: [] for c in classes} for s in SPLITS}
    for source in sorted(sources, key=lambda r: r['source_id']):
        key = source['source_id']
        if key not in indexed:
            continue
        identity = (source['split'], source['recording_id'], source['class_name'])
        if identity != indexed[key] or owners.get(source['recording_id']) != (source['split'], source['class_name']):
            raise ValueError(f'Cross-split or class mismatch: {key}')
        pools[source['split']][source['class_name']].append(source)
    if sum(len(p) for split in pools.values() for p in split.values()) != len(indexed):
        raise ValueError('Accepted source metadata is incomplete.')
    if any(not p for split in pools.values() for p in split.values()):
        raise ValueError('Each split needs PF, MB and RV components.')
    return pools


def make_plan(cfg, pools):
    plans = {}
    classes = cfg['classes']
    options = cfg['triple']
    for split in SPLITS:
        total = options['samples_per_split'][split]
        by = pools[split]
        sizes = [len(by[c]) for c in classes]
        capacity = int(np.prod(sizes, dtype=np.int64))
        if total <= 0 or total > capacity:
            raise ValueError(f'{split}: requested {total} triples; capacity is {capacity}.')
        random = rng(cfg['generation_seed'], 'triple-v1', split)
        schedules = [core_schedule(by[c], total, rng(cfg['generation_seed'], 'triple-cores', split, c)) for c in classes]
        # Cycles distribute reuse across components; a finite unique fallback resolves collisions.
        fallback = iter(random.choice(capacity, size=total, replace=False).tolist())
        used = set()
        contexts = approximate_items(total, ('C0', 'C1', 'C2'), options['context_ratio'])
        random.shuffle(contexts)
        orientations = [tuple(p) for p in ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))]
        random.shuffle(orientations)
        records = []
        for i, cond in enumerate(contexts):
            picks = tuple(int(schedule[i]) for schedule in schedules)
            flat = int(np.ravel_multi_index(picks, sizes))
            if flat in used:
                flat = next(candidate for candidate in fallback if candidate not in used)
                picks = tuple(int(v) for v in np.unravel_index(flat, sizes))
            used.add(flat)
            left, right = orientations[i % len(orientations)]
            sample_id = f'{split}_triple_{i:06d}'
            row = dict(sample_id=sample_id, array_index=i, split=split,
                       kind='triple', label_name='+'.join(classes), label_index=7, count=3,
                       context_condition=cond, augmentation_type='none',
                       left_dominant=left, right_dominant=right, side=i % 2,
                       nominal_sir_db='', **parameters(cfg, random),
                       wav_path=f'data/{split}/wav/{sample_id}.wav' if options['export_wavs'] else '')
            for k, (cls, pick) in enumerate(zip(classes, picks), 1):
                source = by[cls][pick]
                row[f'source{k}_id'] = source['source_id']
                for field in ('recording_id', 'parent_core_id', 'start_sample', 'stop_sample',
                              'channel_id', 'class_name', 'component_path'):
                    row[f'source{k}_{field}'] = source[field]
                row[f'source{k}_relative_level_db'] = options['relative_level_db'][k-1]
            row.update({f'target_{c}': 1 for c in classes})
            records.append(row)
        plans[split] = records
    return plans


def envelopes(row, cfg):
    n = int(cfg['sample_rate'] * cfg['duration_seconds'])
    e = np.ones((3, n), dtype=np.float64)
    cond = row['context_condition']
    if cond == 'C0':
        return e
    t = np.arange(n) / cfg['sample_rate']
    eps = cfg['epsilon']
    rise = eps + (1-eps)*.5*(1-np.cos(np.pi*np.clip((t-float(row['left']))/float(row['fade_left']), 0, 1)))
    fall = eps + (1-eps)*.5*(1+np.cos(np.pi*np.clip((t-(cfg['duration_seconds']-float(row['right'])-float(row['fade_right'])))/float(row['fade_right']), 0, 1)))
    if cond == 'C2' or (cond == 'C1' and int(row['side']) == 0):
        for k in range(3):
            if k != int(row['left_dominant']):
                e[k] *= rise
    if cond == 'C2' or (cond == 'C1' and int(row['side']) == 1):
        for k in range(3):
            if k != int(row['right_dominant']):
                e[k] *= fall
    return e


def render(row, source_map, cfg, cache):
    env = envelopes(row, cfg)
    parts, detail = [], {}
    for k in (1, 2, 3):
        key = row[f'source{k}_id']
        if key not in cache:
            cache[key] = load_component(source_map[key])
        x = cache[key]
        gain = 10**(float(row[f'source{k}_relative_level_db'])/20) / rms(x)
        detail[f'source{k}_gain'] = gain
        parts.append(x * gain * env[k-1])
    raw = sum(parts)
    level = rms(raw)
    if not np.isfinite(level) or level <= 0:
        raise ValueError(f'Invalid mixture: {row["sample_id"]}')
    common_gain = cfg['output_rms'] / level
    out = (raw * common_gain).astype(np.float32)
    detail.update(common_gain=common_gain, output_rms=rms(out), peak=float(np.abs(out).max()))
    return out, detail


def settings(config_path=None):
    user_cfg, base = config(config_path)
    options = dict(user_cfg.get('triple', {}))
    if not options.get('output_directory'):
        raise ValueError('Configure triple.output_directory separately from the base dataset.')
    dest = (ROOT / options['output_directory']).resolve()
    if (not dest.is_relative_to(ROOT/'datasets') or dest == base or
            dest.is_relative_to(base) or base.is_relative_to(dest)):
        raise ValueError('Triple output must be a separate datasets/ directory.')
    index = rows(base/'DATA_INDEX.csv')
    cfg = read(base/'CONFIG.json')
    if cfg['classes'] != ['PF', 'MB', 'RV']:
        raise ValueError('This extension uses PF/MB/RV and preserves label 7 for PF+MB+RV.')
    requested = options.get('samples_per_split')
    options['samples_per_split'] = {}
    for split in SPLITS:
        raw = requested[split] if requested is not None else round(sum(r['split']==split and r['kind']=='double' for r in index)/3)
        if isinstance(raw, bool) or int(raw) != raw or raw <= 0:
            raise ValueError('Triple split counts must be positive integers.')
        options['samples_per_split'][split] = int(raw)
    options.setdefault('context_ratio', cfg['context_ratio'])
    weights = np.asarray([options['context_ratio'][k] for k in ('C0','C1','C2')], float)
    levels = np.asarray(options.get('relative_level_db', [0.,0.,0.]), float)
    if (not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0 or
            levels.shape != (3,) or not np.isfinite(levels).all()):
        raise ValueError('Invalid context weights or source levels.')
    options['relative_level_db'] = levels.tolist()
    options.setdefault('export_wavs', False)
    options.pop('enabled', None)  # The switch selects execution, not waveform content.
    cfg.update(version='shipsear_triple_extension_v1', generation_seed=user_cfg['generation_seed'],
               output_directory=options['output_directory'], triple=options,
               base_dataset=str(base), label_names=['E','PF','MB','RV','PF+MB','PF+RV','MB+RV','PF+MB+RV'],
               sir_definition='relative per-source RMS levels before envelopes; pairwise equal power when all levels are zero',
               sir_db=None, source_policy='existing accepted index components; same recording ownership')
    cfg['single_train_augmentation'] = dict(cfg['single_train_augmentation'], enabled=False)
    return cfg, dest, index


def run(config_path=None, plan_only=False):
    cfg, dest, base_index = settings(config_path)
    if dest.exists():
        complete = (dest/'DATA_INDEX.csv').is_file() and (dest/'CONFIG.json').is_file() and all((dest/'data'/s/'audio.npy').is_file() for s in SPLITS)
        if (dest/'.generating').exists() or not complete:
            raise ValueError('Incomplete triple output; use a new triple.output_directory.')
        if read(dest/'CONFIG.json') != cfg:
            raise ValueError('Triple configuration changed; use a new triple.output_directory.')
        print(f'Existing triple extension kept unchanged: {dest}', flush=True)
        return dest
    sources = read(ROOT/'metadata/sources.json')
    pools = accepted_pools(base_index, sources, read(ROOT/'metadata/recordings.json'), cfg['classes'])
    plans = make_plan(cfg, pools)
    summary = dict(base_dataset=cfg['base_dataset'], output=str(dest), seed=cfg['generation_seed'],
                   labels={'PF+MB+RV':7}, count=3, export_wavs=cfg['triple']['export_wavs'],
                   splits={s:dict(n=len(part), contexts=dict(Counter(r['context_condition'] for r in part)),
                                  source_pool={c:len(p) for c,p in pools[s].items()}) for s,part in plans.items()})
    print(summary, flush=True)
    if plan_only:
        return summary
    source_map = {r['source_id']:r for r in sources}
    # Existing outputs are never overwritten. Only the new extension is written.
    dest.mkdir(parents=True, exist_ok=False)
    marker = dest/'.generating'
    marker.write_text('Triple generation in progress', encoding='utf-8')
    write(dest/'CONFIG.json', cfg)
    final_index = []
    for split, records in plans.items():
        folder = dest/'data'/split
        folder.mkdir(parents=True)
        if cfg['triple']['export_wavs']:
            (folder/'wav').mkdir()
        audio = np.lib.format.open_memmap(folder/'audio.npy', mode='w+', dtype=np.float32,
                                         shape=(len(records), int(cfg['sample_rate']*cfg['duration_seconds'])))
        cache = {}
        for i, row in enumerate(records):
            wave, detail = render(row, source_map, cfg, cache)
            audio[i] = wave
            if row['wav_path']:
                sf.write(dest/row['wav_path'], wave, cfg['sample_rate'], subtype='FLOAT')
            row.update(detail)
            final_index.append(row)
        audio.flush()
        del audio
        print(f'{split}: {len(records)} triples generated', flush=True)
    table(dest/'DATA_INDEX.csv', final_index)
    write(dest/'SUMMARY.json', summary)
    marker.unlink()
    print(f'Generated triple extension: {dest}', flush=True)
    return dest
