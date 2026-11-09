"""Screen source-conditioned projection directions with the sealed ShortLocal-GN."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import msvcrt
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
RESEARCH = HERE.parents[3] / 'final' / 'data_gen_v8_single_physical' / 'research'
SNAPSHOT = RESEARCH / 'shipsear_shortlocal_gn_freeze_20261108' / 'snapshot' / 'shipsear_finalization_20261103'
REFERENCE = SNAPSHOT / 'runs' / 'SE-Q0-ShortLocal-GN-13'
COMPONENTS = WORKSPACE / 'ShipsEar_DataGen_Final_0dB'
sys.path.insert(0, str(SNAPSHOT))
from train_formal import FixedDataset, ShortLocalSetModel, now, save_state, write_json
from short_local import ShortLocalViews

CLASSES = ('PF', 'MB', 'RV')
VIEWS = ('waveform', 'spectral')
LEVELS = (5, 0)
ARMS = ('shared', 'anchor')
RANK = 4
ATTENUATION = 0.5
WEAK_DB = -6
SWEEP_DB = 3
OUTPUT_RMS = 0.05
WEAK = 10 ** (WEAK_DB / 20)
LOW, HIGH = 10 ** (-SWEEP_DB / 20), 10 ** (SWEEP_DB / 20)
FIT = ((0, 0, 0), (1, 0, 0), (0, WEAK, 0), (1, WEAK, 0), (0, 0, WEAK), (1, 0, WEAK))
CHECK = ((1, 0, 0), (1, WEAK, 0), (1, 0, WEAK), (LOW, WEAK, 0), (HIGH, WEAK, 0),
         (LOW, 0, WEAK), (HIGH, 0, WEAK))
GATE = ('At 5 dB anchor-to-noise, the anchor-specific projection must capture a higher fraction of '
        'anchor-level variation than weak-class swaps in each view, and its mean selectivity across '
        'views must exceed the shared rank-matched projection. The 0 dB panel is descriptive.')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(out, event, **details):
    row = dict(time=now(), event=event, **details)
    write_json(out / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError('The frozen GN protocol requires CUDA.')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return load_json(REFERENCE / 'config_resolved.json')


class FrozenGn:
    """Observe the two original projections without replacing the GN forward."""

    def __init__(self, config):
        saved = torch.load(REFERENCE / 'best.pt', map_location='cpu', weights_only=False)
        seal = load_json(REFERENCE / 'selection_sealed.json')
        if (saved['experiment_id'], saved['epoch'], saved['score']) != (
                config['experiment_id'], seal['epoch'], seal['score']):
            raise ValueError('GN weights differ from the archived selection.')
        self.front = ShortLocalViews(config).eval().requires_grad_(False)
        self.model = ShortLocalSetModel(saved['config'], saved['normalization'])
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.cuda().eval().requires_grad_(False)
        self.interaction = self.model.tabm.backbone.blocks[0][0]

    @torch.no_grad()
    def encode(self, *features):
        projected = []

        def capture(_layer, _inputs, output):
            projected.append(output)

        with ExitStack() as stack, torch.autocast('cuda', dtype=torch.bfloat16):
            for layer in (self.interaction.left, self.interaction.right):
                stack.callback(layer.register_forward_hook(capture).remove)
            logits = self.model(*features)
        if len(projected) != 2 or any(v.shape != (*logits.shape[:2], 64) for v in projected):
            raise RuntimeError('Expected one 64-dimensional projection per view and GN member.')
        return *projected, logits

    def __call__(self, waves):
        return self.encode(*self.front(waves))


@dataclass(frozen=True)
class Source:
    id: str
    recording: str
    label: int
    path: str


@dataclass(frozen=True)
class Triplet:
    anchor: Source
    first: Source
    second: Source
    noise_seed: int


def rms(wave):
    return np.sqrt(np.mean(np.square(wave)))


class Panel:
    """One matched weak-class swap per distinct C0 anchor component."""

    def __init__(self, dataset, seed):
        self.split = dataset.split
        sources = {}
        for row in dataset.rows:
            if row['kind'] != 'double' or row['context_condition'] != 'C0':
                continue
            for side in (1, 2):
                stem = f'source{side}'
                source = Source(row[f'{stem}_id'], row[f'{stem}_recording_id'],
                                CLASSES.index(row[f'{stem}_class_name']), row[f'{stem}_component_path'])
                if sources.setdefault(source.id, source) != source:
                    raise ValueError(f'Conflicting component metadata for {source.id}.')
        by_class = [{s.recording: [] for s in sources.values() if s.label == label}
                    for label in range(len(CLASSES))]
        for source in sorted(sources.values(), key=lambda item: item.id):
            by_class[source.label][source.recording].append(source)

        rng = np.random.default_rng([seed, 850, int(self.split == 'Val')])
        self.triplets = []
        for anchor in sorted(sources.values(), key=lambda item: item.id):
            weak = []
            for label, recordings in enumerate(by_class):
                if label == anchor.label:
                    continue
                candidates = recordings[rng.choice(sorted(recordings))]
                weak.append(candidates[int(rng.integers(len(candidates)))])
            self.triplets.append(Triplet(anchor, *weak, int(rng.integers(2 ** 32))))
        self.labels = np.array([t.anchor.label for t in self.triplets])
        self.recordings = np.array([t.anchor.recording for t in self.triplets])
        self.component_cache = {}
        if not len(self.triplets):
            raise RuntimeError(f'No C0 triplets for {self.split}.')

    def component(self, source):
        if source.path not in self.component_cache:
            wave, rate = sf.read(COMPONENTS / source.path, dtype='float64')
            if rate != 16000 or wave.shape != (80000,) or not np.isfinite(wave).all() or rms(wave) == 0:
                raise ValueError(f'Invalid waveform component {source.path}.')
            self.component_cache[source.path] = wave / rms(wave)
        return self.component_cache[source.path]

    def waves(self, start, stop, level, variants):
        batches = []
        for triplet in self.triplets[start:stop]:
            a, b, c = (self.component(source) for source in (triplet.anchor, triplet.first, triplet.second))
            noise = np.random.default_rng(triplet.noise_seed).standard_normal(len(a))
            noise -= noise.mean()
            noise *= 10 ** (-level / 20) / rms(noise)
            gain = OUTPUT_RMS / rms(a + noise)
            batch = [(ca * a + cb * b + cc * c + noise) * gain for ca, cb, cc in variants]
            batches.extend(batch)
        return np.asarray(batches, dtype=np.float32)


def verify_replay(frozen, val, config):
    n = config['evaluation_batch_size']
    raw = [frozen.front(val.batch(np.arange(start, start + config['batch_size'])))
           for start in range(0, n, config['batch_size'])]
    _, _, logits = frozen.encode(*(torch.cat(parts) for parts in zip(*raw)))
    p = logits.float().softmax(-1).cpu().numpy().mean(1)
    with np.load(REFERENCE / 'Val_original.npz') as saved:
        np.testing.assert_array_equal(val.ids[:n], saved['ids'][:n])
        np.testing.assert_array_equal(p.argmax(-1), saved['predictions'][:n])
        np.testing.assert_allclose(p, saved['probabilities'][:n], rtol=1e-5, atol=1e-6)


def verify_pairs(train, val):
    if set(train.recordings) & set(val.recordings):
        raise ValueError('Train and Val source recordings overlap.')
    n, a, b, ab, c, ac = train.waves(0, 1, 5, FIT)
    np.testing.assert_allclose(a - n, ab - b, atol=1e-7)
    np.testing.assert_allclose(a - n, ac - c, atol=1e-7)
    a, ab, ac, low_b, high_b, low_c, high_c = val.waves(0, 1, 5, CHECK)
    np.testing.assert_allclose(high_b - low_b, high_c - low_c, atol=1e-7)
    np.testing.assert_allclose(rms(ab - a), rms(ac - a), rtol=1e-6)


def capture_panel(frozen, panel, config, level, variants):
    group_batch = config['batch_size'] // len(variants)
    pieces = ([], [], [])
    for start in range(0, len(panel.triplets), group_batch):
        waves = panel.waves(start, start + group_batch, level, variants)
        for item, tensor in zip(pieces, frozen(waves)):
            item.append(tensor.cpu().unflatten(0, (-1, len(variants))))
    return tuple(torch.cat(item) for item in pieces)


def covariance(deltas, recordings):
    counts = Counter(recordings)
    weights = torch.tensor([1 / counts[key] for key in recordings], dtype=torch.float64)
    weights /= weights.sum()
    rows = (deltas.double() * weights.sqrt()[:, None, None, None]).movedim(2, 0).flatten(1, 2)
    return (rows.transpose(-1, -2) @ rows) / deltas.shape[1]


def fit_bases(panel, cache):
    if panel.split != 'Train':
        raise ValueError('The projection bases must be fitted on Train.')
    result = {arm: {} for arm in ARMS}
    for i, view in enumerate(VIEWS):
        deltas = torch.cat([x[i][:, 1::2].float() - x[i][:, ::2].float() for x in cache.values()], dim=1)
        matrices = torch.stack([covariance(deltas[panel.labels == label], panel.recordings[panel.labels == label])
                                for label in range(len(CLASSES))])
        # Eigenvectors of the uncentered covariance are the right singular directions.
        result['anchor'][view] = torch.linalg.eigh(matrices).eigenvectors[..., -RANK:].float().contiguous()
        shared = torch.linalg.eigh(matrices.mean(0)).eigenvectors[..., -RANK:].float()
        result['shared'][view] = shared.expand_as(result['anchor'][view]).contiguous()
    return result


def fraction(delta, basis):
    coordinates = torch.einsum('ntkd,nkdr->ntkr', delta, basis)
    energy = delta.square().sum((1, 2, 3)).clamp_min(1e-20)
    return (coordinates.square().sum((1, 2, 3)) / energy).numpy()


def recording_means(values, panel):
    by_recording = {key: float(100 * values[panel.recordings == key].mean())
                    for key in np.unique(panel.recordings)}
    by_class = {name: float(np.mean([by_recording[key] for key in np.unique(panel.recordings[panel.labels == i])]))
                for i, name in enumerate(CLASSES)}
    return dict(macro=float(np.mean(list(by_class.values()))), by_class=by_class, by_recording=by_recording)


def screen(panel, cache, bases):
    measurements = {}
    for level, values in cache.items():
        arms = {}
        for arm, directions in bases.items():
            views = {}
            for view, features in zip(VIEWS, values):
                x = features.float()
                anchor_delta = torch.stack((x[:, 4] - x[:, 3], x[:, 6] - x[:, 5]), dim=1)
                swap_delta = (x[:, 1] - x[:, 2]).unsqueeze(1)
                basis = directions[view][panel.labels]
                anchor = fraction(anchor_delta, basis)
                swap = fraction(swap_delta, basis)
                views[view] = dict(anchor_energy=recording_means(anchor, panel),
                                   swap_energy=recording_means(swap, panel),
                                   selectivity=recording_means(anchor - swap, panel))
            arms[arm] = views
        measurements[f'{level}dB'] = arms
    primary = measurements['5dB']
    selectivity = {arm: {view: row['selectivity']['macro'] for view, row in views.items()}
                   for arm, views in primary.items()}
    view_checks = {view: score > 0 for view, score in selectivity['anchor'].items()}
    exceeds_shared = np.mean(list(selectivity['anchor'].values())) > np.mean(list(selectivity['shared'].values()))
    return dict(passed=bool(all(view_checks.values()) and exceeds_shared),
                primary_view_checks=view_checks, exceeds_shared_mean=bool(exceeds_shared),
                measurements=measurements)


def run():
    config = configure()
    out = HERE / 'runs' / f'SE-AnchorProjection-GN-{config["seed"]}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if (out / 'screen.json').is_file():
            print(f'Screen already recorded at {out}.', flush=True)
            return
        spec = dict(reference=config['experiment_id'], data_root=config['data_root'], seed=config['seed'],
                    component_root=str(COMPONENTS), levels_db=list(LEVELS), rank=RANK, attenuation=ATTENUATION,
                    weak_relative_db=WEAK_DB, anchor_sweep_db=SWEEP_DB, reference_rms=OUTPUT_RMS, gate=GATE)
        write_json(out / 'protocol.json', spec)
        frozen = FrozenGn(config)
        train_data, val_data = (FixedDataset(config['data_root'], split) for split in ('Train', 'Val'))
        train, val = (Panel(data, config['seed']) for data in (train_data, val_data))
        verify_replay(frozen, val_data, config)
        verify_pairs(train, val)
        panels = {panel.split: dict(triplets=len(panel.triplets), recordings=len(set(panel.recordings)))
                  for panel in (train, val)}
        write_json(out / 'panel_index.json', {p.split: [asdict(t) for t in p.triplets] for p in (train, val)})
        status(out, 'replay_and_pairs_verified', panels=panels)
        caches = {}
        for panel, variants in ((train, FIT), (val, CHECK)):
            cache = {}
            for level in LEVELS:
                start = time.perf_counter()
                cache[level] = capture_panel(frozen, panel, config, level, variants)
                status(out, 'features_cached', split=panel.split, level_db=level,
                       seconds=round(time.perf_counter() - start, 2))
            caches[panel.split] = cache
        bases = fit_bases(train, caches['Train'])
        result = screen(val, caches['Val'], bases)
        if result['passed']:
            save_state(out / 'screen_features.pt', dict(bases=bases, caches=caches))
        write_json(out / 'screen.json', dict(protocol=spec, panels=panels, **result))
        summary = {level: {arm: {view: tuple(round(row[key]['macro'], 2) for key in ('anchor_energy', 'swap_energy'))
                                  for view, row in views.items()} for arm, views in arms.items()}
                   for level, arms in result['measurements'].items()}
        status(out, 'screen_completed', passed=result['passed'], energy_percent=summary)


if __name__ == '__main__':
    run()
