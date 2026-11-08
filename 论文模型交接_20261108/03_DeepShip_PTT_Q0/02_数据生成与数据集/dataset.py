"""Model-independent reader for the generated DeepShip source-set task.

The default batch returns the stored waveform. Optional Single envelopes are
applied only when explicitly requested on Train. Additional AWGN is generated
online and is defined relative to the already recorded/propagated waveform.
"""
from itertools import combinations
import csv
import json
from pathlib import Path
import numpy as np


DEFAULT_CLASSES = ("Cargo", "Passengership", "Tanker", "Tug")


def set_schema(class_names=None):
    """Return label names, class-presence vectors and source counts."""
    classes = tuple(DEFAULT_CLASSES if class_names is None else class_names)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("At least two unique vessel classes are required.")
    sets = [()] + [(i,) for i in range(len(classes))] + list(combinations(range(len(classes)), 2))
    names = ["E"] + ["+".join(classes[i] for i in members) for members in sets[1:]]
    presence = np.zeros((len(sets), len(classes)), dtype=np.float32)
    for label, members in enumerate(sets):
        presence[label, list(members)] = 1
    return tuple(names), presence, np.asarray([len(s) for s in sets], dtype=np.int64)


def double_envelopes(row, cfg):
    """C0 full overlap, C1 one edge, C2 alternating edge dominance.

    ``first`` selects the source dominant at the beginning. ``side`` selects
    whether the C1-only transition is at the beginning (0) or end (1).
    Every envelope retains the weak source at ``mixture.epsilon``.
    """
    n = int(round(float(cfg["sample_rate"]) * float(cfg["duration_seconds"])))
    e = [np.ones(n, dtype=np.float64), np.ones(n, dtype=np.float64)]
    condition = row["context_condition"]
    if condition == "C0":
        return tuple(e)
    if condition not in ("C1", "C2"):
        raise ValueError("Double context_condition must be C0, C1 or C2.")
    first, side = int(row["first"]), int(row["side"])
    if first not in (0, 1) or side not in (0, 1):
        raise ValueError("Envelope first/side must be 0 or 1.")
    left, right, fl, fr = (float(row[k]) for k in ("left", "right", "fade_left", "fade_right"))
    if min(left, right, fl, fr) < 0:
        raise ValueError("Envelope intervals cannot be negative.")
    duration, fs = float(cfg["duration_seconds"]), float(cfg["sample_rate"])
    if left + fl + right + fr >= duration:
        raise ValueError("Envelope intervals leave no middle overlap.")
    eps = float(cfg["mixture"]["epsilon"])
    if not 0 <= eps <= 1:
        raise ValueError("mixture.epsilon must be between zero and one.")
    t = np.arange(n, dtype=np.float64) / fs
    if fl:
        rise = eps + (1-eps) * .5 * (1-np.cos(np.pi*np.clip((t-left)/fl, 0, 1)))
    else:
        rise = np.where(t < left, eps, 1.0)
    if fr:
        fall = eps + (1-eps) * .5 * (1+np.cos(np.pi*np.clip((t-(duration-right-fr))/fr, 0, 1)))
    else:
        fall = np.where(t < duration-right, 1.0, eps)
    if condition == "C2" or side == 0:
        e[1-first] = rise
    if condition == "C2" or side == 1:
        e[first] = fall
    return tuple(e)


def single_envelope(row, cfg):
    """Train-only Single view; base Single files remain unchanged."""
    kind = row.get("augmentation_type", "flat") or "flat"
    if kind == "flat":
        return np.ones(int(round(cfg["sample_rate"] * cfg["duration_seconds"])), dtype=np.float64)
    if kind not in ("edge", "middle"):
        raise ValueError("Unknown Single augmentation_type: " + str(kind))
    a, b = double_envelopes(dict(row, context_condition="C2"), cfg)
    return a*b if kind == "middle" else (a, b)[int(row["side"])]


class MixedDataset:
    """Read one fixed split using memory-mapped float32 waveforms."""
    def __init__(self, root, split="Train"):
        if split not in ("Train", "Val", "Test"):
            raise ValueError("split must be Train, Val or Test.")
        self.root, self.split = Path(root).resolve(), split
        self.cfg = json.loads((self.root/"CONFIG.json").read_text(encoding="utf-8-sig"))
        summary = json.loads((self.root/"SUMMARY.json").read_text(encoding="utf-8-sig"))
        if summary.get("status") != "complete":
            raise ValueError("Dataset generation has not completed.")
        self.class_names = tuple(self.cfg["classes"])
        self.set_names, membership, self.set_counts = set_schema(self.class_names)
        with (self.root/"DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.rows = sorted((r for r in csv.DictReader(stream) if r["split"] == split),
                               key=lambda r: int(r["array_index"]))
        self.labels = np.asarray([int(r["label_index"]) for r in self.rows], dtype=np.int64)
        self.counts = np.asarray([int(r["count"]) for r in self.rows], dtype=np.int64)
        if np.any(self.labels < 0) or np.any(self.labels >= len(self.set_names)):
            raise ValueError("DATA_INDEX label is outside the configured set schema.")
        if any(int(r["array_index"]) != i for i, r in enumerate(self.rows)):
            raise ValueError("DATA_INDEX array indices must be contiguous in each split.")
        if not np.array_equal(self.counts, self.set_counts[self.labels]):
            raise ValueError("DATA_INDEX count does not match its vessel-set label.")
        self.targets = membership[self.labels].copy()
        for i, row in enumerate(self.rows):
            if row.get("target") and not np.array_equal(np.asarray(json.loads(row["target"])), self.targets[i]):
                raise ValueError("DATA_INDEX presence target does not match its label.")
        self.ids = np.asarray([r["sample_id"] for r in self.rows], dtype=str)
        self.contexts = np.asarray([r.get("context_condition", "") for r in self.rows], dtype=str)
        self.audio = np.load(self.root/"data"/split/"audio.npy", mmap_mode="r")
        n = int(round(self.cfg["sample_rate"] * self.cfg["duration_seconds"]))
        if self.audio.dtype != np.float32 or self.audio.shape != (len(self.rows), n):
            raise ValueError("Stored waveform shape/dtype differs from DATA_INDEX and CONFIG.")

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, augment_single=False):
        """Return audio, class presence, set labels and counts as copied arrays."""
        if isinstance(indices, slice):
            indices = np.arange(len(self))[indices]
        ids = np.atleast_1d(indices).astype(np.int64)
        if np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError("Batch index is outside the requested split.")
        waves = np.array(self.audio[ids], dtype=np.float32, copy=True)
        enabled = bool(self.cfg.get("single_augmentation", {}).get("enabled", False))
        if self.split == "Train" and augment_single and enabled:
            for position, index in enumerate(ids):
                row = self.rows[int(index)]
                if self.counts[index] == 1 and row.get("augmentation_type", "flat") != "flat":
                    view = waves[position].astype(np.float64) * single_envelope(row, self.cfg)
                    level = np.sqrt(np.mean(view*view))
                    if not np.isfinite(level) or level <= 0:
                        raise ValueError("Single training envelope produced invalid waveform energy.")
                    waves[position] = np.asarray(view * (float(self.cfg["output_rms"])/level), dtype=np.float32)
        return waves, self.targets[ids].copy(), self.labels[ids].copy(), self.counts[ids].copy()


def recognition_metrics(labels, predictions, contexts=None, class_names=None):
    """Set accuracy and count errors for the configured 0/1/2-source task."""
    names, _, counts = set_schema(class_names)
    labels, predictions = np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("labels and predictions must be equal-length one-dimensional arrays.")
    if np.any(labels < 0) or np.any(labels >= len(names)) or np.any(predictions < 0) or np.any(predictions >= len(names)):
        raise ValueError("Label or prediction is outside the set schema.")
    context = np.full(len(labels), "", dtype=str) if contexts is None else np.asarray(contexts, dtype=str)
    if context.shape != labels.shape:
        raise ValueError("contexts must align with labels.")
    truth_count, predicted_count = counts[labels], counts[predictions]
    correct = labels == predictions
    def accuracy(mask):
        return float(100*correct[mask].mean()) if np.any(mask) else None
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    count_confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    np.add.at(count_confusion, (truth_count, predicted_count), 1)
    result = dict(n=len(labels), acc=accuracy(np.ones(len(labels), dtype=bool)),
        single_acc=accuracy(truth_count == 1), double_acc=accuracy(truth_count == 2), noise_acc=accuracy(truth_count == 0),
        count_acc=float(100*np.mean(truth_count == predicted_count)) if len(labels) else None,
        single_to_double=int(((truth_count == 1) & (predicted_count == 2)).sum()),
        double_to_single=int(((truth_count == 2) & (predicted_count == 1)).sum()),
        single_type_errors=int(((truth_count == 1) & (predicted_count == 1) & ~correct).sum()),
        double_pair_errors=int(((truth_count == 2) & (predicted_count == 2) & ~correct).sum()),
        confusion=confusion.tolist(), final_count_confusion=count_confusion.tolist(),
        per_set_acc={name: accuracy(labels == i) for i, name in enumerate(names)})
    for condition in ("C0", "C1", "C2"):
        result["double_"+condition+"_acc"] = accuracy((truth_count == 2) & (context == condition))
    return result


def add_noise(waves, snr_db, seed, background_mask=None, noise_reference_rms=.05,
              sample_ids=None, output_rms=.05):
    """Add online AWGN with deterministic noise per numeric sample ID.

    SNR compares added noise against the entire source waveform, which may
    already contain natural ambience. The final common RMS gain preserves
    this ratio. Reusing numeric IDs and seed reuses noise directions across SNRs.
    If IDs are omitted, rows are numbered within this call.
    """
    original = np.asarray(waves, dtype=np.float32)
    one_dimensional = original.ndim == 1
    x = np.atleast_2d(original)
    if original.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError("waves must be a finite waveform or batch.")
    if snr_db is None:
        return original.copy()
    n = len(x)
    ids = np.arange(n, dtype=np.int64) if sample_ids is None else np.asarray(sample_ids, dtype=np.int64)
    mask = np.zeros(n, dtype=bool) if background_mask is None else np.asarray(background_mask, dtype=bool)
    levels = np.broadcast_to(np.asarray(snr_db, dtype=np.float64), (n,))
    if ids.shape != (n,) or mask.shape != (n,) or np.any(ids < 0) or not np.isfinite(levels).all():
        raise ValueError("Noise IDs, background mask and SNRs must align with the batch.")
    if output_rms <= 0 or noise_reference_rms <= 0 or int(seed) < 0 or x.shape[1] < 2:
        raise ValueError("Noise levels, seed and waveform length are invalid.")
    result = np.empty_like(x)
    for i, wave in enumerate(x):
        random = np.random.default_rng(np.random.SeedSequence([int(seed), int(ids[i]), 811]))
        noise = random.standard_normal(len(wave))
        noise -= noise.mean()
        noise /= np.sqrt(np.mean(noise*noise))
        reference = float(noise_reference_rms) if mask[i] else float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
        view = wave.astype(np.float64) + noise * reference / (10**(levels[i]/20))
        rms = np.sqrt(np.mean(view*view))
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError("Cannot normalize the noisy waveform.")
        result[i] = view * (float(output_rms)/rms)
    return result[0] if one_dimensional else result
