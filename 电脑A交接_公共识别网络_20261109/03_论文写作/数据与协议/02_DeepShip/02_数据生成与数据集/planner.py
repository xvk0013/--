"""Deterministic source selection and mixture plans, without reading audio.

The source table is the completed prescreen's SOURCE_INDEX.csv. Existing
selected intervals are retained by default. All later derived samples inherit
their source recording's split; neither labels nor split ownership are inferred
from recognition scores.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np


SPLITS = ("Train", "Val", "Test")


def _truth(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _rng(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, parts)]))


def _apportion(total, weights):
    values = np.asarray(weights, dtype=float)
    if total < 0 or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Counts and sampling weights must be nonnegative and finite.")
    exact = int(total) * values / values.sum()
    result = np.floor(exact).astype(int)
    order = sorted(range(len(values)), key=lambda i: (-(exact[i] - result[i]), i))
    for index in order[:int(total) - int(result.sum())]:
        result[index] += 1
    return result.tolist()


def _quality(row):
    try:
        value = float(row.get("quality_score") or 0)
        return value if math.isfinite(value) else 0
    except (ValueError, TypeError):
        return 0


def _group(row):
    return row.get("split_group_id") or row["recording_id"]


def _read_sources(cfg):
    path = Path(cfg["sources"]["index_file"])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    expected = int(cfg["sample_rate"] * cfg["duration_seconds"])
    classes = list(cfg["classes"])
    seen, ownership, candidates = set(), {}, []
    for original in records:
        row = dict(original)
        source_id = row["source_id"]
        if source_id in seen:
            raise ValueError("Duplicate prescreen source_id: " + source_id)
        seen.add(source_id)
        if row["class_name"] not in classes:
            continue
        if row["split"] not in SPLITS:
            raise ValueError("Unknown source split: " + row["split"])
        parent = _group(row)
        if parent in ownership and ownership[parent] != row["split"]:
            raise ValueError("Source recording/event crosses splits: " + parent)
        ownership[parent] = row["split"]
        if _truth(row.get("selected", 0)) and not _truth(row["candidate_pass"]):
            raise ValueError("Existing selected source failed the prescreen: " + source_id)
        if not _truth(row["candidate_pass"]):
            continue
        if int(row["target_sample_rate"]) != int(cfg["sample_rate"]) or int(row["num_samples"]) != expected:
            raise ValueError("Prescreen source length/rate differs from generator: " + source_id)
        row["start_sample"] = int(row["start_sample"])
        if int(row["end_sample"]) - row["start_sample"] != expected or row["start_sample"] < 0:
            raise ValueError("Invalid source interval: " + source_id)
        row["raw_relative_path"] = row["raw_relative_path"].replace("\\", "/")
        relative = Path(row["raw_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Expected a relative raw source path: " + source_id)
        candidates.append(row)
    if not candidates:
        raise ValueError("The prescreen contains no matching candidate_pass sources.")
    return candidates


def _select_group(pool, target, cap, preserve, random):
    """Preserve the old subset, then supplement with recording-diverse intervals."""
    by_record = defaultdict(list)
    for row in pool:
        by_record[_group(row)].append(row)
    bases = [row for row in pool if preserve and _truth(row.get("selected", 0))]
    if len(bases) > target:
        raise ValueError(f"Existing selected count {len(bases)} exceeds requested quota {target}; use a new plan with preserve_existing_selected=false to deliberately select a smaller subset.")
    counts = Counter(_group(row) for row in bases)
    if any(value > cap for value in counts.values()):
        raise ValueError("Existing selection exceeds selection.max_per_recording.")
    capacity = sum(min(cap, len(rows)) for rows in by_record.values())
    if capacity < target:
        raise ValueError(f"Qualified source capacity is {capacity}, requested {target}; retain the screening rules and lower the quota or explicitly increase the recording cap.")
    chosen = {row["source_id"] for row in bases}
    output = list(bases)
    positions, active, chosen_positions = {}, {}, {}
    for recording, rows in by_record.items():
        ordered = sorted(rows, key=lambda row: (row["raw_relative_path"], row["start_sample"], row["source_id"]))
        positions[recording] = {row["source_id"]: i for i, row in enumerate(ordered)}
        active[recording] = [row for row in ordered if row["source_id"] not in chosen]
        chosen_positions[recording] = [positions[recording][row["source_id"]] for row in ordered if row["source_id"] in chosen]
    names = sorted(by_record)
    names = [names[int(i)] for i in random.permutation(len(names))]
    priority = {name: i for i, name in enumerate(names)}
    while len(output) < target:
        available = [name for name in names if active[name] and counts[name] < cap]
        available.sort(key=lambda name: (counts[name] > 0, counts[name], priority[name]))
        if not available:
            raise RuntimeError("Source capacity accounting and allocation disagree.")
        for recording in available:
            if len(output) >= target:
                break
            occupied = chosen_positions[recording]
            def score(row):
                position = positions[recording][row["source_id"]]
                distance = min(abs(position-old) for old in occupied) if occupied else 0
                return (distance, _quality(row), -position)
            row = max(active[recording], key=score)
            active[recording].remove(row)
            output.append(row)
            counts[recording] += 1
            occupied.append(positions[recording][row["source_id"]])
    return sorted(output, key=lambda row: row["source_id"])


def _empty_row(cfg, split, kind, label, label_name, target):
    row = dict(sample_id="", split=split, array_index=-1, kind=kind,
        label_index=int(label), label_name=label_name, count=int(sum(target)),
        target=json.dumps(list(map(int, target)), separators=(",", ":")),
        context_condition="S" if kind == "single" else "E" if kind == "background" else "C0",
        first=0, side=0, left=0.0, right=0.0, fade_left=0.0, fade_right=0.0,
        sir_db=float(cfg["mixture"]["sir_db"]), augmentation_type="flat", noise_index=-1, noise_seed=0,
        sample_rate=int(cfg["sample_rate"]), num_samples=int(cfg["sample_rate"]*cfg["duration_seconds"]))
    for number in (1, 2):
        prefix = f"source{number}_"
        row.update({prefix+key: "" for key in ("id", "recording_id", "group_id", "class_name", "raw_relative_path")})
        row.update({prefix+key: -1 for key in ("start_sample", "component_index")})
        row[prefix+"channel_id"] = 0
    return row


def _set_source(row, number, source):
    prefix = f"source{number}_"
    mapping = dict(id="source_id", recording_id="recording_id", group_id="split_group_id",
        class_name="class_name", raw_relative_path="raw_relative_path", start_sample="start_sample",
        component_index="component_index", channel_id="channel_id")
    for destination, key in mapping.items():
        row[prefix+destination] = source.get(key, _group(source) if key == "split_group_id" else "")


def _envelope(row, cfg, random):
    mixture = cfg["mixture"]
    for key in ("left", "right"):
        row[key] = float(random.uniform(*mixture["context_seconds"]))
    for key in ("fade_left", "fade_right"):
        row[key] = float(random.uniform(*mixture["fade_seconds"]))
    row["first"], row["side"] = int(random.integers(2)), int(random.integers(2))


def _unique_balanced_pairs(left, right, total, random):
    """Enumerate an injective Latin-style pairing with balanced endpoint counts.

    LCM blocks enumerate disjoint residue classes. Thus the complete n*m
    sequence contains every cross-class pair once, and either endpoint's
    prefix reuse counts differ by at most one.
    """
    n, m = len(left), len(right)
    if total > n*m:
        raise ValueError(f"Unique pair capacity {n*m} is smaller than requested {total}.")
    a = [left[int(i)] for i in random.permutation(n)]
    b = [right[int(i)] for i in random.permutation(m)]
    block = math.lcm(n, m)
    pairs = [(a[t % n], b[(t + t//block) % m]) for t in range(total)]
    random.shuffle(pairs)
    return pairs


def make_plan(cfg):
    """Return the complete final DATA_INDEX plan; this function writes no files."""
    classes = list(cfg["classes"])
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("At least two distinct ship classes are required.")
    sample_rate, duration = int(cfg["sample_rate"]), float(cfg["duration_seconds"])
    if sample_rate <= 0 or duration <= 0 or not math.isclose(sample_rate*duration, round(sample_rate*duration)):
        raise ValueError("Invalid sample rate or core duration.")
    seed = int(cfg["seed"])
    selection = cfg["selection"]
    cap = int(selection["max_per_recording"])
    if cap <= 0:
        raise ValueError("selection.max_per_recording must be positive.")
    preserve = bool(selection.get("preserve_existing_selected", True))
    pool = _read_sources(cfg)
    by_split_class = defaultdict(list)
    for row in pool:
        by_split_class[(row["split"], row["class_name"])].append(row)
    picked = {}
    for si, split in enumerate(SPLITS):
        target = int(selection["per_class"][split])
        if target <= 0:
            raise ValueError("Every split's per-class source quota must be positive.")
        for ci, name in enumerate(classes):
            try:
                picked[(split, name)] = _select_group(by_split_class[(split, name)], target, cap,
                    preserve, _rng(seed, 113, si, ci))
            except ValueError as error:
                raise ValueError(f"{name}/{split}: {error}") from error
    component_index = 0
    channel_count = int(cfg["bellhop"]["channel_count"])
    enabled = bool(cfg["bellhop"].get("enabled", True))
    if enabled and channel_count <= 0:
        raise ValueError("bellhop.channel_count must be positive.")
    for si, split in enumerate(SPLITS):
        for ci, name in enumerate(classes):
            sources = picked[(split, name)]
            random = _rng(seed, 271, si, ci)
            channels = []
            if enabled:
                while len(channels) < len(sources):
                    channels.extend((1+random.permutation(channel_count)).tolist())
            else:
                channels = [0]*len(sources)
            for source, channel in zip(sources, channels):
                source["component_index"], source["channel_id"] = component_index, int(channel)
                component_index += 1
    result = []
    pair_types = list(combinations(range(len(classes)), 2))
    mixture = cfg["mixture"]
    for si, split in enumerate(SPLITS):
        rows = []
        for ci, name in enumerate(classes):
            sources = picked[(split, name)]
            random = _rng(seed, 311, si, ci)
            augmentation = cfg["single_augmentation"]
            counts = _apportion(len(sources), [augmentation[k] for k in ("flat", "edge", "middle")])
            schedule = [kind for kind, n in zip(("flat", "edge", "middle"), counts) for _ in range(n)]
            random.shuffle(schedule)
            for source, augmentation_type in zip(sources, schedule):
                target = [int(index == ci) for index in range(len(classes))]
                row = _empty_row(cfg, split, "single", ci+1, name, target)
                _set_source(row, 1, source)
                row["sample_id"] = f"{split}_S{len(rows):06d}"
                if split == "Train" and bool(cfg["single_augmentation"].get("enabled", True)):
                    row["augmentation_type"] = augmentation_type
                    _envelope(row, cfg, random)
                rows.append(row)
        single_count = len(rows)
        double_count = int(round(single_count*float(mixture["double_per_single"])))
        background_count = int(round(single_count*float(mixture["background_per_single"])))
        if double_count < 0 or background_count < 0:
            raise ValueError("Mixture sample ratios cannot be negative.")
        pair_counts = _apportion(double_count, [1]*len(pair_types))
        double_index = 0
        for pi, ((a, b), count) in enumerate(zip(pair_types, pair_counts)):
            random = _rng(seed, 419, si, pi)
            pairs = _unique_balanced_pairs(picked[(split, classes[a])], picked[(split, classes[b])], count, random)
            conditions = ("C0", "C1", "C2")
            condition_counts = _apportion(count, [mixture["context_ratio"][c] for c in conditions])
            schedule = [condition for condition, size in zip(conditions, condition_counts) for _ in range(size)]
            random.shuffle(schedule)
            target = [int(index in (a, b)) for index in range(len(classes))]
            for (left, right), condition in zip(pairs, schedule):
                row = _empty_row(cfg, split, "double", 1+len(classes)+pi,
                    classes[a]+"+"+classes[b], target)
                row["sample_id"] = f"{split}_D{double_index:06d}"
                double_index += 1
                row["context_condition"] = condition
                _set_source(row, 1, left); _set_source(row, 2, right)
                _envelope(row, cfg, random)
                rows.append(row)
        random = _rng(seed, 557, si)
        for index in range(background_count):
            row = _empty_row(cfg, split, "background", 0, "E", [0]*len(classes))
            row["sample_id"] = f"{split}_E{index:06d}"
            row["noise_index"] = index
            row["noise_seed"] = int(random.integers(0, 2**32, dtype=np.uint64))
            rows.append(row)
        for index, row in enumerate(rows):
            row["array_index"] = index
        result.extend(rows)
    return result
