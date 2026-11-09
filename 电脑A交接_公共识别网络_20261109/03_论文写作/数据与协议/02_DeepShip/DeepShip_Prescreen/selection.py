"""Deterministic, recording-diverse budgets after acoustic screening."""
from collections import defaultdict
import math
import numpy as np

SPLITS = ("Train", "Val", "Test")


def truth(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def apportion(total, fractions):
    values = np.asarray(fractions, dtype=float)
    exact = int(total) * values / values.sum()
    counts = np.floor(exact).astype(int)
    order = sorted(range(len(values)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:int(total) - int(counts.sum())]:
        counts[i] += 1
    return counts.tolist()


def quality(row):
    try:
        value = float(row.get("quality_score") or 0)
        return value if math.isfinite(value) else 0
    except (TypeError, ValueError):
        return 0


def choose_windows(rows, cfg):
    """Mark rows in place; no failed gate is relaxed to fill a budget."""
    cap = int(cfg["selection"]["max_per_recording"])
    by_record = defaultdict(list)
    for row in rows:
        row["selected"] = 0
        if truth(row["candidate_pass"]):
            by_record[(row["class_name"], row.get("split_group_id") or row["recording_id"])].append(row)
        row["selection_reason"] = "screen_rejected"
    queues = defaultdict(dict)
    for (name, recording), group in sorted(by_record.items()):
        # Spread over the complete ordered candidate region; prefer quality inside each bin.
        ordered = sorted(group, key=lambda r: (float(r["start_seconds"]), r["source_id"]))
        positions = np.array_split(np.arange(len(ordered)), min(cap, len(ordered)))
        kept = [max((ordered[int(i)] for i in ids), key=lambda r: (quality(r), -float(r["start_seconds"]))) for ids in positions]
        chosen = {r["source_id"] for r in kept}
        for row in group:
            row["selection_reason"] = "class_split_cap" if row["source_id"] in chosen else "recording_cap"
        # A file group can contain multiple WAVs; its joint cap remains the same.
        for split in SPLITS:
            part = [r for r in kept if r["split"] == split]
            if part:
                queues[(name, split)][recording] = sorted(part, key=lambda r: (-quality(r), float(r["start_seconds"])))
    fractions = [cfg["split"]["fractions"][s] for s in SPLITS]
    caps = apportion(int(cfg["selection"]["max_per_class"]), fractions)
    for class_index, name in enumerate(cfg["classes"]):
        for split_index, split in enumerate(SPLITS):
            available = queues.get((name, split), {})
            rng = np.random.default_rng([int(cfg["split"]["seed"]), class_index, split_index, 73])
            ids = sorted(available)
            ids = [ids[int(i)] for i in rng.permutation(len(ids))]
            selected = 0
            # Round robin gives different recordings a turn before reusing one.
            while selected < caps[split_index]:
                moved = False
                for recording in ids:
                    if selected >= caps[split_index]:
                        break
                    if available[recording]:
                        row = available[recording].pop(0)
                        row["selected"] = 1
                        row["selection_reason"] = "selected"
                        selected += 1
                        moved = True
                if not moved:
                    break
    return rows
