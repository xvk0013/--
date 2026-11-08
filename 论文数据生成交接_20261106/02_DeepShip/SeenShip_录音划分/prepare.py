"""Prepare a recording-disjoint DeepShip split with evaluation ship names covered in Train."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


HERE = Path(__file__).resolve().parent
SCANS = HERE.parent.parent / "DeepShip_Prescreen/outputs"
GENERATOR = Path("D:/LSJ/Code/UART/deepship")
BASELINE = Path("D:/LSJ/Data/DeepShip_Multitarget_0dB_Energy80_v1")
VERSION = "deepship_multitarget_0db_energy80_seen_ship_v1"
OUTPUT = "D:/LSJ/Data/DeepShip_Multitarget_0dB_Energy80_SeenShip_v1"
SOURCE_FOLDER = "sources/prescreen_energy80_seen_ship"
SPLITS = ("Train", "Val", "Test")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def solve(groups, caps, quotas, cap_per_group, seed):
    group_ids = sorted(groups)
    old_counts = Counter(groups[g]["split"] for g in group_ids)
    group_count = len(group_ids)
    by_name = defaultdict(list)
    lower, upper = [], []
    constraint_rows = []

    def constrain(coefficients, lo, hi):
        row = np.zeros(group_count * len(SPLITS))
        for index, value in coefficients:
            row[index] = value
        constraint_rows.append(row)
        lower.append(lo)
        upper.append(hi)

    maximum = np.ones(group_count * len(SPLITS))
    cost = np.random.default_rng(seed).uniform(0, 0.0001, len(maximum))
    for index, group in enumerate(group_ids):
        info = groups[group]
        offset = index * len(SPLITS)
        constrain([(offset + j, 1) for j in range(len(SPLITS))], 1, 1)
        if info["name"]:
            by_name[info["name"]].append(offset)
        else:
            maximum[offset + 1:offset + len(SPLITS)] = 0
        for j, split in enumerate(SPLITS):
            if split != info["split"]:
                cost[offset + j] += 1000 if "Test" in (split, info["split"]) else 1

    for indices in by_name.values():
        constrain([(i, 1) for i in indices], 1, np.inf)
    for j, split in enumerate(SPLITS):
        constrain([(i * len(SPLITS) + j, 1) for i in range(group_count)],
                  old_counts[split], old_counts[split])
        constrain([(i * len(SPLITS) + j, min(caps[split][g], cap_per_group))
                   for i, g in enumerate(group_ids)], quotas[split], np.inf)
    result = milp(cost, integrality=np.ones(len(maximum)), bounds=Bounds(0, maximum),
                  constraints=LinearConstraint(np.stack(constraint_rows), lower, upper),
                  options={"time_limit": 45})
    if not result.success:
        raise RuntimeError(f"Cannot meet the existing recording and clip budgets: {result.message}")
    return dict(zip(group_ids, (SPLITS[i] for i in result.x.reshape(group_count, -1).argmax(1))))


def main():
    strict = SCANS / "strict_v1/full"
    energy80 = SCANS / "energy80_v1/full"
    records = read_csv(energy80 / "RECORDINGS.csv")
    index80 = read_csv(energy80 / "SOURCE_INDEX.csv")
    index40 = read_csv(strict / "SOURCE_INDEX.csv")
    baseline_selected = {
        (r["split"], r["source1_id"]) for r in read_csv(BASELINE / "DATA_INDEX.csv")
        if r["kind"] == "single"
    }
    cfg = read_json(GENERATOR / "config_energy80.json")
    group_meta = {}
    for record in records:
        name = " ".join(record["ship_name"].casefold().split()) if record["metadata_status"] == "matched" else ""
        group = record["split_group_id"]
        info = dict(name=name, split=record["split"], class_name=record["class_name"])
        if group in group_meta and group_meta[group] != info:
            raise ValueError(f"Ambiguous metadata for recording group {group}")
        group_meta[group] = info
    cap80 = Counter(r["split_group_id"] for r in index80 if r["candidate_pass"] == "1")
    cap40 = Counter(r["split_group_id"] for r in index40 if r["candidate_pass"] == "1")
    capacities = dict(Train=cap80, Val=cap40, Test=cap40)
    assignment, summary = {}, {}
    for i, class_name in enumerate(cfg["classes"]):
        groups = {k: v for k, v in group_meta.items() if v["class_name"] == class_name}
        current = solve(groups, capacities, cfg["selection"]["per_class"],
                        cfg["selection"]["max_per_recording"], cfg["seed"] + i)
        assignment.update(current)
        train_names = {groups[g]["name"] for g, split in current.items() if split == "Train"}
        summary[class_name] = {
            "groups": dict(Counter(current.values())),
            "moved": dict(Counter(f"{groups[g]['split']}->{s}" for g, s in current.items()
                                  if groups[g]["split"] != s)),
            "name_counts": {
                s: len({groups[g]["name"] for g, split in current.items() if split == s and groups[g]["name"]})
                for s in SPLITS
            },
            "unmatched_eval": sum(groups[g]["name"] not in train_names or not groups[g]["name"]
                                  for g, s in current.items() if s != "Train"),
        }
    if any(counts["unmatched_eval"] for counts in summary.values()):
        raise RuntimeError("Evaluation contains a ship name that is absent from Train.")

    stage = HERE / SOURCE_FOLDER
    stage.mkdir(parents=True, exist_ok=True)
    chosen = []
    for r in index80:
        if assignment[r["split_group_id"]] == "Train":
            chosen.append(r)
    for r in index40:
        if assignment[r["split_group_id"]] != "Train":
            chosen.append(r)
    counts = Counter()
    for row in chosen:
        split = assignment[row["split_group_id"]]
        row["split"] = split
        keep = (split, row["source_id"]) in baseline_selected and row["candidate_pass"] == "1"
        row["selected"] = int(keep)
        row["selection_reason"] = "retained_baseline" if keep else (
            "available" if row["candidate_pass"] == "1" else "screen_rejected"
        )
        if keep:
            counts[row["raw_relative_path"]] += 1
    path_caps = Counter(r["raw_relative_path"] for r in chosen if r["candidate_pass"] == "1")
    partition = []
    for record in records:
        key = record["split_group_id"]
        new_split = assignment[key]
        partition.append(dict(recording_id=record["recording_id"], split=new_split,
                              prior_split=record["split"], class_name=record["class_name"],
                              ship_name=group_meta[key]["name"]))
        record["split"] = new_split
        record["windows_candidate_pass"] = path_caps[record["raw_relative_path"]]
        record["windows_selected"] = counts[record["raw_relative_path"]]
    write_csv(stage / "SOURCE_INDEX.csv", chosen)
    write_csv(stage / "RECORDINGS.csv", records)
    write_csv(HERE / "PARTITION.csv", partition)
    write_json(HERE / "PARTITION_SUMMARY.json", summary)
    cfg.update(version=VERSION, output_directory=OUTPUT, smoke_output_directory=OUTPUT + "_smoke",
               sources=dict(index_file=f"{SOURCE_FOLDER}/SOURCE_INDEX.csv",
                            recordings_file=f"{SOURCE_FOLDER}/RECORDINGS.csv"))
    write_json(HERE / "config_energy80_seen_ship.json", cfg)
    q0 = read_json(GENERATOR / "Q0_DeepShip/config_energy80.json")
    q0.update(data_root=OUTPUT, dataset_version=VERSION)
    write_json(HERE / "q0_config_energy80_seen_ship.json", q0)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
