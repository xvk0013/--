"""Extend the background-swap dataset with a PF recording exchange."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE / "data"
BANK = Path("D:/LSJ/Code/UART/shipsear/ShipsEar_DataGen_Final_0dB")
SPLITS = ("Train", "Val", "Test")
SOURCE_FIELDS = (
    "id", "recording_id", "parent_core_id", "start_sample", "stop_sample",
    "channel_id", "class_name", "component_path",
)

sys.path.insert(0, str(BANK / "generator"))
from generate import render, source_map_from_index


def build_dataset(base, destination, swap, expected_owners, class_name, version, split_rule):
    config = json.loads((base / "CONFIG.json").read_text(encoding="utf-8"))
    with (base / "DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, original = reader.fieldnames, list(reader)

    singles = {
        recording: sorted(
            (row for row in original if row["kind"] == "single" and row["source1_recording_id"] == recording),
            key=lambda row: int(row["source1_start_sample"]),
        )
        for recording in swap
    }
    if any(not rows or any(row["label_name"] != class_name for row in rows) for rows in singles.values()):
        raise ValueError(f"The exchange requires {class_name} recordings with single-target cores.")
    donor_by_core = {
        outgoing["source1_id"]: incoming
        for recording, donor in swap.items()
        for outgoing, incoming in zip(singles[recording], singles[donor])
    }
    plans = {split: [] for split in SPLITS}
    changed = {split: Counter() for split in SPLITS}
    used = Counter()
    assigned_pairs = set()
    for original_row in original:
        row = dict(original_row)
        split = row["split"]
        wave_source = original_row
        for side in ("source1", "source2"):
            recording = row[f"{side}_recording_id"]
            if recording not in swap:
                continue
            donor = donor_by_core.get(row[f"{side}_id"])
            if row["kind"] == "double" and len(singles[recording]) != len(singles[swap[recording]]):
                partner = row["source2_id" if side == "source1" else "source1_id"]
                choices = [item for item in singles[swap[recording]]
                           if (item["source1_id"], partner) not in assigned_pairs]
                donor = min(choices, key=lambda item: used[item["source1_id"]])
                used[donor["source1_id"]] += 1
                assigned_pairs.add((donor["source1_id"], partner))
            if donor is None:
                changed[split]["removed_single"] += 1
                break
            for field in SOURCE_FIELDS:
                row[f"{side}_{field}"] = donor[f"source1_{field}"]
            donor_number = donor["source1_recording_id"].rsplit("_", 1)[-1]
            row["sample_id"] = f"{original_row['sample_id']}_{class_name}{donor_number}"
            if row["kind"] == "single":
                for field in ("source1_gain", "source2_gain", "common_gain", "output_rms", "peak", "wav_path"):
                    row[field] = donor[field]
                wave_source = donor
            else:
                row["wav_path"] = ""
                wave_source = None
            changed[split][row["kind"]] += 1
            plans[split].append((row, wave_source))
            break
        else:
            plans[split].append((row, wave_source))

    for outgoing, incoming in swap.items():
        split = singles[outgoing][0]["split"]
        donor_number = incoming.rsplit("_", 1)[-1]
        for donor in singles[incoming][len(singles[outgoing]):]:
            index_in_split = len(plans[split])
            row = dict(donor, split=split, sample_id=f"{split}_{index_in_split:06d}_{class_name}{donor_number}")
            plans[split].append((row, donor))
            changed[split]["added_single"] += 1

    for rows in plans.values():
        for index_in_split, (row, _) in enumerate(rows):
            row["array_index"] = str(index_in_split)

    index = [row for split in SPLITS for row, _ in plans[split]]
    source_map = source_map_from_index(index)
    owners, pairs = {}, set()
    for row in index:
        split = row["split"]
        for key in ("source1_recording_id", "source2_recording_id"):
            source = row[key]
            if source and owners.setdefault(source, split) != split:
                raise ValueError(f"Source recording crosses partitions: {source}.")
        if row["kind"] == "double":
            pair = (split, row["source1_id"], row["source2_id"])
            if pair in pairs:
                raise ValueError(f"Duplicate double-source pair: {pair}.")
            pairs.add(pair)
    if any(owners[source] != split for source, split in expected_owners.items()):
        raise ValueError("Unexpected recording exchange.")
    paired_cores = {row[f"source{k}_id"] for row in index if row["kind"] == "double" for k in (1, 2)}
    if any(row["source1_id"] not in paired_cores for rows in singles.values() for row in rows):
        raise ValueError("A moved core is absent from the new Double mixes.")

    destination.mkdir(exist_ok=False)
    source_arrays = {split: np.load(base / "data" / split / "audio.npy", mmap_mode="r") for split in SPLITS}
    summary = {}
    for split in SPLITS:
        folder = destination / "data" / split
        folder.mkdir(parents=True)
        path = folder / "audio.npy"
        rows = plans[split]
        audio = np.lib.format.open_memmap(
            path, mode="w+", dtype=source_arrays[split].dtype, shape=(len(rows), source_arrays[split].shape[1])
        )
        cache = {}
        for index_in_split, (row, wave_source) in enumerate(rows):
            if wave_source is None:
                audio[index_in_split], details = render(row, source_map, config, cache)
                row.update(details)
            else:
                audio[index_in_split] = source_arrays[wave_source["split"]][int(wave_source["array_index"])]
        audio.flush()
        del audio
        summary[split] = {
            "samples": len(rows),
            "changed": dict(changed[split]),
            "by_set": dict(Counter(row["label_name"] for row, _ in rows)),
        }

    with (destination / "DATA_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index)
    config.update(
        version=version,
        output_directory=str(destination),
        base_dataset=str(base),
        split_rule=split_rule,
        recording_changes={
            **config.get("recording_changes", {}),
            **{source: [singles[source][0]["split"], split] for source, split in expected_owners.items()},
        },
    )
    for filename, data in (("CONFIG.json", config), ("SUMMARY.json", summary)):
        (destination / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement", type=int, choices=(53, 63), default=53)
    replacement = parser.parse_args().replacement
    incoming_id = f"ShipsEar_{replacement}"
    build_dataset(
        BASE, HERE / f"data_pf65_{replacement}",
        {incoming_id: "ShipsEar_65", "ShipsEar_65": incoming_id},
        {"ShipsEar_65": "Train", incoming_id: "Test"},
        "PF", f"shipsear_background_and_pf65_{replacement}_20261104",
        "Difficulty-guided background and PF recording exchanges.",
    )


if __name__ == "__main__":
    main()
