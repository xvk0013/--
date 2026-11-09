"""Read-only recording inventory, metadata matching and recording-level splits.

The supplied ship-name JSON is only a diagnostic hint.  The original metafile
rows, waveform filename and duration determine whether a recording is matched.
No audio is decoded, copied or modified by this module.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf


SPLITS = ("Train", "Val", "Test")
DEFAULT_AIS = {
    "Cargo": list(range(70, 80)),
    "Passengership": list(range(60, 70)),
    "Tanker": list(range(80, 90)),
    "Tug": [31, 32, 52],
}


def _name(value: Any) -> str:
    return " ".join(str(value or "").upper().split())


def read_metadata(class_directory: Path) -> list[dict[str, Any]]:
    """Read the dataset's headerless ``*-metafile`` CSV, retaining row identity."""
    tables = sorted(
        p for p in class_directory.iterdir()
        if p.is_file() and "metafile" in p.name.lower()
    )
    if not tables:
        return []
    if len(tables) != 1:
        raise ValueError(f"Expected one metafile in {class_directory}, found {len(tables)}")
    result = []
    with tables[0].open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or not any(value.strip() for value in row):
                continue
            if not row[0].strip().isdigit():
                if line_number == 1:
                    continue
                raise ValueError(f"Invalid metadata row id at {tables[0]}:{line_number}")
            if len(row) < 6:
                raise ValueError(f"Incomplete metadata row at {tables[0]}:{line_number}")
            try:
                duration = float(row[5].strip())
                ais = int(row[1].strip()) if row[1].strip() else None
            except ValueError as error:
                raise ValueError(f"Invalid metadata value at {tables[0]}:{line_number}") from error
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"Invalid metadata duration at {tables[0]}:{line_number}")
            result.append({
                "row_id": int(row[0].strip()),
                "ais_type": ais,
                "ship_name": row[2].strip(),
                "date": row[3].strip(),
                "time": row[4].strip().zfill(6),
                "duration_seconds": duration,
                "distance": row[6].strip() if len(row) > 6 else "",
                "metadata_line": line_number,
            })
    return result


def _is_time(value: str) -> bool:
    return (len(value) == 6 and value.isdigit()
            and int(value[:2]) < 24 and int(value[2:4]) < 60 and int(value[4:]) < 60)


def match_metadata(
    raw_relative_path: str,
    duration_seconds: float,
    rows: Sequence[Mapping[str, Any]],
    metadata_config: Mapping[str, Any],
    ship_hint: str | None = None,
) -> dict[str, Any]:
    """Match a row conservatively; timestamp+duration can correct folder dates.

    A six-digit filename is a clock time, not a row index.  A matching folder
    date takes priority among equally plausible timestamp/duration matches.
    Ship-name hints never resolve an otherwise ambiguous match.
    """
    path = Path(raw_relative_path)
    stem = path.stem
    date_match = re.match(r"(\d{8})", path.parent.name)
    folder_date = date_match.group(1) if date_match else ""
    tolerance = float(metadata_config.get("duration_tolerance_seconds", 2.0))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("metadata.duration_tolerance_seconds must be finite and nonnegative")

    def close(row: Mapping[str, Any]) -> bool:
        return abs(float(row["duration_seconds"]) - duration_seconds) <= tolerance

    def finish(candidates: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any] | None:
        if not candidates:
            return None
        same_date = [r for r in candidates if folder_date and r["date"] == folder_date]
        pool = same_date or list(candidates)
        if len(pool) != 1:
            return dict(metadata_status="ambiguous", metadata_match_method=method,
                        metadata_note=f"{len(pool)} plausible original metadata rows", matched_row=None)
        row = dict(pool[0])
        notes = []
        if folder_date and row["date"] != folder_date:
            notes.append("folder_date_mismatch")
        if ship_hint and _name(ship_hint) != _name(row["ship_name"]):
            notes.append("ship_mapping_mismatch")
        if not row["ship_name"]:
            notes.append("ship_name_missing")
        return dict(metadata_status="matched", metadata_match_method=method,
                    metadata_note=";".join(notes), matched_row=row)

    if _is_time(stem):
        hits = [r for r in rows if r["time"] == stem and close(r)]
        answer = finish(hits, "filename_time_and_duration")
        if answer is not None:
            return answer
        # A clock-bearing filename must not be ignored to pick an unrelated
        # event merely because its date and duration happen to agree.
        status = "duration_mismatch" if any(r["time"] == stem for r in rows) else "not_found"
        return dict(metadata_status=status, metadata_match_method="filename_time_and_duration",
                    metadata_note="No unique original event agrees with filename time and duration",
                    matched_row=None)

    if stem.isdigit() and len(stem) < 6:
        hits = [r for r in rows if int(r["row_id"]) == int(stem)]
        same_day_hits = [r for r in hits if close(r) and (not folder_date or r["date"] == folder_date)]
        answer = finish(same_day_hits, "filename_row_id_and_duration")
        if answer is not None:
            return answer
        # Some raw filenames use an earlier file counter, while the original
        # metadata row ids shift later in the table.  A unique date+duration
        # agreement is stronger than accidentally matching that counter on a
        # different day; this does not widen the duration tolerance.
        answer = finish([r for r in rows if folder_date and r["date"] == folder_date and close(r)],
                        "folder_date_and_duration")
        if answer is not None:
            if answer["matched_row"] is not None and answer["matched_row"]["row_id"] != int(stem):
                answer["metadata_note"] = ";".join(filter(None, [answer["metadata_note"], "filename_row_id_mismatch"]))
            return answer
        if hits:
            return dict(metadata_status="duration_mismatch", metadata_match_method="filename_row_id_and_duration",
                        metadata_note="No original event agrees with folder date, filename row id and duration",
                        matched_row=None)

    answer = finish([r for r in rows if folder_date and r["date"] == folder_date and close(r)],
                    "folder_date_and_duration")
    if answer is not None:
        return answer
    return dict(metadata_status="not_found", metadata_match_method="",
                metadata_note="No original metadata row matches this recording", matched_row=None)


def _apportion(total: int, fractions: Sequence[float]) -> list[int]:
    values = np.asarray(fractions, dtype=np.float64)
    if values.shape != (len(SPLITS),) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("split fractions must be three finite nonnegative values with positive sum")
    targets = total * values / values.sum()
    counts = np.floor(targets).astype(int)
    for i in sorted(range(len(SPLITS)), key=lambda i: (-(targets[i] - counts[i]), i))[:total - int(counts.sum())]:
        counts[i] += 1
    positive = np.flatnonzero(values > 0)
    if total >= len(positive):
        for i in positive:
            if counts[i] == 0:
                donors = [j for j in positive if counts[j] > 1]
                donor = max(donors, key=lambda j: (counts[j] - targets[j], counts[j], -j))
                counts[donor] -= 1
                counts[i] += 1
    return counts.tolist()


def assign_splits(records: list[dict[str, Any]], config: Mapping[str, Any]) -> None:
    """Assign complete parent recording groups, irrespective of later eligibility."""
    settings = config.get("split", {})
    recording_ids = {r["recording_id"] for r in records}
    groups = {r.get("split_group_id", r["recording_id"]) for r in records}
    partition_file = settings.get("partition_file")
    ownership: dict[str, str] = {}
    if partition_file:
        with Path(partition_file).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"recording_id", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("partition_file needs recording_id and split columns")
            for row in reader:
                recording_id = row["recording_id"].strip().replace("\\", "/")
                split = row["split"].strip()
                if split not in SPLITS:
                    raise ValueError(f"Invalid split {split!r} in partition_file")
                if recording_id in ownership and ownership[recording_id] != split:
                    raise ValueError(f"Recording occurs in multiple splits: {recording_id}")
                ownership[recording_id] = split
        missing, extra = recording_ids - ownership.keys(), ownership.keys() - recording_ids
        if missing or extra:
            raise ValueError(f"partition_file coverage differs: {len(missing)} missing, {len(extra)} unknown recording groups")
        grouped = {}
        for record in records:
            group = record.get("split_group_id", record["recording_id"])
            split = ownership[record["recording_id"]]
            if group in grouped and grouped[group] != split:
                raise ValueError(f"Confirmed metadata event occurs in multiple splits: {group}")
            grouped[group] = split
        ownership = grouped
    else:
        fractions = settings.get("fractions", {"Train": 0.7, "Val": 0.15, "Test": 0.15})
        values = [float(fractions.get(s, 0)) for s in SPLITS]
        seed = int(settings.get("seed", 42))
        for class_index, class_name in enumerate(config["classes"]):
            class_groups = sorted({r.get("split_group_id", r["recording_id"]) for r in records if r["class_name"] == class_name})
            counts = _apportion(len(class_groups), values)
            random = np.random.default_rng(np.random.SeedSequence([seed, class_index]))
            order = random.permutation(class_groups).tolist()
            cursor = 0
            for split, size in zip(SPLITS, counts):
                ownership.update({recording_id: split for recording_id in order[cursor:cursor + size]})
                cursor += size
    for record in records:
        record["split"] = ownership[record.get("split_group_id", record["recording_id"])]


def build_inventory(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one auditable row per raw WAV with a fixed recording-level split."""
    root = Path(config["raw_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DeepShip raw_root does not exist: {root}")
    classes = list(config["classes"])
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes must be a nonempty list of distinct directory names")
    metadata_cfg = config.get("metadata", {})
    mapping_file = metadata_cfg.get("mapping_file")
    mapping: dict[str, Any] = {}
    if mapping_file:
        mapping_path = Path(mapping_file)
        mapping_path = mapping_path if mapping_path.is_absolute() else root / mapping_path
        if mapping_path.is_file():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    excluded = {str(v).replace("\\", "/").strip("/") for v in metadata_cfg.get("exclude_recordings", [])}
    require_match = bool(metadata_cfg.get("require_match", True))
    allowed = metadata_cfg.get("allowed_ais_codes", DEFAULT_AIS)
    channel = int(config.get("channel", 0))
    records: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        directory = root / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {directory}")
        meta_rows = read_metadata(directory)
        paths = sorted((p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
                       key=lambda p: p.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            parent = path.parent.relative_to(root).as_posix()
            recording_id = parent if path.parent != directory else f"{class_name}/{path.stem}"
            row: dict[str, Any] = dict(recording_id=recording_id, split_group_id=recording_id, class_name=class_name, class_index=class_index,
                raw_relative_path=relative, raw_path=str(path), sample_rate=0, channels=0, subtype="", frames=0,
                duration_seconds=0.0, split="", eligible=1, exclusion_reason="", ship_name="",
                metadata_status="not_checked", metadata_match_method="", metadata_note="")
            reasons: list[str] = []
            try:
                info = sf.info(str(path))
                row.update(sample_rate=int(info.samplerate), channels=int(info.channels), subtype=str(info.subtype),
                           frames=int(info.frames), duration_seconds=float(info.duration))
                if info.frames <= 0 or info.samplerate <= 0:
                    reasons.append("empty_recording")
                if channel < 0 or channel >= info.channels:
                    reasons.append("channel_unavailable")
            except (RuntimeError, OSError) as error:
                reasons.append("audio_header_error")
                row["metadata_note"] = f"{type(error).__name__}: {error}"
            hint = mapping.get(class_name, {}).get(path.parent.name) if isinstance(mapping.get(class_name, {}), dict) else None
            if not reasons:
                matched = match_metadata(relative, row["duration_seconds"], meta_rows, metadata_cfg, hint)
                match_row = matched.pop("matched_row")
                row.update(matched)
                if match_row is not None:
                    row.update(ship_name=match_row["ship_name"], metadata_row_id=match_row["row_id"],
                               ais_type=match_row["ais_type"], metadata_line=match_row["metadata_line"], metadata_date=match_row["date"],
                               metadata_time=match_row["time"], metadata_duration_seconds=match_row["duration_seconds"],
                               distance=match_row["distance"])
                    if class_name in allowed and match_row["ais_type"] not in allowed[class_name]:
                        reasons.append("ais_class_conflict")
                    elif metadata_cfg.get("group_matched_event", True):
                        row["split_group_id"] = f"{class_name}/event_{match_row['date']}_{match_row['time']}"
                elif require_match:
                    reasons.append("metadata_" + row["metadata_status"])
            if recording_id in excluded or relative in excluded:
                reasons.append("configured_exclusion")
            row["eligible"] = int(not reasons)
            row["exclusion_reason"] = ";".join(reasons)
            records.append(row)
    if not records:
        raise ValueError(f"No WAV recordings found in configured classes below {root}")
    assign_splits(records, config)
    return records
