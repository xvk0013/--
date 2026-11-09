"""Automatic, read-only DeepShip source screening with plan/pilot/full modes."""
import os
for _thread_option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_option, "1")

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
import numpy as np
from inventory import build_inventory
from screening import scan_recording
from selection import choose_windows, truth

HERE = Path(__file__).resolve().parent
SPLITS = ("Train", "Val", "Test")
SOURCE_FIELDS = [
    "source_id", "recording_id", "split_group_id", "class_name", "class_index", "split", "raw_relative_path",
    "start_sample", "end_sample", "num_samples", "target_sample_rate", "raw_sample_rate",
    "start_seconds", "end_seconds", "candidate_pass", "selected", "exclusion_reason", "selection_reason",
    "energy_db", "energy_rank_fraction", "persistent_subwindows", "max_zero_run_ms",
    "transient_energy_fraction", "tonal_lines", "broadband_stability", "spectral_flatness",
    "spectral_pass", "quality_score",
]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=()):
    path = Path(path)
    keys = list(fields)
    for key in sorted({k for row in rows for k in row} - set(keys)):
        keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in clean(row).items()})
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg["raw_root"] = str(resolve_path(cfg["raw_root"], path.parent))
    cfg["output_root"] = str(resolve_path(cfg["output_root"], path.parent))
    partition = cfg["split"].get("partition_file")
    if partition:
        cfg["split"]["partition_file"] = str(resolve_path(partition, path.parent))
    if not Path(cfg["raw_root"]).is_dir():
        raise ValueError("raw_root is not an existing directory: " + cfg["raw_root"])
    names = cfg["classes"]
    if not names or len(names) != len(set(names)):
        raise ValueError("classes must contain unique names.")
    fractions = cfg["split"]["fractions"]
    if set(fractions) != set(SPLITS) or any(float(v) <= 0 for v in fractions.values()):
        raise ValueError("split.fractions needs positive Train/Val/Test entries.")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-6):
        raise ValueError("split.fractions must sum to one.")
    p = cfg["screening"]
    if not 0 < p["energy_keep_fraction"] <= 1:
        raise ValueError("energy_keep_fraction must be in (0,1].")
    if int(cfg["target_sample_rate"] * cfg["window_seconds"]) <= 0 or int(cfg["channel"]) < 0:
        raise ValueError("Invalid sample rate, duration or channel.")
    subwindows = cfg["window_seconds"] / p["persistence_subwindow_seconds"]
    if not math.isclose(subwindows, round(subwindows)) or not 0 < p["persistence_required"] <= subwindows:
        raise ValueError("Persistence subwindows must divide window_seconds, and required count must fit.")
    low, high = p["analysis_band_hz"]
    if not 0 < low < high < cfg["target_sample_rate"] / 2:
        raise ValueError("analysis_band_hz must lie strictly between zero and Nyquist.")
    if not 0 < p["max_transient_energy_fraction"] <= 1:
        raise ValueError("max_transient_energy_fraction must be in (0,1].")
    spectrum = p["spectrum"]
    if not 0 < int(spectrum["hop"]) <= int(spectrum["n_fft"]) <= int(cfg["target_sample_rate"] * cfg["window_seconds"]):
        raise ValueError("spectrum hop/n_fft must fit the source window.")
    if int(spectrum["local_median_bins"]) < 3 or int(spectrum["local_median_bins"]) % 2 != 1:
        raise ValueError("spectrum.local_median_bins must be an odd number >=3.")
    if not 0 <= spectrum["tonal_persistence"] <= 1 or not 0 <= spectrum["max_broadband_flatness"] <= 1:
        raise ValueError("Spectrum persistence/flatness must lie in [0,1].")
    if not -1 <= spectrum["min_broadband_stability"] <= 1:
        raise ValueError("min_broadband_stability must lie in [-1,1].")
    if any(int(cfg["selection"][k]) <= 0 for k in ("max_per_recording", "max_per_class")):
        raise ValueError("Selection budgets must be positive.")
    return cfg


def safe_output(directory, raw_root):
    directory = Path(directory).resolve()
    if directory == Path(raw_root).resolve() or Path(raw_root).resolve() in directory.parents:
        raise ValueError("Output must be outside the original DeepShip directory.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def select_pilot(records, cfg, count):
    selected = []
    for class_index, name in enumerate(cfg["classes"]):
        groups = sorted({r.get("split_group_id") or r["recording_id"] for r in records if r["class_name"] == name and r["split"] == "Train" and truth(r["eligible"])})
        random = np.random.default_rng([int(cfg["split"]["seed"]), class_index, 107])
        wanted = {groups[int(i)] for i in random.permutation(len(groups))[:count]}
        selected.extend(r for r in records if r["class_name"] == name and (r.get("split_group_id") or r["recording_id"]) in wanted and truth(r["eligible"]))
    return selected


def scan_one(record, cfg):
    try:
        rows, diagnostic = scan_recording(record, cfg)
        return rows, diagnostic, None
    except Exception as error:
        return [], {}, f"{type(error).__name__}: {error}"


def stop_own_pool(pool):
    """Interrupt only this executor's workers; never wait for its queued corpus."""
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)
    pool.shutdown(wait=True, cancel_futures=True)


def scan_records(records, cfg, receive):
    """Bound submitted work and keep the current recording visible during waits."""
    workers = max(1, int(cfg["runtime"]["workers"]))
    heartbeat = max(.1, float(cfg["runtime"].get("heartbeat_seconds", 5)))
    if workers == 1:
        for record in records:
            begun = time.monotonic()
            finished = threading.Event()
            def show_wait():
                while not finished.wait(heartbeat):
                    print(f"[WAIT] {record['raw_relative_path']} running={time.monotonic()-begun:.1f}s (single worker)", flush=True)
            notifier = threading.Thread(target=show_wait, daemon=True)
            notifier.start()
            try:
                result = scan_one(record, cfg)
            finally:
                finished.set()
                notifier.join(timeout=1)
            receive(record, result)
        return
    timeout = float(cfg["runtime"].get("recording_timeout_seconds", 90))
    if timeout <= 0:
        raise ValueError("runtime.recording_timeout_seconds must be positive.")
    iterator = iter(records)
    pool = ProcessPoolExecutor(max_workers=workers)
    pending = {}
    last_message = time.monotonic()
    def submit_one():
        record = next(iterator, None)
        if record is not None:
            pending[pool.submit(scan_one, record, cfg)] = (record, time.monotonic())
    try:
        # At most one submitted recording per worker; no queue of 609 recordings.
        for _ in range(workers):
            submit_one()
        while pending:
            done, _ = wait(pending, timeout=min(heartbeat, 1.0), return_when=FIRST_COMPLETED)
            now = time.monotonic()
            for future in sorted(done, key=lambda f: pending[f][0]["raw_relative_path"]):
                record, begun = pending.pop(future)
                receive(record, future.result())
                last_message = time.monotonic()
                submit_one()
            late = [(record, now-begun) for record, begun in pending.values() if now-begun > timeout]
            if late:
                names = "; ".join(f"{record['raw_relative_path']} ({seconds:.1f}s)" for record, seconds in late)
                raise TimeoutError(f"Recording worker exceeded {timeout:g}s: {names}. Retry with -Workers 1 to bypass the process pool.")
            if pending and now-last_message >= heartbeat:
                names = "; ".join(f"{record['raw_relative_path']} {now-begun:.1f}s" for record, begun in pending.values())
                print(f"[WAIT] {names}", flush=True)
                last_message = now
    except BaseException:
        stop_own_pool(pool)
        raise
    else:
        pool.shutdown(wait=True)


def summarize(cfg, mode, records, rows, elapsed, errors=(), preview=None):
    def group_id(row):
        return row.get("split_group_id") or row["recording_id"]
    groups = []
    for name in cfg["classes"]:
        for split in SPLITS:
            rr = [r for r in records if r["class_name"] == name and r["split"] == split]
            ww = [r for r in rows if r["class_name"] == name and r["split"] == split]
            selected = [r for r in ww if truth(r.get("selected"))]
            capacities = defaultdict(int)
            for record in rr:
                if record.get("scan_status") in ("complete", "failed"):
                    capacities[group_id(record)] += int(record.get("windows_candidate_pass") or 0)
            amounts = list(capacities.values())
            groups.append(dict(class_name=name, split=split,
                recording_groups=len({group_id(r) for r in rr}),
                eligible_recording_groups=len({group_id(r) for r in rr if truth(r["eligible"])}),
                scanned_recording_groups=len({group_id(r) for r in rr if r.get("scan_status") == "complete"}),
                scanned_windows=len(ww), passed_windows=sum(truth(r["candidate_pass"]) for r in ww),
                selected_windows=len(selected), selected_recording_groups=len({group_id(r) for r in selected}),
                zero_candidate_recordings=sum(value == 0 for value in amounts),
                min_candidates_per_recording=min(amounts) if amounts else None,
                median_candidates_per_recording=float(np.median(amounts)) if amounts else None,
                max_candidates_per_recording=max(amounts) if amounts else None,
                min_positive_candidates_per_recording=min((value for value in amounts if value > 0), default=None)))
    reasons = Counter()
    for row in rows:
        if not truth(row["candidate_pass"]):
            reasons.update(v for v in row.get("exclusion_reason", "").split("|") if v)
    selected_n = sum(truth(r.get("selected")) for r in rows)
    return dict(status="complete_with_errors" if errors else "complete", mode=mode, version=cfg["version"], completed_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=elapsed, inventory_files=len(records), windows_scanned=len(rows),
        candidate_windows=sum(truth(r["candidate_pass"]) for r in rows), selected_windows=selected_n,
        selected_recordings=len({group_id(r) for r in rows if truth(r.get("selected"))}),
        groups=groups, exclusion_counts=dict(reasons), scan_errors=list(errors), preview=preview,
        estimated_future_mixture_count=round(selected_n / 0.3),
        estimated_future_mixture_GiB=round(selected_n / 0.3 * cfg["target_sample_rate"] * cfg["window_seconds"] * 4 / 2**30, 3),
        interpretation="Automatic acoustic screening, not a confirmed per-second vessel-presence annotation. Original files unchanged.")


def write_report(directory, summary, cfg):
    lines = ["# DeepShip初筛结果", "",
        f"模式：{summary['mode']}；状态：{summary['status']}；耗时：{summary['elapsed_seconds']/60:.2f}分钟。", "",
        f"扫描{summary['windows_scanned']}个完整窗口，自动通过{summary['candidate_windows']}个，配额选入{summary['selected_windows']}个，来自{summary['selected_recordings']}个录音组。", "",
        "原始录音未修改。SOURCE_INDEX.csv的selected=1可供后续生成器读取；candidate_pass=1但未入选表示超过录音或类别配额。", "",
        "全部入选由CONFIG.json中的自动规则和配额决定。相对电平不是实测SNR，筛选结果不作逐秒船舶存在标注。", "",
        "| 类别 | 集合 | 录音组 | 来源合格组 | 已扫描组 | 窗口 | 自动通过 | 入选 | 入选录音组 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["groups"]:
        lines.append("| " + " | ".join(str(row[k]) for k in ("class_name", "split", "recording_groups", "eligible_recording_groups", "scanned_recording_groups", "scanned_windows", "passed_windows", "selected_windows", "selected_recording_groups")) + " |")
    lines += ["", "## 每条录音的合格片段容量", "",
        "以下是配额限制前的自动合格数，不受每录音20片或每类1000片的选取上限影响。最少合格数包含零片录音；另列有合格片段录音的最小值。", "",
        "| 类别 | 集合 | 零片录音 | 最少合格 | 非零最少 | 中位数 | 最多合格 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["groups"]:
        values = [row[k] for k in ("class_name", "split", "zero_candidate_recordings", "min_candidates_per_recording", "min_positive_candidates_per_recording", "median_candidates_per_recording", "max_candidates_per_recording")]
        lines.append("| " + " | ".join("—" if value is None else str(value) for value in values) + " |")
    lines += ["", "## 自动淘汰原因", "", "同一窗口可以有多个原因。", ""]
    for reason, count in sorted(summary["exclusion_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {reason}: {count}")
    if not summary["selected_windows"] and summary["mode"] != "plan":
        lines += ["", "本轮没有入选片段。请查看来源匹配和淘汰原因；工具不会自动放宽门槛凑数。"]
    if summary["scan_errors"]:
        lines += ["", "## 读取或计算错误", ""]
        lines.extend(f"- {r['recording_id']}: {r['error']}" for r in summary["scan_errors"])
    lines += ["", "## 后续规模预算", "",
        f"以当前入选Single按30/60/10构建，约{summary['estimated_future_mixture_count']}条最终样本，波形约{summary['estimated_future_mixture_GiB']:.3f}GiB；此处尚未生成混合数据。", "",
        "参数以CONFIG.json为准；全部阈值见配置。若调整阈值，请重新运行扫描并使用新版本输出目录。原始Test识别结果不参与筛选。", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    if args.workers is not None:
        cfg["runtime"]["workers"] = max(1, args.workers)
    if args.pilot_recordings_per_class is not None:
        cfg["runtime"]["pilot_recordings_per_class"] = args.pilot_recordings_per_class
    if args.plots:
        cfg["preview"]["enabled"] = True
    directory = resolve_path(args.run_dir, cfg_path.parent) if args.run_dir else Path(cfg["output_root"]) / args.mode
    directory = safe_output(directory, cfg["raw_root"])
    if (directory / "SOURCE_INDEX.csv").exists() and not args.overwrite:
        raise ValueError("This run already has an index. Use a new output_root or --overwrite.")
    begun = time.perf_counter()
    write_json(directory / "CONFIG.json", cfg)
    write_json(directory / "SUMMARY.json", dict(status="running", mode=args.mode))
    records, rows, errors = [], [], []
    try:
        records = build_inventory(cfg)
        for record in records:
            record.update(scan_status="not_scanned", windows_scanned=0, windows_candidate_pass=0, windows_selected=0)
        wanted = [r for r in records if truth(r["eligible"])]
        if args.mode == "pilot":
            wanted = select_pilot(records, cfg, int(cfg["runtime"]["pilot_recordings_per_class"]))
        if args.mode == "plan":
            wanted = []
        print(f"Inventory: {len(records)} files; {sum(truth(r['eligible']) for r in records)} source-eligible; scanning {len(wanted)} ({args.mode}).", flush=True)
        rows, errors = [], []
        completed = 0
        def receive(record, result):
            nonlocal completed
            windows, diagnostic, error = result
            completed += 1
            record["scan_status"] = "failed" if error else "complete"
            record["scan_error"] = error or ""
            record["windows_scanned"] = len(windows)
            record["windows_candidate_pass"] = sum(truth(r["candidate_pass"]) for r in windows)
            for key in ("trailing_seconds", "reference_windows", "reference_fallback", "seconds"):
                if key in diagnostic:
                    record["scan_" + key] = diagnostic[key]
            if error:
                errors.append(dict(recording_id=record["recording_id"], raw_relative_path=record["raw_relative_path"], error=error))
                print(f"ERROR {record['recording_id']}: {error}", flush=True)
            rows.extend(windows)
            if completed % max(1, int(cfg["runtime"]["progress_every"])) == 0 or completed == len(wanted):
                elapsed = time.perf_counter()-begun
                duration = diagnostic.get('seconds', 0)
                remaining = elapsed/max(1,completed)*(len(wanted)-completed)/60
                print(f"[{completed}/{len(wanted)}] {record['raw_relative_path']} {duration:.2f}s pass={record['windows_candidate_pass']}/{len(windows)} total_windows={len(rows)} elapsed={elapsed/60:.2f}min ETA={remaining:.2f}min", flush=True)
        scan_records(wanted, cfg, receive)
        rows.sort(key=lambda r: (int(r["class_index"]), r["recording_id"], r["raw_relative_path"], int(r["start_sample"])))
        choose_windows(rows, cfg)
        selected_counts = Counter(r["raw_relative_path"] for r in rows if truth(r["selected"]))
        for record in records:
            record["windows_selected"] = selected_counts[record["raw_relative_path"]]
        write_csv(directory / "RECORDINGS.csv", records, ("recording_id", "class_name", "split", "raw_relative_path", "eligible", "exclusion_reason"))
        write_csv(directory / "SOURCE_INDEX.csv", rows, SOURCE_FIELDS)
        preview_result = None
        if cfg["preview"]["enabled"] and rows:
            from preview import create_previews
            preview_result = create_previews(cfg, directory, rows)
        summary = summarize(cfg, args.mode, records, rows, time.perf_counter() - begun, errors, preview_result)
        write_json(directory / "SUMMARY.json", summary)
        write_report(directory, summary, cfg)
        print(f"COMPLETE: selected {summary['selected_windows']}/{len(rows)} windows from {summary['selected_recordings']} recordings; errors={len(errors)}.", flush=True)
        print(f"Report: {directory / 'REPORT.md'}", flush=True)
        print(f"Index:  {directory / 'SOURCE_INDEX.csv'}", flush=True)
        return not errors
    except BaseException as error:
        # Retain completed measurements in the same two indexes, without selecting a final dataset.
        if rows:
            for row in rows:
                row.update(selected=0, selection_reason="incomplete_scan")
            write_csv(directory / "SOURCE_INDEX.csv", rows, SOURCE_FIELDS)
            write_csv(directory / "RECORDINGS.csv", records)
        write_json(directory / "SUMMARY.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", mode=args.mode, error=str(error), elapsed_seconds=time.perf_counter()-begun, completed_recordings=sum(r.get('scan_status')=='complete' for r in records), completed_windows=len(rows)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"), help="All relative paths are relative to this JSON file.")
    parser.add_argument("--mode", choices=("plan", "pilot", "full"), default="pilot")
    parser.add_argument("--run-dir", help="Override the exact output run folder.")
    parser.add_argument("--workers", type=int, help="Parallel recordings; default from config (2).")
    parser.add_argument("--pilot-recordings-per-class", type=int, help="Train recording groups per class in pilot only.")
    parser.add_argument("--plots", action="store_true", help="Write a few optional diagnostic PNGs; does not affect selection.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this tool's index/report in the chosen run folder; never deletes raw audio.")
    args = parser.parse_args()
    if args.pilot_recordings_per_class is not None and args.pilot_recordings_per_class <= 0:
        parser.error("--pilot-recordings-per-class must be positive.")
    success = run(args)
    if success is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
