"""Read-only, recording-local screening of complete acoustic source windows.

Energy and spectral tests are configurable selection proxies.  They do not
estimate ground-truth SNR or establish that a vessel is physically present.
All indices in the returned source rows refer to the target-rate waveform;
the original audio file is never changed or normalized by this module.
"""
from __future__ import annotations

from collections import Counter
from math import ceil, gcd
from pathlib import Path
import time

import numpy as np
from scipy import ndimage, signal
import soundfile as sf


def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run, including runs at the two boundaries."""
    if not mask.size or not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return int(np.max(edges[1::2] - edges[::2]))


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64))) if x.size else 0.0


def _db(power: float | np.ndarray):
    result = 10.0 * np.log10(np.maximum(power, 1e-30))
    return float(result) if np.ndim(result) == 0 else result


def _pcm_saturation_fraction(raw: np.ndarray, subtype: str) -> float:
    """Only known integer PCM has a meaningful normalized full-scale limit."""
    if not subtype.startswith("PCM_"):
        return 0.0
    try:
        bits = int(subtype.split("_")[1])
    except (ValueError, IndexError):
        return 0.0
    if bits < 2 or not raw.size:
        return 0.0
    top = np.float32(1.0 - 2.0 ** (1 - bits))
    return float(np.mean((raw <= -1.0) | (raw >= top)))


def _shape_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def spectral_metrics(x: np.ndarray, fs: int, band, settings: dict) -> dict:
    """Return tonal OR stable-broadband support on a five-second candidate.

    Tonal peaks are judged relative to a local frequency-median floor. A
    configurable +/- frequency-bin tolerance allows modest within-clip drift.
    Consecutive supported bins count as one line, not several independent ones.
    Broadband evidence uses the shape of one-second average spectra, so changes
    in absolute gain cannot alone create spectral consistency.
    """
    n_fft = int(settings["n_fft"])
    hop = int(settings["hop"])
    if n_fft > x.size or not 0 < hop <= n_fft:
        raise ValueError("Spectrum n_fft/hop must fit the complete source window")
    frequencies, centers, coefficients = signal.stft(
        x, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft-hop,
        detrend=False, boundary=None, padded=False,
    )
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    psd = np.abs(coefficients[mask]).astype(np.float64) ** 2
    if psd.shape[0] < 3 or psd.shape[1] == 0:
        raise ValueError("Analysis band must contain at least three FFT bins")
    log_psd = _db(psd)
    width = int(settings["local_median_bins"])
    if width < 3 or width % 2 != 1:
        raise ValueError("local_median_bins must be an odd integer >= 3")
    floor = ndimage.median_filter(log_psd, size=(width, 1), mode="nearest")
    peaks = np.zeros_like(log_psd, dtype=bool)
    peaks[1:-1] = (
        (log_psd[1:-1] >= log_psd[:-2])
        & (log_psd[1:-1] > log_psd[2:])
        & (log_psd[1:-1] - floor[1:-1] >= float(settings["tonal_excess_db"]))
    )
    tolerance = int(settings.get("frequency_tolerance_bins", 1))
    if tolerance:
        peaks = ndimage.maximum_filter1d(
            peaks.astype(np.uint8), size=2*tolerance+1, axis=0, mode="constant"
        ).astype(bool)
    persistent = peaks.mean(axis=1) >= float(settings["tonal_persistence"])
    supported = np.flatnonzero(persistent)
    lines = int(1 + np.count_nonzero(np.diff(supported) > 1)) if supported.size else 0

    # Exact non-overlapping one-second intervals within the source core.
    duration = x.size / fs
    spectra = []
    for index in range(int(np.floor(duration))):
        active = (centers >= index) & (centers < index + 1)
        if active.any():
            spectra.append(np.log(np.maximum(psd[:, active].mean(axis=1), 1e-30)))
    correlations = [_shape_correlation(a, b) for a, b in zip(spectra, spectra[1:])]
    stability = float(np.median(correlations)) if correlations else 0.0
    average = np.maximum(psd.mean(axis=1), 1e-30)
    flatness = float(np.clip(np.exp(np.log(average).mean()) / average.mean(), 0.0, 1.0))
    tonal_ok = lines >= int(settings["min_persistent_lines"])
    broadband_ok = (
        stability >= float(settings["min_broadband_stability"])
        and flatness <= float(settings["max_broadband_flatness"])
    )
    return dict(
        tonal_lines=lines,
        broadband_stability=stability,
        spectral_flatness=flatness,
        spectral_pass=int(tonal_ok or broadband_ok),
        tonal_support=int(tonal_ok),
        broadband_support=int(broadband_ok),
    )


def _transient_fraction(x: np.ndarray, fs: int, milliseconds: float) -> float:
    energy = np.square(x, dtype=np.float64)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    length = min(x.size, max(1, int(round(milliseconds * fs / 1000.0))))
    cumulative = np.r_[0.0, np.cumsum(energy)]
    strongest = float(np.max(cumulative[length:] - cumulative[:-length]))
    return float(np.clip(strongest / total, 0.0, 1.0))


def _not_eligible(value) -> bool:
    return value is False or value in (0, "0", "false", "False")


def scan_recording(record: dict, cfg: dict) -> tuple[list[dict], dict]:
    """Analyze one recording, returning every complete core plus diagnostics.

    This function holds at most one decoded recording and its analysis waveform
    in memory. It does not write audio, load a recognizer, or perform Bellhop.
    Automatic recording and class quotas are applied by the caller.
    """
    begun = time.perf_counter()
    diagnostic = {
        "recording_id": record["recording_id"],
        "raw_relative_path": record["raw_relative_path"],
        "split": record.get("split", ""),
        "windows": 0,
        "technical_windows": 0,
        "candidate_windows": 0,
        "reason_counts": {},
    }
    if _not_eligible(record.get("eligible", 1)):
        diagnostic.update(
            skipped=True, skip_reason=record.get("exclusion_reason", "record_ineligible"),
            seconds=0.0,
        )
        return [], diagnostic

    settings = cfg["screening"]
    target_fs = int(cfg["target_sample_rate"])
    window_samples = int(round(float(cfg["window_seconds"]) * target_fs))
    sub_samples = int(round(float(settings["persistence_subwindow_seconds"]) * target_fs))
    if target_fs <= 0 or window_samples <= 0 or sub_samples <= 0 or window_samples % sub_samples:
        raise ValueError("Window length must contain a whole number of positive persistence subwindows")
    if not 0 < float(settings["energy_keep_fraction"]) <= 1:
        raise ValueError("energy_keep_fraction must be in (0,1]")
    if not 0 <= int(settings["persistence_required"]) <= window_samples // sub_samples:
        raise ValueError("persistence_required exceeds the number of subwindows")
    band = tuple(float(v) for v in settings["analysis_band_hz"])
    if not 0 < band[0] < band[1] < target_fs / 2:
        raise ValueError("analysis_band_hz must be strictly inside target Nyquist")

    path = Path(record["raw_path"])
    info = sf.info(str(path))
    raw, raw_fs = sf.read(str(path), dtype="float32", always_2d=True)
    channel = int(cfg.get("channel", 0))
    if not 0 <= channel < raw.shape[1]:
        raise ValueError(f"Requested channel {channel}, but {path.name} has {raw.shape[1]} channels")
    raw = raw[:, channel]
    if int(record.get("sample_rate", raw_fs)) != raw_fs:
        raise ValueError(f"Audio sample rate changed since inventory: {path}")
    finite = np.isfinite(raw)
    # Replacement is analysis-only: the original non-finite mask still rejects
    # its affected windows. It prevents one bad sample poisoning the whole filter.
    clean = np.where(finite, raw, np.float32(0.0))
    factor = gcd(raw_fs, target_fs)
    wave = signal.resample_poly(clean, target_fs//factor, raw_fs//factor) if raw_fs != target_fs else clean.copy()
    wave -= np.mean(wave, dtype=np.float64)
    n_windows = len(wave) // window_samples
    diagnostic.update(
        raw_sample_rate=raw_fs, target_sample_rate=target_fs,
        target_samples=len(wave), windows=n_windows,
        trailing_seconds=float((len(wave) % window_samples) / target_fs),
    )
    if not n_windows:
        diagnostic.update(skipped=True, skip_reason="shorter_than_window", seconds=time.perf_counter()-begun)
        return [], diagnostic
    sos = signal.butter(int(settings["filter_order"]), band, btype="bandpass", fs=target_fs, output="sos")
    analysis = signal.sosfiltfilt(sos, wave).astype(np.float32)
    rows = []
    technical_ok = []
    reasons_by_window = []
    subwindow_powers = []
    for index in range(n_windows):
        start, end = index*window_samples, (index+1)*window_samples
        # Mapping via integer sample indices also works for noninteger resampling ratios.
        first_raw = start*raw_fs // target_fs
        last_raw = min(len(raw), int(ceil(end*raw_fs / target_fs)))
        source = raw[first_raw:last_raw]
        clip = analysis[start:end]
        energy = _power(clip)
        sub_energy = np.mean(np.square(clip.reshape(-1, sub_samples), dtype=np.float64), axis=1)
        max_zero_ms = 1000.0*_longest_run(source == 0.0)/raw_fs
        saturation = _pcm_saturation_fraction(source, info.subtype)
        technical = []
        if not finite[first_raw:last_raw].all():
            technical.append("nonfinite")
        if not source.size or np.all(source == 0):
            technical.append("all_zero")
        if np.sqrt(energy) < float(settings["min_rms"]):
            technical.append("no_band_energy")
        if max_zero_ms >= float(settings["zero_run_ms"]):
            technical.append("digital_dropout")
        if saturation > float(settings["max_pcm_saturation_fraction"]):
            technical.append("pcm_saturation")
        transient = _transient_fraction(clip, target_fs, float(settings["transient_window_ms"]))
        row = dict(
            source_id=f"{Path(record['raw_relative_path']).with_suffix('').as_posix()}@{start}",
            recording_id=record["recording_id"], class_name=record["class_name"],
            split_group_id=record.get("split_group_id", record["recording_id"]),
            class_index=int(record["class_index"]), split=record["split"],
            raw_relative_path=record["raw_relative_path"],
            start_sample=start, end_sample=end, num_samples=window_samples,
            target_sample_rate=target_fs, raw_sample_rate=raw_fs,
            start_seconds=float(start/target_fs), end_seconds=float(end/target_fs),
            energy_db=_db(energy), energy_rank_fraction=None,
            analysis_rms=float(np.sqrt(energy)),
            persistent_subwindows=0,
            activity_reference_db=None,
            minimum_subwindow_delta_db=None,
            max_zero_run_ms=float(max_zero_ms),
            raw_pcm_saturation_fraction=saturation,
            transient_energy_fraction=transient,
            tonal_lines=None, broadband_stability=None, spectral_flatness=None,
            spectral_pass=None, tonal_support=None, broadband_support=None,
            technical_pass=int(not technical), candidate_pass=0,
            exclusion_reason="", quality_score=0.0,
        )
        rows.append(row)
        technical_ok.append(not technical)
        reasons_by_window.append(technical)
        subwindow_powers.append(sub_energy)

    # A corrupt or isolated-shock window must neither inflate the activity
    # baseline nor consume the strongest-window budget ahead of valid sources.
    eligible = [i for i, ok in enumerate(technical_ok) if ok
                and rows[i]["transient_energy_fraction"] <= float(settings["max_transient_energy_fraction"])]
    reference_indices = eligible or [i for i, ok in enumerate(technical_ok) if ok]
    reference_powers = []
    for index in reference_indices:
        core = analysis[index*window_samples:(index+1)*window_samples]
        n_seconds = len(core)//target_fs
        if n_seconds:
            values = np.mean(np.square(core[:n_seconds*target_fs].reshape(n_seconds, target_fs), dtype=np.float64), axis=1)
            reference_powers.extend(values.tolist())
    if not reference_powers:
        reference_powers = [0.0]
    reference_db = float(np.percentile(_db(np.asarray(reference_powers)), float(settings["activity_reference_percentile"])))
    activity_floor = reference_db - float(settings["activity_margin_db"])
    for row, powers in zip(rows, subwindow_powers):
        levels = _db(powers)
        row["persistent_subwindows"] = int(np.count_nonzero(levels >= activity_floor))
        row["activity_reference_db"] = reference_db
        row["minimum_subwindow_delta_db"] = float(np.min(levels)-reference_db)
    # Ceiling retains at least one candidate in a short but otherwise valid recording.
    ordered = sorted(eligible, key=lambda i: (-rows[i]["energy_db"], rows[i]["start_sample"]))
    kept_rank_count = int(ceil(len(ordered)*float(settings["energy_keep_fraction"])))
    strong = set(ordered[:kept_rank_count])
    for rank, index in enumerate(ordered):
        rows[index]["energy_rank_fraction"] = float(rank/max(1, len(ordered)-1))

    spectrum_settings = settings["spectrum"]
    enabled = bool(spectrum_settings.get("enabled", True))
    compute_all = bool(cfg.get("runtime", {}).get("compute_spectrum_for_all", False))
    for index, row in enumerate(rows):
        reasons = reasons_by_window[index]
        if technical_ok[index] and index not in strong:
            reasons.append("outside_strong_energy_fraction")
        if row["persistent_subwindows"] < int(settings["persistence_required"]):
            reasons.append("insufficient_persistence")
        if row["transient_energy_fraction"] > float(settings["max_transient_energy_fraction"]):
            reasons.append("isolated_transient")
        if enabled and (compute_all or not reasons):
            row.update(spectral_metrics(
                analysis[row["start_sample"]:row["end_sample"]], target_fs, band, spectrum_settings
            ))
            if not row["spectral_pass"]:
                reasons.append("insufficient_spectral_support")
        elif not enabled:
            row["spectral_pass"] = 1
        # Not computed is retained as None, rather than mislabeled as a spectral failure.
        row["candidate_pass"] = int(not reasons)
        row["exclusion_reason"] = "|".join(dict.fromkeys(reasons))
        rank_strength = 1.0 - (row["energy_rank_fraction"] if row["energy_rank_fraction"] is not None else 1.0)
        persistence_strength = row["persistent_subwindows"]/(window_samples/sub_samples)
        tonal_strength = min(1.0, float(row["tonal_lines"] or 0)/max(1, int(spectrum_settings["min_persistent_lines"])))
        broadband_strength = max(0.0, float(row["broadband_stability"] or 0.0))
        evidence = max(tonal_strength, broadband_strength) if enabled else 0.5
        transient_strength = max(0.0, 1.0-row["transient_energy_fraction"])
        row["quality_score"] = float(30*rank_strength + 25*persistence_strength + 25*evidence + 20*transient_strength)

    reason_counts = Counter(code for row in rows for code in row["exclusion_reason"].split("|") if code)
    diagnostic.update(
        skipped=False,
        technical_windows=int(sum(technical_ok)),
        reference_windows=len(eligible),
        reference_fallback=not bool(eligible),
        candidate_windows=int(sum(row["candidate_pass"] for row in rows)),
        reason_counts=dict(sorted(reason_counts.items())),
        seconds=float(time.perf_counter()-begun),
    )
    return rows, diagnostic
