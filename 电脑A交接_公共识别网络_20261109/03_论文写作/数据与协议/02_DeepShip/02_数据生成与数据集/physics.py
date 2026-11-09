"""Read existing Bellhop E99 FIRs and propagate indexed source cores.

Indices are zero-based at the target sample rate.  Continuous recording context
is retained across core boundaries; padding occurs only outside the recording.
The stored first-arrival time is metadata, not an added waveform delay.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
import operator

import h5py
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.io import loadmat
from scipy.signal import resample_poly
import soundfile as sf


def _scalar(value, name):
    values = np.asarray(value)
    if values.size != 1 or not np.isfinite(values).all() or np.iscomplexobj(values):
        raise ValueError(f"Invalid channel scalar: {name}")
    return float(values.reshape(-1)[0])


def _integer(value, name):
    number = _scalar(value, name)
    if not number.is_integer():
        raise ValueError(f"Channel {name} must be an integer")
    return int(number)


def _channel(raw, expected_fs):
    h = np.asarray(raw["h"])
    if np.iscomplexobj(h) or not h.size or not np.isfinite(h).all():
        raise ValueError("An E99 channel needs a finite, real FIR")
    h = np.ascontiguousarray(h.reshape(-1), dtype=np.float64)
    if not np.any(h):
        raise ValueError("An E99 FIR cannot be all zero")
    fs = _integer(raw["fs"], "fs")
    channel_id = _integer(raw["id"], "id")
    memory = _integer(raw["memory_samples"], "memory_samples")
    delay = _integer(raw["numerical_delay_samples"], "numerical_delay_samples")
    if fs != int(expected_fs):
        raise ValueError(f"FIR sample rate {fs} differs from requested {expected_fs}")
    if channel_id < 1 or memory != len(h)-1 or not 0 <= delay <= memory:
        raise ValueError(f"Invalid FIR memory/delay or id in channel {channel_id}")
    result = dict(h=h, fs=fs, id=channel_id, M=memory, D=delay,
                  memory_samples=memory, numerical_delay_samples=delay)
    for name in ("range_km", "source_depth_m", "receiver_depth_m", "first_arrival_s"):
        if name in raw:
            result[name] = _scalar(raw[name], name)
    return result


def load_channels(path, expected_fs=16000):
    """Return FIR dictionaries keyed by their actual channel id.

    MATLAB v7.3 cell references are read with h5py.  Id-based ordering avoids
    confusing the HDF5-transposed cell grid with MATLAB's column-major order.
    Earlier MAT versions are also accepted through scipy.io.loadmat.
    """
    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as source:
            if "channels" not in source:
                raise ValueError("The MAT bank has no 'channels' variable")
            references = source["channels"][()].reshape(-1)
            raw_channels = []
            for reference in references:
                group = source[reference]
                names = ("h", "fs", "id", "memory_samples", "numerical_delay_samples",
                         "range_km", "source_depth_m", "receiver_depth_m", "first_arrival_s")
                raw_channels.append({name: group[name][()] for name in names if name in group})
    else:
        raw = loadmat(path, simplify_cells=True)
        if "channels" not in raw:
            raise ValueError("The MAT bank has no 'channels' variable")
        cells = raw["channels"]
        if isinstance(cells, dict):
            raw_channels = [cells]
        elif isinstance(cells, np.ndarray):
            raw_channels = cells.reshape(-1).tolist()
        else:
            raw_channels = list(cells)
    result = {}
    for raw in raw_channels:
        channel = _channel(raw, expected_fs)
        if channel["id"] in result:
            raise ValueError(f"Duplicate FIR channel id: {channel['id']}")
        result[channel["id"]] = channel
    if not result:
        raise ValueError("The FIR bank is empty")
    return dict(sorted(result.items()))


def read_recording(path, target_fs=16000, channel=0):
    """Read/resample a whole recording once and remove its recording-wide DC.

    No per-core gain, denoising, analysis-band restriction or clipping is applied.
    The returned float64 mono array can serve all selected cores of this file.
    """
    target_fs = operator.index(target_fs)
    channel = operator.index(channel)
    if target_fs <= 0 or channel < 0:
        raise ValueError("Invalid target sample rate or channel")
    audio, source_fs = sf.read(str(path), dtype="float64", always_2d=True)
    if not len(audio) or channel >= audio.shape[1]:
        raise ValueError(f"Recording or requested channel is empty: {path}")
    wave = audio[:, channel]
    if not np.isfinite(wave).all():
        raise ValueError(f"Non-finite waveform samples: {path}")
    if source_fs != target_fs:
        divisor = gcd(source_fs, target_fs)
        wave = resample_poly(wave, target_fs//divisor, source_fs//divisor)
    wave = np.ascontiguousarray(wave, dtype=np.float64)
    wave -= np.mean(wave, dtype=np.float64)
    return wave


def propagate_core(wave, start_sample, n_samples, channel):
    """Apply one existing E99 FIR, using real history and numerical lookahead.

    ``gain = 1/RMS(core)`` scales the entire extracted context.  For FIR memory
    M and numerical delay D, the context is [start-(M-D), end+D), followed by
    linear convolution and ``v[M:M+N]``.  The result is float64 and receives no
    output normalization; receiver-side SIR/public gain belong to mixing.
    """
    wave = np.asarray(wave)
    start = operator.index(start_sample)
    size = operator.index(n_samples)
    memory = int(channel.get("M", channel.get("memory_samples", -1)))
    delay = int(channel.get("D", channel.get("numerical_delay_samples", -1)))
    h = np.asarray(channel["h"], dtype=np.float64).reshape(-1)
    end = start+size
    if wave.ndim != 1 or size <= 0 or start < 0 or end > len(wave):
        raise ValueError("Core must be a complete positive-length interval inside the recording")
    if memory != len(h)-1 or not 0 <= delay <= memory:
        raise ValueError("Inconsistent FIR memory/delay")
    core = wave[start:end]
    before = float(np.sqrt(np.mean(np.square(core, dtype=np.float64))))
    if not np.isfinite(before) or before <= 0:
        raise ValueError("Source core is silent or non-finite")
    gain = 1.0/before
    history = memory-delay
    left = max(0, start-history)
    right = min(len(wave), end+delay)
    pre = max(0, history-start)
    post = max(0, end+delay-len(wave))
    context = np.empty(history+size+delay, dtype=np.float64)
    context[:pre] = 0
    context[pre:pre+right-left] = wave[left:right]
    if post:
        context[-post:] = 0
    context *= gain
    if not np.isfinite(context).all():
        raise ValueError("Source history/lookahead is non-finite")
    fft_size = next_fast_len(len(context)+len(h)-1)
    convolved = irfft(rfft(context, fft_size)*rfft(h, fft_size), fft_size)
    result = np.ascontiguousarray(convolved[memory:memory+size], dtype=np.float64)
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError("Propagation produced a silent or non-finite core")
    info = dict(input_rms_before=before, input_gain=gain,
                input_rms_after=float(np.sqrt(np.mean(np.square(context[history:history+size])))),
                history_samples=history, real_history_samples=history-pre,
                history_padding_samples=pre, numerical_delay_samples=delay,
                real_lookahead_samples=delay-post, lookahead_padding_samples=post,
                convolution_fft_samples=int(fft_size))
    return result, info
