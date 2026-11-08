"""Small downstream interface to the selected raw core intervals."""
import csv
import json
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def iter_selected(run_directory, split=None, require_complete=True):
    """Yield selected SOURCE_INDEX rows; this does not normalize or propagate audio."""
    root = Path(run_directory)
    if require_complete:
        summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8-sig"))
        if summary["status"] != "complete":
            raise ValueError("The prescreen run has not completed.")
    with (root / "SOURCE_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["selected"] == "1" and (split is None or row["split"] == split):
                yield row


def load_core(row, raw_root, channel=0):
    """Read the 5-second core for inspection; original files remain read-only.

    A Bellhop renderer must read real history/lookahead around this interval,
    not convolve this isolated core with zero padding at every core boundary.
    """
    root = Path(raw_root).resolve()
    path = (root / row["raw_relative_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source interval must be inside raw_root.")
    with sf.SoundFile(path, mode="r") as source:
        rate = source.samplerate
        first = round(float(row["start_seconds"]) * rate)
        size = round((float(row["end_seconds"]) - float(row["start_seconds"])) * rate)
        source.seek(first)
        audio = source.read(size, dtype="float32", always_2d=True)
    if len(audio) != size or channel >= audio.shape[1]:
        raise ValueError("Source interval or channel is unavailable.")
    audio = audio[:, channel]
    target = int(row["target_sample_rate"])
    if rate != target:
        common = gcd(rate, target)
        audio = resample_poly(audio, target // common, rate // common)
    count = int(row["num_samples"])
    if len(audio) < count:
        raise ValueError("Resampled core is shorter than its declared length.")
    return np.asarray(audio[:count], dtype=np.float32), target
