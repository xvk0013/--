"""Check source ownership, file readability and audio validity; no hash manifests."""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
from common import ROOT, read, SPLITS


def run(raw_only=False):
    cfg = read(ROOT/'config.json')
    sources = read(ROOT/'metadata/sources.json')
    recordings = read(ROOT/'metadata/recordings.json')
    by_recording = {r['recording_id']:r for r in recordings}
    assert len(by_recording) == len(recordings), 'A recording has multiple assignments.'
    assert len({r['source_id'] for r in sources}) == len(sources)
    assert all(r['split'] in SPLITS for r in recordings)
    for source in sources:
        owner = by_recording[source['recording_id']]
        assert source['split'] == owner['split'] and source['class_name'] == owner['class_name']
    if raw_only:
        for r in recordings:
            info = sf.info(Path(cfg['raw_root'])/r['raw_relative_path'])
            assert info.frames > 0 and info.samplerate > 0
        print('Raw audio files readable:', len(recordings), flush=True)
        return
    paths = [(r['component_path'], True) for r in sources]
    paths += [(r['background_path'], False) for r in recordings if r['class_name'] == 'E']
    for path, core in paths:
        wave, fs = sf.read(ROOT/path, dtype='float32')
        assert fs == cfg['sample_rate'] and wave.ndim == 1 and np.isfinite(wave).all()
        assert np.any(wave != 0), f'Silent source: {path}'
        assert (len(wave) == 80000 if core else len(wave) >= 80000)
    print('Source checks passed:', len(sources), 'target cores.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-only', action='store_true')
    run(parser.parse_args().raw_only)
