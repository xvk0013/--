"""Essential data checks; observed proportions are reported, never forced."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
from common import ROOT, SPLITS, read, rows, config


def run(dest, check_wavs=False):
    dest = Path(dest)
    assert not (dest/'.generating').exists(), 'Generation is incomplete.'
    cfg = read(dest/'CONFIG.json')
    index = rows(dest/'DATA_INDEX.csv')
    assert index and len({r['sample_id'] for r in index}) == len(index)
    assert all(r['split'] in SPLITS for r in index)
    recordings = read(ROOT/'metadata/recordings.json')
    owners = {r['recording_id']:r for r in recordings}
    assert len(owners) == len(recordings), 'Recording ownership must be unique.'
    sources = {r['source_id']:r for r in read(ROOT/'metadata/sources.json')}
    labels = ['E', *cfg['classes'], *['+'.join(p) for p in combinations(cfg['classes'], 2)]]
    summary = {}
    for split in SPLITS:
        part = [r for r in index if r['split'] == split]
        assert part and [int(r['array_index']) for r in part] == list(range(len(part)))
        audio = np.load(dest/'data'/split/'audio.npy', mmap_mode='r')
        assert audio.shape == (len(part), cfg['sample_rate']*cfg['duration_seconds'])
        assert audio.dtype == np.float32
        singles, doubles, pairs = set(), set(), set()
        for i, r in enumerate(part):
            expected = []
            for k in (1, 2):
                source_id = r[f'source{k}_id']
                if not source_id:
                    continue
                rec = owners[r[f'source{k}_recording_id']]
                assert rec['split'] == split, 'Cross-split source.'
                assert rec['class_name'] == r[f'source{k}_class_name']
                if rec['class_name'] != 'E':
                    source = sources[source_id]
                    assert source['recording_id'] == rec['recording_id'] and source['split'] == split
                    assert r[f'source{k}_component_path'] == source['component_path']
                    for field in ('start_sample','stop_sample','channel_id'):
                        assert int(r[f'source{k}_{field}']) == int(source[field])
                    expected.append(rec['class_name'])
                    (singles if r['kind'] == 'single' else doubles).add(source_id)
                else:
                    assert r['kind'] == 'noise' and k == 1
                    assert r[f'source{k}_component_path'] == rec['background_path']
                    start, stop = int(r['source1_start_sample']), int(r['source1_stop_sample'])
                    assert start >= 0 and stop-start == cfg['sample_rate']*cfg['duration_seconds']
            assert r['label_name'] == ('+'.join(expected) or 'E')
            assert labels[int(r['label_index'])] == r['label_name']
            assert int(r['count']) == len(expected)
            for c in cfg['classes']:
                assert int(r['target_'+c]) == int(c in expected)
            assert r['kind'] == ('noise' if not expected else 'single' if len(expected) == 1 else 'double')
            if r['kind'] == 'double':
                assert len(set(expected)) == 2 and r['context_condition'] in ('C0','C1','C2')
                assert int(r['first']) in (0,1) and int(r['side']) in (0,1)
                timings = [float(r[k]) for k in ('left','right','fade_left','fade_right')]
                assert all(np.isfinite(v) and v > 0 for v in timings) and sum(timings) < cfg['duration_seconds']
                pair = tuple(sorted((r['source1_id'], r['source2_id'])))
                assert pair not in pairs, 'Repeated source pair.'
                pairs.add(pair)
            wave = audio[i]
            assert np.isfinite(wave).all() and np.any(wave != 0), f'Invalid audio: {r["sample_id"]}'
            if check_wavs:
                wav, fs = sf.read(dest/r['wav_path'], dtype='float32')
                assert fs == cfg['sample_rate'] and np.array_equal(wav, wave)
        expected_sources = {k for k,v in sources.items() if v['split'] == split}
        assert singles == doubles == expected_sources, 'An eligible core is missing from Single or Double.'
        used_background = {r['source1_recording_id'] for r in part if r['kind'] == 'noise'}
        assert used_background == {k for k,v in owners.items() if v['split'] == split and v['class_name'] == 'E'}
        summary[split] = dict(n=len(part), types=dict(Counter(r['kind'] for r in part)),
                              contexts=dict(Counter(r['context_condition'] for r in part if r['kind'] == 'double')))
    print('CHECKS PASSED', summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--wavs', action='store_true', help='Also check every WAV against its array row.')
    args = parser.parse_args()
    run(args.root or config()[1], args.wavs)
