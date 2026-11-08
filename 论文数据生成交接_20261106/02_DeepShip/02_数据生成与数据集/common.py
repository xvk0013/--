"""Small IO and numerical helpers; no experiment or model dependencies."""
import csv
import json
import math
from pathlib import Path
import numpy as np

SPLITS = ('Train', 'Val', 'Test')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = ['sample_id','split','array_index','kind','label_index','count','target','context_condition']
    fields += sorted({key for row in rows for key in row}-set(fields))
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def resolve_path(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(base)/path).resolve()


def load_config(path):
    path=Path(path).resolve();cfg=read_json(path)
    for key in ('raw_root','output_directory','smoke_output_directory'):
        cfg[key]=str(resolve_path(cfg[key],path.parent))
    for key in ('index_file','recordings_file'):
        cfg['sources'][key]=str(resolve_path(cfg['sources'][key],path.parent))
    cfg['bellhop']['channel_file']=str(resolve_path(cfg['bellhop']['channel_file'],path.parent))
    if len(set(cfg['classes']))!=len(cfg['classes']) or len(cfg['classes'])<2:
        raise ValueError('classes must contain at least two unique vessel types.')
    if cfg['sample_rate']<=0 or cfg['duration_seconds']<=0 or cfg['output_rms']<=0:
        raise ValueError('Sample rate, duration and output_rms must be positive.')
    if cfg['channel']<0 or cfg['selection']['max_per_recording']<1:
        raise ValueError('Invalid input channel or per-recording limit.')
    if set(cfg['selection']['per_class'])!=set(SPLITS) or any(v<1 for v in cfg['selection']['per_class'].values()):
        raise ValueError('selection.per_class needs positive Train/Val/Test counts.')
    p=cfg['mixture']
    if set(p['context_ratio'])!={'C0','C1','C2'} or any(v<0 for v in p['context_ratio'].values()) or not math.isclose(sum(p['context_ratio'].values()),1.0,abs_tol=1e-6):
        raise ValueError('context_ratio needs nonnegative C0/C1/C2 weights summing to one.')
    if not 0<=p['epsilon']<=1 or not math.isfinite(float(p['sir_db'])):
        raise ValueError('Invalid weak-source amplitude or SIR.')
    if any(len(p[k])!=2 or p[k][0]<0 or p[k][0]>p[k][1] for k in ('context_seconds','fade_seconds')):
        raise ValueError('Envelope parameter ranges must be ordered nonnegative pairs.')
    if p['context_seconds'][1]*2+p['fade_seconds'][1]*2>=cfg['duration_seconds']:
        raise ValueError('Envelope regions leave no central overlap in this duration.')
    if p['double_per_single']<0 or p['background_per_single']<0:
        raise ValueError('Mixture count ratios cannot be negative.')
    if cfg['background']['model']!='wenz_three_component':
        raise ValueError('This version implements wenz_three_component backgrounds.')
    return cfg


def rms(wave, axis=None, keepdims=False):
    return np.sqrt(np.mean(np.square(wave,dtype=np.float64),axis=axis,keepdims=keepdims))


def normalize(wave, level):
    value=rms(wave)
    if not np.isfinite(wave).all() or not np.isfinite(value) or value<=0:
        raise ValueError('Cannot normalize a silent or non-finite generated waveform.')
    gain=float(level/value)
    return np.asarray(wave*gain,dtype=np.float32),gain


def safe_output(directory, raw_root):
    directory=Path(directory).resolve();raw=Path(raw_root).resolve()
    if directory==raw or directory.is_relative_to(raw):
        raise ValueError('Generated data must be outside the original DeepShip directory.')
    directory.mkdir(parents=True,exist_ok=True)
    return directory
