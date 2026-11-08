"""Render reusable receiving components, then dataset arrays from a fixed plan."""
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from common import SPLITS, read_json, write_json, normalize, rms
from physics import load_channels, read_recording, propagate_core
from dataset import double_envelopes


def single_rows(rows):
    return sorted((row for row in rows if row['kind']=='single'), key=lambda row:int(row['source1_component_index']))


def component_key(cfg, rows):
    return dict(version=cfg['version'],raw_root=cfg['raw_root'],sample_rate=cfg['sample_rate'],
        duration_seconds=cfg['duration_seconds'],channel=cfg['channel'],bellhop=cfg['bellhop'],
        source_ids=[row['source1_id'] for row in single_rows(rows)],
        channel_ids=[int(row['source1_channel_id']) for row in single_rows(rows)])


def render_components(cfg, directory, rows, overwrite=False):
    directory=Path(directory);target=directory/'components';target.mkdir(exist_ok=True)
    path=target/'audio.npy';state_path=target/'STATE.json'
    singles=single_rows(rows);n=int(cfg['sample_rate']*cfg['duration_seconds'])
    key=component_key(cfg,rows)
    if state_path.exists():
        state=read_json(state_path)
        if state.get('status')=='complete' and state.get('key')==key and path.exists() and not overwrite:
            print('Receiving components already complete; reusing them.',flush=True)
            return state
        if not overwrite:
            raise ValueError('Existing components are incomplete or use a different source plan; use --overwrite-components or a new output directory.')
    channels=load_channels(cfg['bellhop']['channel_file'],expected_fs=cfg['sample_rate']) if cfg['bellhop']['enabled'] else {}
    if cfg['bellhop']['enabled'] and len(channels)!=cfg['bellhop']['channel_count']:
        raise ValueError('The configured channel count differs from the FIR bank.')
    if [int(row['source1_component_index']) for row in singles]!=list(range(len(singles))):
        raise ValueError('Component indices must be contiguous and unique.')
    array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(singles),n))
    by_file=defaultdict(list)
    for row in singles:
        by_file[row['source1_raw_relative_path']].append(row)
    start=time.perf_counter();completed=0;metadata=[None]*len(singles)
    write_json(state_path,dict(status='running',key=key))
    try:
        for file_index,(relative,group) in enumerate(sorted(by_file.items()),1):
            raw_root=Path(cfg['raw_root']).resolve();source=(raw_root/relative).resolve()
            if not source.is_relative_to(raw_root):
                raise ValueError('Source file must be within raw_root.')
            wave=read_recording(source,cfg['sample_rate'],cfg['channel'])
            for row in group:
                index=int(row['source1_component_index']);first=int(row['source1_start_sample']);channel_id=int(row['source1_channel_id'])
                if cfg['bellhop']['enabled']:
                    y,info=propagate_core(wave,first,n,channels[channel_id])
                else:
                    core=wave[first:first+n]
                    if len(core)!=n:
                        raise ValueError('The source core extends beyond its original recording.')
                    level=float(rms(core))
                    if not np.isfinite(level) or level<=0:
                        raise ValueError('Invalid dry source core.')
                    y=core/level
                    info=dict(input_gain=1/level,input_rms_before=level,history_padding_samples=0,lookahead_padding_samples=0)
                level=float(cfg['bellhop']['source_core_rms']);y=y*level
                if len(y)!=n or not np.isfinite(y).all() or rms(y)<=0:
                    raise ValueError('Invalid propagated core: '+row['source1_id'])
                array[index]=np.asarray(y,dtype=np.float32)
                metadata[index]=dict(source_id=row['source1_id'],component_index=index,channel_id=channel_id,
                    bellhop_applied=bool(cfg['bellhop']['enabled']),receiver_rms=float(rms(array[index])),
                    input_gain=float(info['input_gain']*level),
                    history_padding_samples=int(info.get('history_padding_samples',0)),
                    lookahead_padding_samples=int(info.get('lookahead_padding_samples',0)))
                completed+=1
            if file_index%max(1,int(cfg['runtime']['progress_every_recordings']))==0 or file_index==len(by_file):
                array.flush()
                print(f"[components {file_index}/{len(by_file)} recordings] {completed}/{len(singles)} cores; {(time.perf_counter()-start)/60:.2f}min; {relative}",flush=True)
            del wave
        array.flush();del array
        array=None
        state=dict(status='complete',key=key,count=completed,seconds=time.perf_counter()-start,components=metadata)
        write_json(state_path,state);return state
    except BaseException as error:
        if array is not None:
            array.flush()
        write_json(state_path,dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',key=key,completed=completed,error=str(error)))
        raise


def prepare_background_parameters(cfg,rows):
    for row in rows:
        if row['kind']!='background':
            continue
        split_index=SPLITS.index(row['split'])
        seed=int(row.get('noise_seed') or np.random.SeedSequence([cfg['seed'],split_index,int(row['array_index']),751]).generate_state(1)[0])
        random=np.random.default_rng(seed)
        row['noise_seed']=seed
        row['noise_wind_speed_m_s']=float(random.uniform(*cfg['background']['wind_speed_m_s']))
        row['noise_shipping_factor']=float(random.uniform(*cfg['background']['shipping_factor']))
        row['background_model']=cfg['background']['model']


def wenz_background(cfg,row):
    n=int(cfg['sample_rate']*cfg['duration_seconds']);fs=cfg['sample_rate']
    random=np.random.default_rng(np.random.SeedSequence([int(row['noise_seed']),379]))
    f=np.fft.rfftfreq(n,1/fs)[1:];fk=f/1000
    p=cfg['background'];wind=float(row['noise_wind_speed_m_s']);shipping=float(row['noise_shipping_factor'])
    turbulence=10**((17-30*np.log10(np.maximum(fk,p['turbulence_min_hz']/1000)))/10)
    turbulence[f<p['turbulence_min_hz']]=0
    ship=10**((40+20*(shipping-.5)+26*np.log10(fk)-60*np.log10(fk+.03))/10)
    waves=10**((50+7.5*np.sqrt(wind)+20*np.log10(fk)-40*np.log10(fk+.4))/10)
    amplitude=np.sqrt(turbulence+ship+waves)
    positive=amplitude*(random.standard_normal(len(f))+1j*random.standard_normal(len(f)))/np.sqrt(2)
    if n%2==0:
        positive[-1]=amplitude[-1]*random.standard_normal()
    spectrum=np.r_[0j,positive]
    y=np.fft.irfft(spectrum,n=n);return y-y.mean()


def render_mixtures(cfg,directory,rows,component_state):
    directory=Path(directory);n=int(cfg['sample_rate']*cfg['duration_seconds'])
    state_path=directory/'components'/'STATE.json'
    if component_state.get('status')!='complete' or component_state.get('key')!=component_key(cfg,rows):
        raise ValueError('Receiving components do not match this source plan.')
    components=np.load(directory/'components'/'audio.npy',mmap_mode='r')
    if components.shape!=(len(single_rows(rows)),n) or components.dtype!=np.float32:
        raise ValueError('Receiving component array has the wrong shape or type.')
    metadata=component_state['components'];result={};batch_size=max(1,int(cfg['runtime']['mix_batch_size']))
    begun=time.perf_counter()
    for split in SPLITS:
        group=sorted((row for row in rows if row['split']==split),key=lambda row:int(row['array_index']))
        if [int(row['array_index']) for row in group]!=list(range(len(group))):
            raise ValueError('Dataset row indices must be contiguous within a split.')
        target=directory/'data'/split;target.mkdir(parents=True,exist_ok=True)
        output=np.lib.format.open_memmap(target/'audio.npy',mode='w+',dtype=np.float32,shape=(len(group),n))
        minimum_rms=float('inf');maximum_rms=0.;maximum_peak=0.;last_progress=0
        for start in range(0,len(group),batch_size):
            for row in group[start:start+batch_size]:
                kind=row['kind'];array_index=int(row['array_index'])
                if kind=='background':
                    mixed=wenz_background(cfg,row)
                else:
                    first=int(row['source1_component_index']);a=np.asarray(components[first],dtype=np.float64)
                    row['source1_receiver_rms']=metadata[first]['receiver_rms']
                    row['source1_input_gain']=metadata[first]['input_gain']
                    row['bellhop_applied']=bool(cfg['bellhop']['enabled'])
                    if kind=='single':
                        mixed=a
                    elif kind=='double':
                        second=int(row['source2_component_index']);b=np.asarray(components[second],dtype=np.float64)
                        g1=1/float(rms(a));g2=10**(-float(row['sir_db'])/20)/float(rms(b))
                        ea,eb=double_envelopes(row,cfg)
                        mixed=a*g1*ea+b*g2*eb
                        row['source1_sir_gain']=g1;row['source2_sir_gain']=g2
                        row['source2_receiver_rms']=metadata[second]['receiver_rms']
                        row['source2_input_gain']=metadata[second]['input_gain']
                    else:
                        raise ValueError('Unknown sample kind: '+str(kind))
                output[array_index],gain=normalize(mixed,cfg['output_rms']);row['output_gain']=gain
                value=float(rms(output[array_index]));peak=float(np.max(np.abs(output[array_index])))
                minimum_rms=min(minimum_rms,value);maximum_rms=max(maximum_rms,value);maximum_peak=max(maximum_peak,peak)
                if cfg['export_wavs']:
                    wav_directory=directory/'wav'/split;wav_directory.mkdir(parents=True,exist_ok=True)
                    sf.write(wav_directory/f'{array_index:07d}.wav',output[array_index],cfg['sample_rate'],subtype='FLOAT')
            end=min(start+batch_size,len(group))
            if end-last_progress>=max(1,int(cfg['runtime']['progress_every_mixtures'])) or end==len(group):
                output.flush();last_progress=end
                print(f"[mixtures {split} {end}/{len(group)}] elapsed={(time.perf_counter()-begun)/60:.2f}min",flush=True)
        output.flush();del output
        result[split]=dict(n=len(group),kind_counts=dict(Counter(row['kind'] for row in group)),
            label_counts=dict(Counter(str(row['label_index']) for row in group)),
            double_context_counts=dict(Counter(row['context_condition'] for row in group if row['kind']=='double')),
            rms_min=minimum_rms,rms_max=maximum_rms,maximum_absolute_sample=maximum_peak)
    return result
