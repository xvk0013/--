"""Optional diagnostic figures; selection remains entirely automatic."""
from pathlib import Path
import math
import numpy as np
from scipy.signal import spectrogram
from dataset import load_core
from selection import truth, quality


def create_previews(cfg, run_directory, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(run_directory)/'preview'
    directory.mkdir(exist_ok=True)
    errors, written = [], 0
    options = cfg['preview']
    for name in cfg['classes']:
        candidates = [r for r in rows if r['class_name']==name]
        selected = sorted((r for r in candidates if truth(r['selected'])),key=lambda r:(-quality(r),r['source_id']))
        limit = max(0,int(options['max_per_class']))
        picked = selected[:limit]
        if options.get('include_rejected', True) and limit:
            rejected = sorted((r for r in candidates if not truth(r['selected']) and r.get('technical_pass',1)),key=lambda r:(-quality(r),r['source_id']))
            if rejected:
                picked = picked[:max(0,limit-1)]+rejected[:1]
        for index,row in enumerate(picked):
            fig = None
            try:
                wave,fs = load_core(row,cfg['raw_root'],int(cfg['channel']))
                if not np.isfinite(wave).all():
                    raise ValueError('Non-finite preview waveform')
                f,t,p = spectrogram(wave,fs=fs,nperseg=2048,noverlap=1536,detrend='constant')
                lo,hi = cfg['screening']['analysis_band_hz']
                keep = (f>=lo)&(f<=hi)
                db = 10*np.log10(np.maximum(p[keep],1e-20))
                fig,axes = plt.subplots(2,1,figsize=(10,5),constrained_layout=True)
                step = max(1,math.ceil(len(wave)/20000))
                axes[0].plot(np.arange(0,len(wave),step)/fs,wave[::step],lw=.5)
                axes[0].set(xlabel='Time (s)',ylabel='Original amplitude')
                axes[1].pcolormesh(t,f[keep],db,shading='auto',cmap='magma',vmin=db.max()-70,vmax=db.max())
                axes[1].set(xlabel='Time (s)',ylabel='Frequency (Hz)')
                flag = 'SELECTED' if truth(row['selected']) else 'NOT SELECTED'
                fig.suptitle(f"{name} | {row['recording_id']} | {row['start_seconds']}s | {flag}\n{row['exclusion_reason'] or row['selection_reason']}",fontsize=9)
                fig.savefig(directory/f'{name}_{index+1:02d}.png',dpi=110)
                written += 1
            except Exception as error:
                errors.append(dict(source_id=row['source_id'],error=str(error)))
            finally:
                if fig is not None:
                    plt.close(fig)
    return dict(written=written,errors=errors)
