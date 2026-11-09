function [wave,info]=v8_read_recording(file,cfg)
% No classification/QC energy thresholds. Corrupt intervals are never filled.
[raw,fs]=audioread(file);
assert(size(raw,2)>=cfg.selected_channel,'V8:Channel','Selected channel absent: %s',file);
wave=double(raw(:,cfg.selected_channel));
bad=~isfinite(wave);
if any(bad)
    edges=diff([false;bad;false]); intervals=[find(edges==1) find(edges==-1)-1];
    error('V8:Corrupt','Nonfinite source intervals at original fs=%g in %s: %s', ...
        fs,file,mat2str(intervals));
end
info=struct('original_fs',fs,'original_samples',numel(wave),'original_channels',size(raw,2), ...
    'selected_channel',cfg.selected_channel,'resampler_support_samples',0);
constant=~isempty(wave) && all(wave==wave(1));
if fs~=cfg.target_fs
    g=gcd(fs,cfg.target_fs); p=cfg.target_fs/g; q=fs/g;
    [wave,b]=resample(wave,p,q);
    info.resampler_support_samples=ceil((numel(b)-1)/(2*q));
end
assert(all(isfinite(wave)),'V8:Corrupt','Nonfinite resampled recording: %s',file);
if constant, wave(:)=0; elseif ~isempty(wave), wave=wave-mean(wave); end
info.samples=numel(wave);
end
