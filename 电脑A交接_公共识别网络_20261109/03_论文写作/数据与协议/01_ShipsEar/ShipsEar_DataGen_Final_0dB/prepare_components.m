function prepare_components(raw_root)
% Rebuild local components from raw audio and the stored Bellhop FIR bank.
root=fileparts(mfilename('fullpath')); addpath(fullfile(root,'physical','matlab'));
cfg=jsondecode(fileread(fullfile(root,'config.json')));
if nargin<1, raw_root=cfg.raw_root; end
rcfg=struct('target_fs',cfg.sample_rate,'selected_channel',1);
sources=jsondecode(fileread(fullfile(root,'metadata','sources.json')));
records=jsondecode(fileread(fullfile(root,'metadata','recordings.json')));
p=load(fullfile(root,cfg.bellhop.bank),'channels'); channels=p.channels;
completed=0;
for ri=1:numel(records)
    rec=records(ri);
    wave=v8_read_recording(fullfile(raw_root,rec.raw_relative_path),rcfg);
    if strcmp(rec.class_name,'E')
        v8_write_wav(fullfile(root,rec.background_path),wave,cfg.sample_rate);
    else
        indices=find(strcmp({sources.recording_id},rec.recording_id));
        for k=indices(:)'
            source=sources(k); channel=channels{source.channel_id};
            y=v9_propagate_99(wave,source.start_sample+1,channel,cfg.sample_rate*cfg.duration_seconds);
            v8_write_wav(fullfile(root,source.component_path),y,cfg.sample_rate);
            completed=completed+1;
        end
    end
    fprintf('Prepared %d/%d recordings; %d target cores.\n',ri,numel(records),completed);
end
assert(completed==numel(sources),'Some source cores were not processed.');
end
