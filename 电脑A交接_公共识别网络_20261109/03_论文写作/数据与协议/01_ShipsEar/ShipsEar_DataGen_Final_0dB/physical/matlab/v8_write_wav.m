function saved=v8_write_wav(file,x,fs)
% IEEE float32 (format tag 3), no clipping, with exact single-precision readback.
assert(isreal(x) && iscolumn(x) && all(isfinite(x)));
saved=single(x); assert(all(isfinite(saved)),'V8:FloatOverflow','float32 overflow: %s',file);
folder=fileparts(file); if ~isfolder(folder), mkdir(folder); end
fid=fopen(file,'wb','ieee-le'); assert(fid>=0,'Cannot write %s',file);
cleanup=onCleanup(@() fclose(fid)); n=numel(saved); bytes=4*n;
fwrite(fid,'RIFF','char'); fwrite(fid,50+bytes,'uint32'); fwrite(fid,'WAVE','char');
fwrite(fid,'fmt ','char'); fwrite(fid,18,'uint32'); fwrite(fid,[3 1],'uint16');
fwrite(fid,[fs 4*fs],'uint32'); fwrite(fid,[4 32 0],'uint16');
fwrite(fid,'fact','char'); fwrite(fid,4,'uint32'); fwrite(fid,n,'uint32');
fwrite(fid,'data','char'); fwrite(fid,bytes,'uint32'); count=fwrite(fid,saved,'single');
assert(count==n,'Incomplete WAV write: %s',file); clear cleanup;
[back,rate]=audioread(file);
assert(rate==fs && isequal(double(saved),back),'V8:Readback','WAV changed values: %s',file);
end
