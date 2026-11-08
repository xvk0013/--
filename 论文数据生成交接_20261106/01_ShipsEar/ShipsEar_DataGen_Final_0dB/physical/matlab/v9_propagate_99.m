function [y,info,x]=v9_propagate_99(wave,first,channel,N)
% Applies to Full and E99 FIRs: exact context is [first-(M-D), last+D].
M=channel.memory_samples; D=channel.numerical_delay_samples; L=M-D;
last=first+N-1; wave=wave(:);
assert(M==numel(channel.h)-1 && L>=0 && D>=0 && first>=1 && last<=numel(wave));
core=wave(first:last); before=sqrt(mean(core.^2));
assert(isfinite(before) && before>0,'E99:Core','Invalid/zero core RMS.'); gain=1/before;
left=max(1,first-L); right=min(numel(wave),last+D);
pre=max(0,L-(first-1)); post=max(0,last+D-numel(wave));
x=[zeros(pre,1);wave(left:right);zeros(post,1)]*gain;
assert(numel(x)==L+N+D && all(isfinite(x)));
nfft=2^nextpow2(numel(x)+numel(channel.h)-1);
v=real(ifft(fft(x,nfft,1).*fft(channel.h(:),nfft,1)));
y=v(M+(1:N));
assert(all(isfinite(y)) && any(y~=0),'E99:Propagation','Invalid propagation output.');
info=struct('input_rms_before',before,'input_gain',gain, ...
    'input_rms_after',sqrt(mean(x(L+(1:N)).^2)), ...
    'history_samples',L,'real_history_samples',L-pre,'history_padding_samples',pre, ...
    'numerical_delay_samples',D,'real_lookahead_samples',D-post,'lookahead_padding_samples',post, ...
    'convolution_fft_samples',nfft);
end
