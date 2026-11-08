function channel=v9_trim_fir_99(full)
% Shortest contiguous >=99% support containing zero lag. Ties: energy, then a.
h=full.h(:); D=full.numerical_delay_samples; n=numel(h);
assert(isreal(h) && all(isfinite(h)) && D>=0 && D<n && D==fix(D));
e=abs(h).^2; total=sum(e); assert(isfinite(total) && total>0);
target=.99*total; cumulative=[0;cumsum(e)]; zero=D+1;
b=max(zero,find(cumulative(2:end)>=target,1));
best_length=inf; candidates=zeros(zero,2); candidate_count=0;
for a=1:zero
    while b<n && cumulative(b+1)-cumulative(a)<target, b=b+1; end
    if cumulative(b+1)-cumulative(a)<target, break; end
    length_here=b-a+1;
    if length_here<best_length
        best_length=length_here; candidate_count=0;
    end
    if length_here==best_length
        candidate_count=candidate_count+1; candidates(candidate_count,:)=[a b];
    end
end
% Sum only the final shortest candidates; summing every moving window is quadratic.
best_energy=-inf; best_a=0; best_b=0;
for k=1:candidate_count
    a=candidates(k,1); b=candidates(k,2); energy=sum(e(a:b));
    if energy>=target && energy>best_energy
        best_a=a; best_b=b; best_energy=energy;
    end
end
assert(best_a>0 && best_a<=zero && best_b>=zero);
channel=full; channel.h=h(best_a:best_b);
channel.full_fir_samples=n; channel.trimmed_fir_samples=numel(channel.h);
channel.trim_left_samples=best_a-1; channel.trim_right_samples=n-best_b;
channel.full_memory_samples=n-1; channel.full_numerical_delay_samples=D;
channel.memory_samples=numel(channel.h)-1;
channel.numerical_delay_samples=D-channel.trim_left_samples;
channel.history_required_samples=channel.memory_samples-channel.numerical_delay_samples;
channel.lookahead_required_samples=channel.numerical_delay_samples;
channel.retained_energy_target=.99; channel.retained_energy_actual=sum(abs(channel.h).^2)/total;
channel.fir_variant='E99'; channel.context_rule='M-D';
assert(channel.retained_energy_actual>=.99 && channel.history_required_samples>=0 && ...
    channel.lookahead_required_samples>=0 && channel.first_arrival_s==full.first_arrival_s);
end
