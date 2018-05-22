% [samplestarts,sampleends] = SelectWrappingSampleSubsets(K,N,nsample)
% K: number of intervals to sample
% N: number of things to sample from
% nsample: interval size

function [samplestarts,sampleends] = SelectWrappingSampleSubsets(K,N,nsample)

coroff = 0;
samplestarts = nan(1,K);
sampleends = nan(1,K);
for k = 1:K,
  samplestarts(k) = coroff+1;
  sampleends(k) = coroff+nsample;
  coroff = coroff + nsample;
  % next sample won't fit? 
  if coroff+nsample > N,
    sampleends(k) = N;
    coroff = 0;
  end
end
