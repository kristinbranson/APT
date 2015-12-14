function p = rcprTestSelectOutput(pRT,regPrm,prunePrm)
% pRT: [NxDxRT1]. Final shapes over multiple replicates
% p: [NxD]. Final shapes, replicates collapsed

[N,D,RT1] = size(pRT); 

if prunePrm.usemaxdensity,
  p = nan(N,D);
  maxpr = nan(1,N);
  pRT1 = reshape(pRT,[N,regPrm.model.nfids,regPrm.model.d,RT1]);
  for n = 1:N,
    pr = 0;
    for part = 1:regPrm.model.nfids,
      d = pdist(reshape(pRT1(n,part,:,:),[regPrm.model.d,RT1])').^2; 
      %d = pdist(reshape(pRT(n,[1,3],:),[2,RT1])').^2;
      % d is pairwise distances between rows of [RT1xd] array of coords for
      % landmark 'part', trial n
      
      w = sum(squareform(exp( -d/prunePrm.maxdensity_sigma^2/2 )),1);
      % squareform(...) is the symmetric matrix of "boltzmann weights"
      % computed from squared-distances in d. small weight => large
      % distance, large weight => small distance
      %
      % sum(...,1) forms a sum-of-weights for the ith replicate. Larger 
      % values mean more likely replicate.
      w = w / sum(w);
      pr = pr + log(w); %accumulate by summing over pts/parts
    end
    [maxpr(n),i] = max(pr); % 'best' or most-likely replicate for this trial
    p(n,:) = pRT(n,:,i);
  end
      
else
  
  p = median(reshape(pRT,[N,D,RT1]),3);

end