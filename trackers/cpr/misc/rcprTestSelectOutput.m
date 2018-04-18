function [p,maxpr,diag] = rcprTestSelectOutput(pRT,model,prunePrm)
% pRT: [NxDxRT1]. Final shapes over multiple replicates
% p: [NxD]. Final shapes, replicates collapsed

warning('This function is deprecated. See Prune.maxdensity.');

assert(ndims(pRT)==3);
[N,D,RT1] = size(pRT); 
diag = cell(N,1);

if prunePrm.usemaxdensity
  p = nan(N,D);
  maxpr = nan(N,1);
  pRT1 = reshape(pRT,[N,model.nfids,model.d,RT1]);
  for n = 1:N
    pr = 0;
    for part = 1:model.nfids
      d = pdist(reshape(pRT1(n,part,:,:),[model.d,RT1])').^2; 
      %d = pdist(reshape(pRT(n,[1,3],:),[2,RT1])').^2;
      % d is pairwise distances between rows of [RT1xd] array of coords for
      % landmark 'part', trial n
      
      % The sum-of-guassians here is ~ a "neighbor-counter" where each
      % neighbor within sigma adds a 1 and all others have no effect.
      %
      % squareform(...) is the symmetric matrix of "boltzmann weights"
      % computed from squared-distances in d. small weight => large
      % distance, large weight => small distance
      %
      % sum(...,1) forms a sum-of-weights for the ith replicate. Larger 
      % values mean more likely replicate.
      w = sum(squareform(exp( -d/prunePrm.maxdensity_sigma^2/2 )),1);
      
      % Normalizing gives a 'probability' per replicate (for this
      % landmark/part). Replicates
      w = w / sum(w);
            
      pr = pr + log(w); %accumulate by summing over pts/parts
      
      diag{n}(:,part) = d; % squared dists
    end
    [maxpr(n),i] = max(pr); % 'best' or most-likely replicate for this trial
    p(n,:) = pRT(n,:,i);
  end      
else
  p = median(pRT,3);
  maxpr = mean(std(pRT,1,3),2);
end