function p = rcprTestSelectOutput(pRT,regPrm,prunePrm)

[N,D,RT1]=size(pRT); 

if prunePrm.usemaxdensity,
  p = nan(N,D);
  maxpr = nan(1,N);
  pRT1 = reshape(pRT,[N,regPrm.model.nfids,regPrm.model.d,RT1]);
  for n = 1:N,
    pr = 0;
    for part = 1:regPrm.model.nfids,
      d = pdist(reshape(pRT1(n,part,:,:),[regPrm.model.d,RT1])').^2;
      %d = pdist(reshape(pRT(n,[1,3],:),[2,RT1])').^2;
      w = sum(squareform(exp( -d/prunePrm.maxdensity_sigma^2/2 )),1);
      w = w / sum(w);
      pr = pr + log(w);
    end
    [maxpr(n),i] = max(pr);
    p(n,:) = pRT(n,:,i);
  end
      
else
  
  p = median(reshape(pRT,[N,D,RT1]),3);

end