function cnt = loadBalance(tot,avail)
% cnt = loadBalance(tot,avail)
%
% tot: scalar positive int, total number of trials needed
% avail: [Nbin] counts of available trials in 'bins' 1:Nbin
%
% cnt: [Nbin] counts of trials to use from each bin, with sum(cnt)=tot

assert(sum(avail)>=tot);
cnt = zeros(size(avail));

nBin = numel(avail);
iBin = 1;
sumcnt = 0;
while sumcnt<tot
  if avail(iBin)>0
    avail(iBin) = avail(iBin)-1;
    cnt(iBin) = cnt(iBin)+1;
    sumcnt = sumcnt+1;
  end
  iBin = mod(iBin,nBin)+1;
end