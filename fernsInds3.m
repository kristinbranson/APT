function [inds,mu,ysFern,count,ysFernCnt] = fernsInds3(X,fids,thrs,Y)

[N,F] = size(X);
[Ntmp,D] = size(Y);
assert(N==Ntmp);
S = numel(fids);

mu = nanmean(Y);
dY = bsxfun(@minus,Y,mu);
S2 = 2^S;

inds = zeros(N,1);
for s=1:S
  f = fids(s);
  for i=1:N
    inds(i) = inds(i)*2;
    if X(i,f)<thrs(s)
      inds(i) = inds(i)+1;
    end
  end
end
inds = inds+1;

count = zeros(S2,1);
ysFern = zeros(S2,D);
ysFernCnt = zeros(S2,D);
for i = 1:N
  s = inds(i);
  count(s) = count(s)+1;
  
  tfgood = ~isnan(dY(i,:));
  ysFern(s,tfgood) = ysFern(s,tfgood) + dY(i,tfgood);
  ysFernCnt(s,tfgood) = ysFernCnt(s,tfgood) + 1;
end
