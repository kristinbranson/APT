function [mu,var,sd,med,mad] = freqCountStats(x,cnt)
% [mu,var,sd] = freqCountStats(x,cnt)
% Compute stats based on frequency counts. 
%
% x: [N] values
% cnt: [N] count for each value
%
% med: median
% mad: median absolute dev (from median)

assert(isvector(x) && isequal(size(x),size(cnt)));

% sort vectors by x for median calcs
[x,idx] = sort(x);
cnt = cnt(idx);

n = sum(cnt);
mu = sum(x.*cnt)/n;
var = sum(x.^2.*cnt)/n - mu^2;
sd = sqrt(var);

cntssum = cumsum(cnt);
iMed = find(cntssum>n/2,1);
med = x(iMed);
absDev = abs(x-med);

if nargout>=5 % to prevent infinite recursion  
  % the mad is the median of absDev
  [~,~,~,mad] = freqCountStats(absDev,cnt); 
end

