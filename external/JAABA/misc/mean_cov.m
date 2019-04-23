% [mu,S] = mean_cov(x)
% x is n x d
function [mu,S] = mean_cov(x)

%[n,d] = size(x);

z = size(x,1);
mu = sum(x,1)/z;

diffs = bsxfun(@minus,x,mu);
S = (diffs'*diffs)/z;
