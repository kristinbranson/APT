% [mu,S] = weighted_mean_cov(x,w)
% x is n x d, w is n x 1
function [mu,S] = weighted_mean_cov(x,w)

%[n,d] = size(x);

z = sum(w);
mu = sum(bsxfun(@times,x,w(:)),1)/z;

diffs = bsxfun(@minus,x,mu);
diffs = bsxfun(@times,diffs,sqrt(w));
S = (diffs'*diffs)/z;
