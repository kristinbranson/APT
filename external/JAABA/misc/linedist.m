function [Dperp,Dparallel] = linedist(Y,U,X)

d = size(X,2);
[C,d1] = size(Y);
if d1 ~= d,
  error('X and Y must be of the same dimensionality');
end
[C1,d2] = size(U);
if C ~= C1 || d1 ~= d2,
  error('Y and U must be the same size');
end

X = X';

Dparallel = bsxfun(@minus,X*U,sum(Y.*U,2)).^2;

Dperp = bsxfun(@plus,sum(X.^2,1),sum(Y.^2,2)) - 2*Y*X - Dparallel;
