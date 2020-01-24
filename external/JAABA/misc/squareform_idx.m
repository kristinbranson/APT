function [i,j] = squareform_idx(value)

n = length(value);
m = ceil(sqrt(2*n));

% what indices do upper-triangular part of the matrix correspond to?
[tmp1,tmp2] = find(tril(ones(m),-1));
i = tmp1(value);
j = tmp2(value);