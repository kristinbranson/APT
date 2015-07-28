function r = neighborIndices(n)
x = -n:1:n;
[~,idx] = sort(abs(x));
r = x(idx);