function [idx,v] = findfirstdim(x,dim)

if isvector(x),
  idx = find(x,1);
  v = x(idx);
  return;
end;

sz = size(x);
nd = ndims(x);

if ~exist('dim'),
  dim = 1;
end;

% put dim first
notdim = setdiff(1:nd,dim);
x = permute(x,[notdim,dim]);
% reshape so that it is a 2D array
x = reshape(x,[prod(sz(notdim)),sz(dim)]);

[it, jt, vt] = find(x);
it = it(:); jt = jt(:); vt = vt(:);
[it, k] = sort(it); 
t = logical(diff([0;it])); 
j = repmat(NaN, [size(x,1) 1]); 
v = j; 
j(it(t)) = jt(k(t)); 
v(it(t)) = vt(k(t));

% shape back
idx = reshape(j,[sz(notdim),1]);
v = reshape(v,[sz(notdim),1]);



