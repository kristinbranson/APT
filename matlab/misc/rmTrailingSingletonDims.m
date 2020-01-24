function sz = rmTrailingSingletonDims(sz)
assert(isvector(sz));
n = numel(sz);
if n>2
  for i=n:-1:2
    if sz(i)~=1
      break;
    end
  end
  % i should be index of last non-1 element of sz, or 2 at minimum
  sz = sz(1:i);
end