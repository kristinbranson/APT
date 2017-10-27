function szassert(x,sz)
sz = rmTrailingSingletonDims(sz);
if ~isequal(size(x),sz)
  me = MException('sz:assert','Unexpected array size.');
  me.throwAsCaller();
end

