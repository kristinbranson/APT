function szassert(x,sz,varargin)
sz = rmTrailingSingletonDims(sz);
assert(isequal(size(x),sz),varargin{:});

