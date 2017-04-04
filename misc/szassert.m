function szassert(x,sz,varargin)
assert(isequal(size(x),sz),varargin{:});