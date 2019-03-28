function y = cellSelectHelper(x,varargin)

y = x(varargin{:});
if iscell(y) && numel(y) == 1,
  y = y{1};
end