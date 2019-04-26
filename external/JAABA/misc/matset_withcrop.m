function x = matset_withcrop(x,y,varargin)

nd = length(varargin);
szx = size(x);

if nd == 1 && nnz(szx~=1) <= 1,
  % special case for vectors
  inbounds = varargin{1} >= 1 & varargin{1} <= max(szx);
  x(varargin{1}(inbounds)) = y(inbounds);
else
  idxx = cell(1,nd);
  idxy = cell(1,nd);
  for d = 1:nd,
    inbounds = varargin{d} >= 1 & varargin{d} <= szx(d);
    idxx{d} = varargin{d}(inbounds);
    idxy{d} = inbounds;
  end
  x(idxx{:}) = y(idxy{:});
end