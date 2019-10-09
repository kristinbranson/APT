function h = subplots(nr,nc,varargin)
h = createsubplots(nr,nc,varargin{:});
h = reshape(h,nr,nc);

