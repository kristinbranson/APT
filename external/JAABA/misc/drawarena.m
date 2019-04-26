function varargout = drawarena(varargin)

if length(varargin) >= 1&& isstruct(varargin{1}),
  r = varargin{1}.r;
  x = varargin{1}.x;
  y = varargin{1}.y;
  varargin = varargin(2:end);
else

if length(varargin) >= 1 && ~isstr(varargin{1}),
  r = varargin{1};
  varargin = varargin(2:end);
else,
  r = 500;
end;

if length(varargin) >= 1 && ~isstr(varargin{1}),
  x = varargin{1};
  varargin = varargin(2:end);
else,
  x = 0;
end;

if length(varargin) >= 1 && ~isstr(varargin{1}),
  y = varargin{1};
  varargin = varargin(2:end);
else,
  y = 0;
end;

end

t = linspace(0,2*pi,200);
h = plot(x+cos(t)*r,y+sin(t)*r,varargin{:});
if nargout > 0,
  varargout{1} = h;
end;