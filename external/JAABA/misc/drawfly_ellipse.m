function varargout = drawfly_ellipse(x,y,varargin)

% draw an isosceles triangle with center (x,y)
% with rotation theta
% with height maj*4
% with base min*4

if isstruct(x),
  fly = x;
  t = y;
  x = fly.x(t);
  y = fly.y(t);
  theta = fly.theta(t);
  maj = fly.a(t);
  min = fly.b(t);  
else
  if nargin < 5,
    error('not enough arguments; usage: drawflyo(x,y,theta,a,b,...)');
  end
  theta = varargin{1};
  maj = varargin{2};
  min = varargin{3};
  varargin = varargin(4:end);
end

if mod(length(varargin),2) == 1,
  linestyle = varargin{1};
  varargin = varargin(2:end);
else
  linestyle = 'b';
end
h = ellipsedrawpatch(maj*2,min*2,x,y,theta,linestyle);
if ~isempty(varargin),
  set(h,varargin{:});
end
if nargout >=1,
  varargout{1} = h;
end