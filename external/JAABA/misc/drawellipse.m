% h = drawellipse(trk,varargin)
% h = drawellipse(x,y,theta,semimaj,semimin,varargin)
% extra arguments will be fed to plot
function varargout = drawellipse(x,varargin)

% draw an isosceles triangle with center (x,y)
% with rotation theta
% with height maj*4
% with base min*4

if isstruct(x),
  fly = x;
  if nargin == 1 || ~isnaturalnumber(varargin{1}),
    x = [fly.x];
    y = [fly.y];
    theta = [fly.theta];
    a = [fly.a];
    b = [fly.b];
  else
    t = varargin{1};
    varargin = varargin(2:end);
    x = fly.x(t);
    y = fly.y(t);
    theta = fly.theta(t);
    a = fly.a(t);
    b = fly.b(t);
  end
else
  if nargin < 5,
    error('not enough arguments; usage: drawflyo(x,y,theta,a,b,...)');
  end
  y = varargin{1};
  theta = varargin{2};
  a = varargin{3};
  b = varargin{4};
  varargin = varargin(5:end);
end

phi = -0.03:0.01:2*pi;

nx = numel(x); ny = numel(y);
na = numel(a); nb = numel(b); 
ntheta = numel(theta);
n = max([nx,ny,na,nb,ntheta]);
h = zeros(1,n);

for i = 1:n,
  X1 = a(min(i,na))*cos(phi);
  Y1 = b(min(i,nb))*sin(phi);
  costheta = cos(theta(min(i,ntheta)));
  sintheta = sin(theta(min(i,ntheta)));
  X = costheta*X1 - sintheta*Y1 + x(min(i,nx));
  Y = sintheta*X1 + costheta*Y1 + y(min(i,ny));
  h(i) = plot(X,Y,varargin{:});
end

if nargout > 0,
  varargout{1} = h;
end;
