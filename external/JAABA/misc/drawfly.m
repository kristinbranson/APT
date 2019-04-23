% h = drawfly(trk,varargin)
% h = drawfly(x,y,theta,quartermaj,quartermin,varargin)
% draw triangle corresponding to fly position
% extra arguments will be fed to patch
function varargout = drawfly(x,y,varargin)

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


if 0,

h = ellipsedraw(maj*2,min*2,x,y,theta);

else

% isosceles triangle not yet rotated or centered
pts = [-maj*2,-min*2,
       -maj*2,min*2,
       maj*2,0];

% rotate
costheta = cos(theta);
sintheta = sin(theta);
R = [costheta,sintheta;-sintheta,costheta];
pts = pts*R;

% translate
pts(:,1) = pts(:,1) + x;
pts(:,2) = pts(:,2) + y;

% plot
if isempty(varargin),
  h = patch(pts([1:3,1],1),pts([1:3,1],2),'b');
else
  h = patch(pts([1:3,1],1),pts([1:3,1],2),varargin{:});
end

end

if nargout > 0,
  varargout{1} = h;
end;
