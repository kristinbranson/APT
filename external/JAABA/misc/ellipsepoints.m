% [X,Y] = ellipsepoints(a,b,x0,y0,phi)
function [X,Y,theta] = ellipsepoints(a,b,x0,y0,phi,npoints)

if (nargin < 2)||(nargin > 6),
    error('Usage: [X,Y,theta] = ellipsepoints(a,b,x0,y0,phi,[npoints])');
    
elseif nargin == 2
    x0 = 0;     y0 = 0;
    phi = 0;    
    
elseif nargin == 3
  phi = x0;
  x0 = 0; y0 = 0;
    
elseif nargin == 4     
    phi = 0;    
    
end

if nargin < 6,
  npoints = 60;
end

% persistent ELLIPSEPOINTS_CACHEDDATA_THETA
% if ~isempty(ELLIPSEPOINTS_CACHEDDATA_THETA) && numel(ELLIPSEPOINTS_CACHEDDATA_THETA) == npoints,
%   theta = ELLIPSEPOINTS_CACHEDDATA_THETA;
% else
%   theta = linspace(0,2*pi,npoints);
%   ELLIPSEPOINTS_CACHEDDATA_THETA = theta;
% end

theta = linspace(0,2*pi,npoints);


% Parametric equation of the ellipse
%----------------------------------------
 x = a*cos(theta);
 y = b*sin(theta);



% Coordinate transform 
%----------------------------------------
 X = cos(phi)*x - sin(phi)*y;
 Y = sin(phi)*x + cos(phi)*y;
 X = X + x0;
 Y = Y + y0;