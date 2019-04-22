% [d,xi,yi,thetai] = ellipsedist_hack(xc,yc,a,b,theta,u,v)
% compute distance from points (u,v) to ellipse (xc,yc,a,b,theta)
function [d,xi,yi,thetai] = ellipsedist_hack(xc,yc,a,b,theta,u,v,npoints)

if nargin < 8,
  npoints = 100;
end

[x,y,theta] = ellipsepoints(a,b,xc,yc,theta,npoints);
[d,i] = min(dist2([x(:),y(:)],[u(:),v(:)]),[],1);
d = sqrt(d);
xi = x(i); yi = y(i); thetai = theta(i);
d = reshape(d,size(u));
