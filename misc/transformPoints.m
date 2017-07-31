function uv = transformPoints(uv0,xy0,th0,xy1,th1)
% uv0: nptsx2 array of (x,y) coordinates
% xy0: single (x0,y0) coordinate pair
% th0: theta0, scalar angle
% xy1: etc
% th1: etc
%
% uv: npts2x array of transformed points
%
% Let F be the transformation (shift+rotatio) that maps (x0,y0,theta0) onto
% (x1,y1,theta1). Then uv(i,:) = F(uv0(i,:)).

assert(size(uv0,2)==2);
npts = size(uv0,1);

assert(numel(xy0)==2);
assert(isscalar(th0));
assert(numel(xy1)==2);
assert(isscalar(th1));
xy0 = xy0(:)';
xy1 = xy1(:)';

dtheta = th1-th0;
rotmat = [cos(dtheta) -sin(dtheta);sin(dtheta) cos(dtheta)];

uv = nan(npts,2);
for i = 1:npts
  dxy0 = uv0(i,:)-xy0;
  dxy = rotmat*dxy0.';  
  uv(i,:) = xy1 + dxy.';
end
