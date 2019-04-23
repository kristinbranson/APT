% bw = ellipsepixels(params,bb)
% params: vector of length 5 describing the ellipse, where
% params(1) is the x coordinate of the center
% params(2) is the y coordinate of the center
% params(3) is the major axis length
% params(4) is the minor axis length
% params(5) is the orientation
% bb is a vector of length 4 describing the bounding box,
% where bb = [ymin, ymax, xmin, xmax]

function bw = ellipsepixels(params,bb,speed)

if params(3) < eps || params(4) < eps,
  bw = false(bb(2)-bb(1)+1,bb(4)-bb(3)+1);
  return;
end;

if ~exist('speed','var'),
  speed = false;
end

if speed,
  bb0 = bb;
  [bb(3),bb(4),bb(1),bb(2)] = ellipse_to_bounding_box(params(1),params(2),params(3)/2,params(4)/2,params(5));
  bb([1,3]) = floor(bb([1,3]));
  bb([2,4]) = ceil(bb([2,4]));
  bb([1,3]) = max(bb([1,3]),bb0([1,3]));
  bb([2,4]) = min(bb([2,4]),bb0([2,4]));
end

[x,y] = meshgrid(bb(3):bb(4),bb(1):bb(2));
[nr,nc] = size(x);

% subtract mean
x = x - params(1);
y = y - params(2);
dx = [x(:)';y(:)'];

% compute covariance matrix
S = axes2cov(params(3)/2,params(4)/2,params(5));
S = chol(S);

% compute Mah distance
dx = solve_tril(S',dx);
d = sum(dx.*dx,1);
d = reshape(d,[nr,nc]);

% threshold distance
bw = d<=4;

if speed,
  bw0 = bw;
  bw = false(bb0(2)-bb0(1)+1,bb0(4)-bb0(3)+1);
  bw(bb(1)-bb0(1)+1:bb(2)-bb0(1)+1,bb(3)-bb0(3)+1:bb(4)-bb0(3)+1) = bw0;
end