% alignLandmarks(pts,mux,muy,theta)
% pts is 2 x n x T, where n is the number of landmarks
% mux is 1 x T, x-coord of centroid of the fly from Ctrax
% muy is 1 x T, x-coord of centroid of the fly from Ctrax
% theta is 1 x T, orientation of the fly from Ctrax

function pts = alignLandmarks(pts,mux,muy,theta)
% pts [2,n,T] 2 = x,y, n = # of landmarks on each fly (17), T = # labeled
% flies (rows in results table)
[d,n,T] = size(pts);
assert(d==2);
assert(numel(mux) == T);
assert(numel(muy) == T);
assert(numel(theta) == T);
% reshape mux, muy so that they can be subtracted from pts. mu has 2,1,T
% pts has 2,17,T => subtracts mu from all 17 pts! 
mu = cat(1,reshape(mux,[1,1,T]),reshape(muy,[1,1,T]));
theta = reshape(theta,[1,1,T]);
costheta = cos(theta);
sintheta = sin(theta);
% find pts in reference to fly centroid (mu)
pts = pts - mu;
% align to the x-axis by transforming with costheta and sintheta
pts = [costheta.*pts(1,:,:) + sintheta.*pts(2,:,:)
  -sintheta.*pts(1,:,:) + costheta.*pts(2,:,:)];
%pts = pts + mu;

