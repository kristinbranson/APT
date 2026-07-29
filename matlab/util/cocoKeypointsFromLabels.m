function keypoints = cocoKeypointsFromLabels(p, tfocc)
% Convert one target's APT label positions to COCO-style keypoint triplets.
%
% p: [1 x pointCount*2] label positions for one target, laid out as
%   [x_all y_all], in absolute movie coordinates (1-based).  For multiview
%   projects, pointCount includes the points of all views, concatenated.
% tfocc: [1 x pointCount] logical, true where the labeled point is occluded.
%
% keypoints: [1 x pointCount*3] COCO keypoints [x1 y1 v1 x2 y2 v2 ...],
%   where v is 2 for labeled-and-visible, 1 for labeled-but-occluded, and 0
%   for unlabeled points.  x and y are set to 0 for unlabeled points.
%   Coordinates stay 1-based; consumers converting to 0-based indexing
%   (e.g. the Python backend) must subtract 1.

pointCount = numel(p) / 2 ;
x = p(1:pointCount) ;
y = p(pointCount+1:end) ;
v = repmat(2, [1 pointCount]) ;
v(logical(tfocc)) = 1 ;
% Unlabeled points are nan; fully-occluded labels without a position are inf.
% Neither has usable coordinates, so mark both as unlabeled.
isUnlabeled = isnan(x) | isnan(y) | isinf(x) | isinf(y) ;
v(isUnlabeled) = 0 ;
x(isUnlabeled) = 0 ;
y(isUnlabeled) = 0 ;
keypoints = reshape([x(:)' ; y(:)' ; v(:)'], 1, []) ;
end  % function
