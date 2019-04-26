%  [intpx,extpx] = ellipseintextpixels(im,ells,r0,r1,r2)
function [intpx,extpx] = ellipseintextpixels(im,ells,r0,r1,r2,debug)

if ~exist('debug','var'),
  debug = false;
end

[nr,nc,ncolors] = size(im);
bw = false(nr,nc);
if r0 < 0,
  se0 = strel('disk',-r0);
end
if r1 > 0,
  se1 = strel('disk',r1);
end
if r2 > 0,
  se2 = strel('disk',r2);
end

% regions inside some ellipse
for i = 1:length(ells),
  ell = ells(i);

  % get a bounding box around the ellipse
  bb = [ell.y+ell.a*[-1,1],ell.x+ell.a*[-1,1]];
  bb([1,3]) = floor(bb([1,3]));
  bb([2,4]) = ceil(bb([2,4]));
  bb = max(bb,1);
  bb(1:2) = min(bb(1:2),nr);
  bb(3:4) = min(bb(3:4),nc);
  
  % get ellipse pixels
  bw1 = ellipsepixels([ell.x,ell.y,ell.a*2,ell.b*2,ell.theta],bb);
  bw(bb(1):bb(2),bb(3):bb(4)) = bw(bb(1):bb(2),bb(3):bb(4)) | bw1;
end

% erode if necessary to get interior points
if r0 < 0,
  bwint = imerode(bw,se0);
else
  bwint = bw;
end

% dilate to get inside ignored points
if r1 > 0,
  bwskip1 = imdilate(bw,se1);
else
  bwskip1 = bw;
end

% dilate more to get outside ignored points
if r2 > 0,
  bwskip2 = ~imdilate(bw,se2);
else
  bwskip2 = ~bw;
end

% points not inside ellipse + r1 and not outside ellipse + r2
bwext = ~bwskip1 & ~bwskip2;

nint = nnz(bwint);
next = nnz(bwext);
intpx = zeros(nint,ncolors);
extpx = zeros(next,ncolors);

for color = 1:ncolors,
  tmp = im(:,:,color);
  intpx(:,color) = tmp(bwint);
  extpx(:,color) = tmp(bwext);
end

if debug,
  clf;
  image(uint8(cat(3,double(bwint)*255,double(bwext)*255,im))); 
  axis image;
end