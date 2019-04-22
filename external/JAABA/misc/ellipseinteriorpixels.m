%  px = ellipseinteriorpixels(im,ell)
function px = ellipseinteriorpixels(im,ell)

[nr,nc,ncolors] = size(im);

% get a bounding box around the ellipse
bb = [ell.y+ell.a*[-1,1],ell.x+ell.a*[-1,1]];
bb([1,3]) = floor(bb([1,3]));
bb([2,4]) = ceil(bb([2,4]));
bb = max(bb,1);
bb(1:2) = min(bb(1:2),nr);
bb(3:4) = min(bb(1:2),nc);

% crop out this region
imbb = im(bb(1):bb(2),bb(3):bb(4),:);
bw = ellipsepixels([ell.x,ell.y,ell.a*2,ell.b*2,ell.theta],bb);
idx = find(bw);
n = length(idx);
px = zeros(n,ncolors);
for color = 1:ncolors,
  px(:,color) = imbb(idx+(color-1)*nr*nc);
end
