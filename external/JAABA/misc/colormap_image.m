% Irgb = colormap_image(I,c,clim)

function Irgb = colormap_image(I,c,clim)

if nargin < 2 || isempty(c),
  c = colormap;
end
if nargin < 3 || isempty(clim),
  clim = nan(1,2);
end
if ndims(I) < 2 || ndims(I) > 3,
  error('input image must be N x M');
end;
if isnan(clim(1)),
  a = min(I(:));
else
  a = clim(1);
end
if isnan(clim(2)),
  b = max(I(:));
else
  b = clim(2);
end
d = b-a;
n = size(c,1);
nr = size(I,1);
nc = size(I,2);
nz = size(I,3);
J = min(n,max(1,round((I - a)/d*(n-1)+1)));
if nz > 1,
  Irgb = reshape(c(J(:),:),[nr,nc,nz,3]);
else
Irgb = reshape(c(J(:),:),[nr,nc,3]);
end