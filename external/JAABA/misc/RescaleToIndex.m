% idx = RescaleToIndex(x,n,l,u)
function idx = RescaleToIndex(x,n,l,u)

if nargin < 3,
  l = min(x(:));
end
if nargin < 4,
  u = max(x(:));
end

r = u-l;
w = r/n;

idx = max(1,min(floor((x-l)/w)+1,n));
