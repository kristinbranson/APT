function y = meanfilter(x,r,dir)

if ~exist('dir','var'),
  if size(x,1) == 1 && size(x,2) > size(x,1),
    dir = 2;
  else
    dir = 1;
  end    
end
sz = size(x);
notdir = true(1,length(sz)); notdir(dir) = false;
notdiridx = find(notdir);
x = permute(x,[dir,notdiridx]);
m = prod(sz(notdir));
x = reshape(x,[sz(dir),m]);

off1 = floor(r/2);
off2 = r - off1 - 1;
y = imfilter(x,ones(r,1),'same',0);
y(1:off1,:) = y(1:off1)./repmat(2*(1:off1),[m,1]);
y(end-off2+1:end,:) = y(end-off2+1:end,:) ./ repmat(2*(off2:-1:1),[m,1]);
y(off1+1:end-off2,:) = y(off1+1:end-off2,:) / r;

y = reshape(y,[sz(dir),sz(notdir)]);
y = permute(y,[2:dir,1,dir+1:length(sz)]);