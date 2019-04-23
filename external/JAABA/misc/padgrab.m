% [y,npadl,npadu] = padgrab(x,padv,l1,u1,l2,u2,...)
function [y,npadl,npadu] = padgrab(x,padv,varargin)

% parse arguments
usagestring = 'Usage: padgrab(x,padv,l1,u1,l2,u2,...)';
nd1 = ndims(x);
if mod(numel(varargin),2) ~= 0 || isempty(varargin),
  error(usagestring);
end
isones = cellfun(@(x) x == 1, varargin(1:2:end-1)) & cellfun(@(x) x == 1, varargin(2:2:end));
sz = size(x);
isones(1:nd1) = isones(1:nd1) & sz == 1;
nd = find(~isones,1,'last');
if isempty(nd),
  nd = 1;
end
varargin = varargin(1:2*nd);
isrowvec = nd == 1 && nd1 == 2 && size(x,1) == 1;
if isrowvec,
  varargin = [{1,1},varargin];
  nd = 2;
end
if mod(nd,1) ~= 0 || nd > nd1,
  error(usagestring);
end
l = cell2mat(varargin(1:2:end));
u = cell2mat(varargin(2:2:end));
%if any(u < l),
%  error(usagestring);
%end
%sz = size(x);
if nd < nd1,
  x = reshape(x,[sz(1:nd-1),prod(sz(nd:end)),1]);
  sz = size(x);
end

l1 = max(min(l,sz),1);
u1 = max(min(u,sz),1);
a = cell(1,nd);
aisempty = false(1,nd);
for i = 1:nd,
  if sz(i) <= 0 || l(i) > u(i) || u(i) < 1 || l(i) > sz(i),
    a{i} = [];
    aisempty(i) = true;
  else
    a{i} = l1(i):u1(i);
  end
end
y = x(a{:});
npadl = l1-l+aisempty;
npadu = u-u1;
npadl = npadl + (npadu < 0).*npadu;

idxlow = u < 1;
if any(idxlow),
  npadl(idxlow) = u(idxlow) - l(idxlow) + 1;
  npadu(idxlow) = 0;
end
idxhigh = l > sz;
if any(idxhigh),
  npadu(idxhigh) = u(idxhigh) - l(idxhigh) + 1;
  npadl(idxhigh) = 0;
end

if any(npadl > 0),
  y = padarray(y,npadl,padv,'pre');
end
if any(npadu > 0),
  y = padarray(y,npadu,padv,'post');
end

% if dotranspose,
%   y = y';
% end
