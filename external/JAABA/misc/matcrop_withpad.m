function mcrop = matcrop_withpad(m,varargin)

subs = varargin;

ndim = length(subs);
sz = size(m);
if sz(1) == 1 && ndim == 1,
  subs = [1,subs];
  ndim = 2;
elseif length(sz) ~= ndim,
  error('size(m) does not match number of input indices');
end

mcrop = m;
for dim = 1:ndim,
  if subs{dim}(1) > subs{dim}(2),
    mcrop = eval(['mcrop(',repmat(':,',[1,dim-1]),'[]',repmat(',:',[1,ndim-dim]),');']);
    continue;
  end
  npadprev = max(0,1 - subs{dim}(1));
  npadafter = max(0,subs{dim}(2)-sz(dim));
  subs{dim}(1) = max(1,subs{dim}(1));
  subs{dim}(2) = min(sz(dim),subs{dim}(2));
  mcrop = eval(['mcrop(',repmat(':,',[1,dim-1]),sprintf('%d:%d',subs{dim}(1),subs{dim}(2)),...
    repmat(',:',[1,ndim-dim]),');']);
  if npadprev > 0,
    tmppad = size(mcrop);
    tmppad(dim) = npadprev;
    mcrop = cat(dim,zeros(tmppad),mcrop);
  end
  if npadafter > 0,
    tmppad = size(mcrop);
    tmppad(dim) = npadafter;
    mcrop = cat(dim,mcrop,zeros(tmppad));
  end
end