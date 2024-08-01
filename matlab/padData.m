function data = padData(dat,t0,t1,nfrm)
% dat.data: [npt x nfrmdat] arr. nfrmdat=t1-t0+1
% data: [npt x nfrm]

if isstruct(dat)
  x = dat.data;
else
  x = dat;
end
sz = size(x);
assert(sz(2)==t1-t0+1);
data = cat(2,nan(sz(1),t0-1),x,nan(sz(1),nfrm-t1));