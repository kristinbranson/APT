% i'm not sure if this is exactly correct, but it is close enough

function samples = samplepnoreplace(x,k,w,maxiter)

if ~exist('maxiter','var'),
  maxiter = k;
end;
maxiter = min(maxiter,k);

if isscalar(x),
  n = x;
  x = 1:n;
else
  n = length(x);
end;

if length(w) ~= n,
  error('length of weights must be equivalent to number to sample from');
end

ignore = w <= 0;
x(ignore) = [];
n = n - nnz(ignore);
w(ignore) = [];

if k > n,
  error('k > n');
end

if k == 0,
  samples = [];
  return;
elseif k == n,
  samples = x;
  return;
end

dosample = false(1,n);
nneed = k;

for iter = 1:maxiter,
  idxleft = find(~dosample);
  newsamples = idxleft(randsample(length(idxleft),nneed,true,w(~dosample)));
  dosample(newsamples) = true;
  nneed = k - nnz(dosample);
  if nneed <= 0,
    break;
  end;
end;

if nneed > 0,
  idxleft = find(~dosample);
  newsamples = idxleft(randsample(length(idxleft),nneed));
  dosample(newsamples) = true;
end;

samples = x(dosample);
