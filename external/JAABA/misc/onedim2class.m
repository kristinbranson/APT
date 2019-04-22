% [thresh,lower,cost] = onedim2class(x,l,costmatrix,weights,lower)
% inputs:
% x is the N x 1 matrix of N unidimensional data points
% l is the N x 1 matrix of N corresponding labels, where l(i) can
% be either -1 or 1
% thresh is the best threshold such that almost everything on one
% side of the threshold is of one class and almost of everything on
% the other side is the other class.
% optional:
% costmatrix is the 2 x 2 matrix. the cost of misclassifying a point with
% label costmatrix(1,1) is costmatrix(1,2) and the cost of misclassifying a
% point with label costmatrix(2,1) is costmatrix(2,2). by default, this is 
% [-1 1 ; 1 1]. set to [] to use the default. 
% weights is a vector of length n, where w(n) is the weight of the nth
% sample. set to [] to use default w(n) = 1 for all n. 
% lower is a scalar. if lower == 1, then class 1 must be below the
% threshold and class -1 must be above. if lower == -1, then class
% -1 must be below the threshold and class 1 must be above. if
% lower == 0, then the best ordering is chosen
function [thresh,lower,cost] = onedim2class(x,l,costmatrix,weights,lower)

% parse inputs
n = length(x);
if nargin < 3 || isempty(costmatrix),
  costmatrix = [-1 1 ; 1 1];
else
  if numel(costmatrix) ~= 4 || rows(costmatrix) ~= 2,
    error('costmatrix must be 2 x 2');
  end;
  if abs(costmatrix(1,1)) ~= 1 || abs(costmatrix(2,1)) ~= 1,
    error('costmatrix(1,1) and constmatrix(2,1) can either be 1 or -1');
  end;
  if costmatrix(1,1) == costmatrix(2,1),
    error('costmatrix(1,1) == costmatrix(2,1)');
  end;
end;
if nargin < 4 || isempty(weights),
  weights = ones(n,1);
end

if nargin < 5,
  lower = 0;
else
  if abs(lower) ~= 1 && lower ~= 0,
    error('lower can be -1, 0, or 1');
  end;
end;
x = x(:);
l = l(:);
if length(x) ~= length(l),
  error('x and l must both be N x 1 vectors');
end;
classes = unique(l);
if length(classes) ~= 2, 
  error('l must contain exactly 2 unique values, -1 and 1');
end;

% sort the data
% Sort the data based on response value
[sortedx,sortedi] = sort(x);
sortedl = l(sortedi);
sortedweights = weights(sortedi);

% collapse entries of x that are identical
isid = [false;diff(sortedx) == 0];
if any(isid),
  [startid,endid] = get_interval_ends(isid);
  for i = 1:length(startid),
    t0 = startid(i)-1;
    t1 = endid(i)-1;
    tmp = sum(sortedweights(t0:t1).*sortedl(t0:t1));
    sortedweights(t0) = abs(tmp);
    sortedl(t0) = sign(tmp);
  end
  sortedx(isid) = [];
  sortedl(isid) = [];
  sortedweights(isid) = [];
end

if lower == 0,
  [thresh_lowerpos,cost_lowerpos] = main(sortedx, sortedl, costmatrix, sortedweights, 1);
  [thresh_lowerneg,cost_lowerneg] = main(sortedx, sortedl, costmatrix, sortedweights, -1);  
  if cost_lowerpos < cost_lowerneg,
    thresh = thresh_lowerpos;
    cost = cost_lowerpos;
    lower = 1;
  else
    thresh = thresh_lowerneg;
    cost = cost_lowerneg;
    lower = -1;
  end;
else 
  [thresh,cost] = main(sortedx, sortedl, costmatrix, sortedweights,lower);
end;

function [thresh,cost] = main(x, l, costmatrix, weights, lower)

N = length(x);
labels = l;

l(labels==costmatrix(1,1)) = costmatrix(1,1)*costmatrix(1,2).*weights(labels==costmatrix(1,1));
l(labels==costmatrix(2,1)) = costmatrix(2,1)*costmatrix(2,2).*weights(labels==costmatrix(2,1));
suml1 = cumsum([0;l]);
suml2 = [flipud(cumsum(flipud(l)));0];
if lower == -1,
  ncorrect = -suml1 + suml2;
else
  ncorrect = suml1 - suml2;
end
[maxncorrect,besti] = max(ncorrect);

% Choose threshold to be in the middle
if besti == 1,
  thresh = x(1)-eps;
elseif besti == N+1,
  thresh = x(N)+eps;
else
  thresh = mean(x(besti-1:besti));
end;
cost = -maxncorrect;