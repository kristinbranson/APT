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
% lower is a scalar. if lower == 1, then class 1 must be below the
% threshold and class -1 must be above. if lower == -1, then class
% -1 must be below the threshold and class 1 must be above. if
% lower == 0, then the best ordering is chosen
function [thresh1,thresh2,cost] = onedim2interval(x,l,costmatrix,weights)

% parse inputs
n = length(x);
if nargin < 3 || isempty(costmatrix),
  costmatrix = [-1 1 ; 1 1];
else
  if numel(costmatrix) ~= 4 | rows(costmatrix) ~= 2,
    error('costmatrix must be 2 x 2');
  end;
  if abs(costmatrix(1,1)) ~= 1 | abs(costmatrix(2,1)) ~= 1,
    error('costmatrix(1,1) and constmatrix(2,1) can either be 1 or -1');
  end;
  if costmatrix(1,1) == costmatrix(2,1),
    error('costmatrix(1,1) == costmatrix(2,1)');
  end;
end;
if nargin < 4 || isempty(weights),
  weights = ones(n,1);
end

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

N = length(x);
labels = l;

l(labels==costmatrix(1,1)) = costmatrix(1,1)*costmatrix(1,2).*weights(labels==costmatrix(1,1));
l(labels==costmatrix(2,1)) = costmatrix(2,1)*costmatrix(2,2).*weights(labels==costmatrix(2,1));
cs = cumsum([0;l]);
S = cs(end);
cs = cs*2;

% 1:thresh1 = costmatrix(1,1), thresh1+1:thresh2 = costmatrix(2,1),
% thresh2+1:N = costmatrix(1,1)
bestscore = -inf;
for i = 1:N+1,
  ncorrect = -cs(i) - S + cs(i:end);
  [maxcorrect,j] = max(ncorrect);
  if maxcorrect > bestscore,
    bestscore = maxcorrect;
    bestj = i + j - 1;
    besti = i;
  end
end

% Choose threshold to be in the middle
if besti == 1,
  thresh1 = x(1)-eps;
elseif besti == N+1,
  thresh1 = x(N)+eps;
else
  thresh1 = mean(x(besti-1:besti));
end
if bestj == 1,
  thresh2 = x(1)-eps;
elseif bestj == N+1,
  thresh2 = x(N)+eps;
else
  thresh2 = mean(x(bestj-1:bestj));
end

cost = -bestscore;