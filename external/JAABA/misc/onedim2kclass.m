% [threshx,cost,classorder] = onedim2kclass(x,idx,...)
%
% finds the k-1 thresholds that best separate the input one-dimensional
% data vector x into k classes
%
% input:
%
% x: n x 1 is the n one-dimensional data points
% idx: n x 1 is the integer labels for each data point
% 
% optional:
%
% 'issorted': true/false, whether the list is sorted already
% 'classorder': 1 x k vector where classorder(1) comes first and 
% classorder(k) comes last. otherwise, classorder is set based on means of each 
% class
%
% output:
% threshx: k+1x1, class classorder(i) is >= threshx(i) and < thresh(i+1)
% cost: number of misclassified samples
% classorder: 1 x k
% 
function [threshx,cost,classorder] = onedim2kclass(x,idx,varargin)

x = x(:);
idx = idx(:);
n = length(x);


[issorted,classorder] = myparse(varargin,'issorted',false,'classorder',nan);

%k = max(idx); % right now, assuming labels are 1,...,k

% make labels 1,...,k
oldidx = idx;
[newidx2oldidx0,tmp,idx] = unique(idx);
%[tmp,oldidx2newidx0] = sort(newidx2oldidx0);
k = length(newidx2oldidx0);
if ~any(isnan(classorder)),
  classorder0 = classorder;
  for i = 1:k,
    classorder(classorder0==newidx2oldidx0(i)) = i;
  end
end

if k == 1,
  threshx = [min(x),max(x)+eps];
  cost = 0;
  if isnan(classorder),
    classorder = newidx2oldidx0;
  end
  return;
end

%if k == 2,
%  % use onedim2class
%  % idx should be -1,1
%  idx = 2*idx-3;
%  [thresh,lower,cost] = onedim2class(x,idx);
%  if lower == -1,
%    order = [1,2];
%  else
%    order = [2,1];
%  end
%  classorder = order;
%  %classorder = newidx2oldidx(order);
%  return;
%end

% choose order for classes based on order of means
if any(isnan(classorder)),
  
  mu = zeros(1,k);
  for k1 = 1:k,
    mu(k1) = mean(x(idx==k1));
  end
  [sortedmu,order] = sort(mu);
  [tmp,oldidx2newidx] = sort(order);
  classorder = newidx2oldidx0(order);
  % relabel
  idx0 = idx;
  for k1 = 1:k,
    idx(idx0==k1) = oldidx2newidx(k1);
  end

else
  [tmp,oldidx2newidx] = sort(classorder);
end


if ~issorted,
  [x,xorder] = sort(x);
  idx = idx(xorder);
  [tmp,xreorder] = sort(xorder);
end
% deal with duplicates: cannot have a class start at i if x(i) == x(i-1)
% isdup(i) = x(i) == x(i-1)
isdup = [x(1:end-1) == x(2:end);false];
dupidx = find(isdup);

% costs(n1+1,k1) is the min cost of classifying samples 1:n1 as classes 1:k1
costs = inf(n+1,k-1);
% prev(n1+1,k1)+1 is the start of class k1 when classifying samples 1:n1
prev = zeros(n+1,k-1);

% base cases

% for i = 1,...,n
% cost(i+1,1) is cost of classifying 1:i as 1 is 
% sum(idx(1:i)~=k) = c(i+1)
% for i = 0, cost(i+1,1) = 0
costs(2:n+1,1) = cumsum(double(idx~=1));
% cannot end at dupidx - 1
costs(dupidx+1) = inf;

% for k1 = 1,...,k cost of classifying 1:0 as 1:k1 is 0
costs(1,:) = 0;

% compute for increasing k, up to k-1
for k1 = 2:k-1,
  
  c = [0;cumsum(double(idx~=k1))];
  
  for n1 = 1:n,
    if isdup(n1), continue; end
    % compute one-class cost for i = 1:n1+1, end = n1
    oneclasscost = c(n1+1)-c(1:n1+1);
    % prev(n1+1,k1) is the start of the last class
    [costs(n1+1,k1),prev(n1+1,k1)] = min( costs(1:n1+1,k1-1) + oneclasscost );
  end
  
end

% compute for n,k
c = [0;cumsum(double(idx~=k))];
oneclasscost = c(n+1)-c(1:n+1);
[cost,prevlast] = min( costs(1:n+1,k-1) + oneclasscost );

% get thresholds
% class k1 goes from thresh(k1) to thresh(k1+1)-1
thresh = ones(k+1,1);
thresh(end) = n+1;
for k1 = k:-1:1,
  if prevlast < 1,
    thresh(1:k1) = 1;
    break;
  end
  thresh(k1) = prevlast;
  if k1 > 1,
    prevlast = prev(prevlast,k1-1);
  end
end

% convert from indices to data points
threshx = zeros(k+1,1);
threshx(thresh<=1) = x(1);
threshx(thresh==n+1) = x(n)+1;
other = thresh > 1 & thresh <= n;
threshx(other) = (x(thresh(other)-1)+x(thresh(other)))/2;
%threshx(other) = x(thresh(other));

if exist('classorder0','var')
  classorder = classorder0;
end

% here is some code for testing
if 0,
n = 100;
k = 3;
ntests = 100;
sig = .1;
for test = 1:ntests,
  %x = sort(rand(n,1));
  idxtrue = randsample(k,n,true);
  mutrue = sort(rand(k,1));
  x = randn(n,1)*sig + mutrue(idxtrue);
  [x,order] = sort(x);
  idxtrue = idxtrue(order);
  tic;
  [mu,cost,idxdp,threshx] = onedimkmeans(x,k);
  t = toc;
  fprintf('DP score = %f, diff = %e, time = %f, centers = ',cost,cost-sum(sumD),t);
  fprintf('%f, ',mu);
  fprintf('\n');
  if cost > sum(sumD),
    if all(idxkm==idxdp),
      fprintf('Bigger cost, diff = %e, but all idx are the same\n',cost-sum(sumD));
      pause(1);
    else
      fprintf('*****Bigger cost, diff = %e, differing indices\n',cost-sum(sumD));
      fprintf('Differing indices: ');
      fprintf('%d, ',find(idxkm~=idxdp));
      fprintf('\n');
      break;
    end
  end
  diffcost(test) = cost-sum(sumD);
  fprintf('\n');
end
end