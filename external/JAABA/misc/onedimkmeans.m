% [mu,cost,idx,threshx] = onedimkmeans(x,k,varargin)
%
% the optimal k-clustering of n 1-dimensional points is computed in 
% time O(n^2*k), space O(n^2+n*k)
% 
% inputs:
% x: n x 1 vector of one-dimensional data points
% k: number of clusters
% optional:
% 'issorted': true/false, assumes the data is sorted from smallest 
% to largest if true
%
% outputs:
% mu: k x 1 vector of one-dimensional means
% cost: cost of clustering
% idx: n x 1 vector of cluster label for each point
% threshx: n+1 x 1 vector, where thresh(k1) <= cluster k1 < thresh(k1+1)
%
% there may currently by some inaccuracies if there are x's with the same value
function [mu,cost,idx,threshx] = onedimkmeans(x,k,varargin)

x = x(:);
n = length(x);

if k == 1,
  mu = mean(x);
  cost = sum((x-mu).^2);
  return;
end

if n < k,
  mu = sort(x( mod(0:k-1,n)+1 ));
  cost = 0;
  idx = ones(n,1);
  thresh = [1;n+1];
  return;
end

issorted = myparse(varargin,'issorted',false);

if ~issorted,
  [x,order] = sort(x);
  [tmp,reorder] = sort(order);
end
% the cost for one cluster from l to u is
% (u-l+1)*(sum(x(l:u).^2)/(u-l+1) - (sum(x(l:u))/(u-l+1)).^2)
% = sum(x(l:u).^2) - sum(x(l:u)).^2/(u-l+1)
% = c2(u+1)-c2(l) - (c(u+1)-c(l)).^2/(u-l+1)
c = [0;cumsum(x)];
c2 = [0;cumsum(x.^2)];
%issame = [false,x(1:end-1)==x(2:end)];

if k == 2,
  % u = i, l = 1, c(1) = 0
  % cost1 = c2(i+1) - (c(i+1)).^2/i;
  % cost1s(n1) is the cost of the cluster with l = 1, u = n1
  cost1s = c2(2:n) - c(2:n).^2./(1:n-1)';
  % cost2s(n1) is the cost of the cluster with l = n1+1, u = n
  cost2s = (c2(n+1)-c2(2:n)) - (c(n+1)-c(2:n)).^2./(n-1:-1:1)';
  costs = cost1s+cost2s;
  [cost,besti] = min(costs);
  mu = [mean(x(1:besti));mean(x(besti+1:end))];
  % may be some numerical inaccuracies with the cumsum approach
  %mu = [c(besti+1)/besti; (c(n+1)-c(besti+1))/(n-besti)];
  idx = ones(n,1);
  idx(besti+1:end) = 2;
  thresh = [1;besti+1;n+1];
  % convert from indices to data points
  threshx = zeros(k+1,1);
  threshx(thresh==1) = x(1);
  threshx(thresh==n+1) = x(n)+eps;
  other = thresh > 1 & thresh <= n;
  threshx(other) = (x(thresh(other)-1)+x(thresh(other)))/2;
  if ~issorted,
    idx = idx(reorder);
  end
  return;
end

% compute one-cluster cost for all [l,u] pairs
% only half of this matrix will actually be defined
oneclustercost = inf(n,n);
%for l = 1:n,
%oneclustercost(l,l:n) = c2(l+1:n+1)-c2(l) - (c(l+1:n+1)-c(l)).^2./(1:n-l+1);
%end
oneclustercost(1,1) = 0;
for u = 2:n,
  oneclustercost(1:u-1,u) = c2(u+1)-c2(1:u-1) - (c(u+1)-c(1:u-1)).^2./(u:-1:2)';
  oneclustercost(u,u) = 0;
end

% the cost for 1:k-1 clusters for samples 1:n
costs = inf(n,k-1);
prev = nan(n,k-1);

% initialize for k1 = 1
costs(:,1) = oneclustercost(1,:)';

% compute for increasing k, up to k-1
% costs(n1,k1) is the cost for clustering samples 1:n1 with k1 clusters
for k1 = 2:k-1,
  for n1 = 1:n,
    if n1 <= k1,
      costs(n1,k1) = 0;
      continue;
    end
    % prev(n1,k1) is the upper bound for the first set of clusters,
    % prev(n1,k1)+1 is the lower bound for the last cluster
    [costs(n1,k1),prev(n1,k1)] = min( costs(1:n1-1,k1-1) + oneclustercost(2:n1,n1) );
  end
end

% compute for n,k
% prevlast is the upper bound for the first set of clusters, 
% the last cluster starts at prevlast+1, that is, l = prevlast+1, u = n
[cost,prevlast] = min( costs(1:n-1,k-1) + oneclustercost(2:n,n) );
%prevlast = prevlast+1;

% get thresholds
% cluster k1 goes from thresh(k1) to thresh(k1+1)-1
thresh = ones(k+1,1);
thresh(end) = n+1;
for k1 = k:-1:1,
  if isnan(prevlast),
    thresh(1:k1) = 1:k1;
    break;
  end
  thresh(k1) = prevlast+1;
  if k1 > 1,
    prevlast = prev(prevlast,k1-1);
  end
end

idx = ones(n,1);
for k1 = 2:k,
  idx(thresh(k1):thresh(k1+1)-1) = k1;
end

% compute means
mu = zeros(k,1);
for k1 = 1:k,
  mu(k1) = mean(x(idx==k1));
end

if any(isnan(mu)),
  keyboard;
end

% recompute cost to avoid numerical inaccuracies
cost = 0;
for k1 = 1:k,
  cost = cost + sum((x(idx==k1)-mu(k1)).^2);
end

% convert from indices to data points
threshx = zeros(k+1,1);
threshx(thresh==1) = x(1);
threshx(thresh==n+1) = x(n)+eps;
other = thresh > 1 & thresh <= n;
threshx(other) = (x(thresh(other)-1)+x(thresh(other)))/2;

if ~issorted,
  idx = idx(reorder);
end

% here is some code for testing vs my kmeans fxn, mykmeans
if 0,
n = 1000;
k = 10;
ntests = 100;
diffcost = zeros(ntests,1);
sig = .1;
for test = 1:ntests,
  %x = sort(rand(n,1));
  idxtrue = randsample(k,n,true);
  mutrue = sort(rand(k,1));
  x = randn(n,1)*sig + mutrue(idxtrue);
  [x,order] = sort(x);
  idxtrue = idxtrue(order);
  tic;
  [idx,C,sumD] = mykmeans(x, k,'start','furthestfirst','replicates',20);  
  t = toc;
  [C,order] = sort(C);
  sumD = sumD(order);
  [tmp,order] = sort(order);
  idxkm = zeros(n,1);
  for k1 = 1:k,
    idxkm(idx==k1) = order(k1);
  end
  fprintf('Kmeans score = %f, time = %f, centers = ',sum(sumD),t);
  fprintf('%f, ',C);
  fprintf('\n');
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