% [mu,idx,mu_idx,Dall,mindists] = furthestfirst(x,k,p1,v1,...)
% 
% x: N x D matrix of data to be clustered. N is the number of data
% points, D is the number of dimensions
% k: number of centers
% Optional PVs:
%   - Start. Either [1xD], 'mean' (default), or empty. first center. If  
%   empty, a random point is used as the first center.
%   - Distance. Distance function (string).
%
% mu: k x D center locations
% idx: N-vector, index into 1:k specifying closest center for each pt
% mu_idx: k-vector, index into 1:N specifying which point selected to be
%   each center. mu is equal to x(mu_idx,:).
% Dall: k x n. Dall(k,n) is the distance from center k to pt n
% mindists: k x 1. mindists(k) is the smallest distance between center_k
%   and {center_1, center_2, ...center_(k-1)}.

function [mu,idx,mu_idx,Dall,mindists] = furthestfirst(x,k,varargin)

[n,d] = size(x);

[start,distance,weights,hWB] = myparse(varargin,...
  'Start','mean',...
  'Distance','sqEuclidean',...
  'weights',ones(n,1),...
  'hWaitBar',[]); % waitbar handle, or true
if isequal(hWB,true)
  hWB = waitbar(0);
  tfWBdel = true;
else
  tfWBdel = false;
end
tfWB = ishandle(hWB);

mu = zeros(k,d);
mu_idx = zeros(k,1);

% if the first center was not input, then choose a random point
if isnumeric(start) && ~isempty(start)
  mu(1,:) = start;
  mu_idx(1) = nan;
elseif strcmpi(start,'mean'),
  mu(1,:) = sum( x.*repmat(weights,[1,d]), 1 ) / sum(weights);
  mu_idx(1) = nan;
else
  i = randsample(n,1,true,weights);
  mu(1,:) = x(i,:);
  mu_idx(1) = i;
end

Dall = inf(k,n); % distances from kth center to nth pt
mindists = nan(k,1);
% running minimum distance across all previous centers to all n pts, ie
% equal to min(Dall(1:i-2,:),[],1)
Dmin = inf(1,n); 
if tfWB
  waitbar(0,hWB,'Computing shape distances');
end
for i = 2:k
  if tfWB
    waitbar(i/k,hWB);
  end
  
  % compute the minimum distance from all points to the centers
  Dall(i-1,:) = distfun(x,mu(i-1,:),distance);
  Dmin = min([Dmin; Dall(i-1,:)],[],1);

  % choose the point furthest from the centers as the next center
  if all(Dmin==0)
    % can happen if x contains repeated rows
    warning('furthestfirst:nochoice','All distances are zero for center %d/%d. Picking random unused point.',i,k);
    j = find(~ismember(1:n,mu_idx),1);
  else
    j = argmax(Dmin);
  end
  mu(i,:) = x(j,:);
  mu_idx(i) = j;
  mindists(i) = Dmin(j);
end
if tfWBdel
  delete(hWB);
end

% set the labels for each point
Dall(k,:) = distfun(x,mu(k,:),distance);
[~,idx] = min(Dall,[],1);
idx = idx(:);

%assert(isequaln(mu,x(mu_idx,:)));
assert(numel(mu_idx)==numel(unique(mu_idx)));

function D = distfun(X, C, dist)
%DISTFUN Calculate point to cluster centroid distances.
[n,p] = size(X);
nclust = size(C,1);
assert(size(C,2)==p);
D = zeros(n,nclust);

switch lower(dist)
case 'sqeuclidean'
    for i = 1:nclust
        D(:,i) = sum((X - C(repmat(i,n,1),:)).^2, 2);
    end
case 'nanmeansq'
    for i = 1:nclust
        D(:,i) = nanmean((X - C(repmat(i,n,1),:)).^2, 2);
    end
case 'cityblock'
    for i = 1:nclust
        D(:,i) = sum(abs(X - C(repmat(i,n,1),:)), 2);
    end
case {'cosine','correlation'}
    % The points are normalized, centroids are not, so normalize
    % them
    normC = sqrt(sum(C.^2, 2));
    if any(normC < eps(class(normC))) % small relative to
                                      % unit-length data points
        error('furthestfirst:ZeroCentroid', ...
              'Zero cluster centroid created.');
    end
    % This can be done without a loop, but the loop saves memory
    % allocations
    for i = 1:nclust
        D(:,i) = 1 - (X * C(i,:)') ./ normC(i);
    end
case 'hamming'
    for i = 1:nclust
        D(:,i) = sum(abs(X - C(repmat(i,n,1),:)), 2) / p;
    end
end

