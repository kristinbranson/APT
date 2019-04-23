% KMEANS K-means clustering.
% IDX = KMEANS(X, K) partitions the points in the N-by-P data matrix
% X into K clusters.  This partition minimizes the sum, over all
% clusters, of the within-cluster sums of point-to-cluster-centroid
% distances.  Rows of X correspond to points, columns correspond to
% variables.  KMEANS returns an N-by-1 vector IDX containing the
% cluster indices of each point.  By default, KMEANS uses squared
% Euclidean distances.
%
% KMEANS treats NaNs as missing data, and removes any rows of X that
% contain NaNs. 
%
% [IDX, C] = KMEANS(X, K) returns the K cluster centroid locations in
% the K-by-P matrix C.
% 
% [IDX, C, SUMD] = KMEANS(X, K) returns the within-cluster sums of
% point-to-centroid distances in the 1-by-K vector sumD.
% 
% [IDX, C, SUMD, D] = KMEANS(X, K) returns distances from each point
% to every centroid in the N-by-K matrix D.
% 
% [ ... ] = KMEANS(..., 'PARAM1',val1, 'PARAM2',val2, ...) allows you to
% specify optional parameter name/value pairs to control the iterative
% algorithm used by KMEANS.  Parameters are:
%
% 'Start': Method used to choose initial cluster centroid positions,
%  sometimes known as "seeds".  Choices are:
%   'furthestfirst' - [default] Use the furthest-first algorithm to initialize
%    k-means. If there is more than one restart, then furthest-first will be
%    initialized randomly. Otherwise, it will be initialized with the sample
%    mean.
%   'sample' - Select K observations from X at random
%   'uniform' - Select K points uniformly at random from the range of X.
%   'cluster' - Perform preliminary clustering phase on random 10% subsample of
%    X.  This preliminary phase is itself initialized using 'sample'.
%   matrix - A K-by-P matrix of starting locations. In this case, you can pass
%    in [] for K, and KMEANS infers K from the first dimension of the matrix.
%    You can also supply a 3D array, implying a value for 'Replicates' from the
%    array's third dimension.
% 
% 'Replicates' - Number of times to repeat the clustering, each with a new set
%  of initial centroids [ positive integer | {1}]
%
% 'Maxiter' - The maximum number of iterations [ positive integer | {100}]
%
% 'EmptyAction' - Action to take if a cluster loses all of its member
%  observations.  Choices are:
%   'error' - [default] Treat an empty cluster as an error
%   'drop' - Remove any clusters that become empty, and set corresponding values
%   in C and D to NaN.
%   'singleton' - Create a new cluster consisting of the one observation
%   furthest from its centroid.
%
% 'Display' - Display level [ 'off' | {'notify'} | 'final' | 'iter' ]
%
function [idx,C,sumD,D] = mykmeans(X, k, varargin)

[N,P] = size(X);
[start,distance,replicates,maxiter,ff_start,emptyaction,display,weights] = ...
    myparse(varargin,'Start','furthestfirst',...
            'Distance','sqEuclidean',...
            'Replicates',1,'Maxiter',100,...
            'furthestfirst_start','sample',...
            'EmptyAction','singleton','Display','off',...
            'weights',ones(N,1));
isweight = ~all(weights==weights(1));
          
if strcmp(start,'furthestfirst'),

  start = zeros([k,P,replicates]);
  if strcmpi(ff_start,'sample')
    for rep = 1:replicates,
      start(:,:,rep) = furthestfirst(X, k, 'Distance', distance,...
                                     'Start','sample','weights',weights);
    end;
  else
    replicates = 1;
    start = furthestfirst(X,k,'Distance',distance,...
                          'Start','mean','weights',weights);
  end;

  if isweight,
    [idx,C,sumD,D] = weightedkmeans(X,k,'Start',start,'Distance',distance,...
      'Replicates',replicates,'Maxiter',maxiter,...
      'EmptyAction',emptyaction,'Display',display,'weights',weights);
  else
    [idx,C,sumD,D] = matlabkmeans(X,k,'Start',start,'Distance',distance,...
      'Replicates',replicates,'Maxiter',maxiter,...
      'EmptyAction',emptyaction,'Display',display);
  end
else

  if isweight,
    [idx,C,sumD,D] = weightedkmeans(X,k,'Start',start,'Distance',distance,...
      'Replicates',replicates,'Maxiter',maxiter,...
      'EmptyAction',emptyaction,'Display',display,'weights',weights);
  else
    [idx,C,sumD,D] = matlabkmeans(X,k,'Start',start,'Distance',distance,...
      'Replicates',replicates,'Maxiter',maxiter,...
      'EmptyAction',emptyaction,'Display',display);
  end
end