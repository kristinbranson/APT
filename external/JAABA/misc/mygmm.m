% [mu,S,priors,post,nll,mix] = mygmm(X,k,...)
% Input:
% X: N x D matrix of N D-dimensional data points
% k: number of clusters
% optional:
% 'Replicates' = 1
% 'CovarType' = 'full'
% 'display' = -1
% 'precision' = .0001
% 'ResetCovar' = 1
% 'MaxIters' = 100
% 'Start' = 'furthestfirst'
% 'distance' = 'sqEuclidean'
% 'weights' = []
function [mu,S,priors,post,nll,mixsave] = mygmm(X,k,varargin)

[nreplicates,covartype,display,precision,resetcovar,maxiters,start,distance,weights] = ...
    myparse(varargin,'Replicates',1,'CovarType','full',...
            'display',-1,'precision',.0001,'ResetCovar',1,'MaxIters',100,...
            'Start','furthestfirst','Distance','sqEuclidean','weights',[]);
kmeansparams = {'Replicates',1,'Start',start,'Distance',distance,...
                'EmptyAction','singleton','Display','off','furthestfirst_start'};
if nreplicates == 1,
  kmeansparams{end+1} = 'mean';
else
  kmeansparams{end+1} = 'sample';
end
              
[N,D] = size(X);
if isstruct(start),
  nreplicates = length(start);
elseif isnumeric(start),
  nreplicates = size(start,3);
end

options = zeros(1,14);
options(1) = display;
options(3) = precision;
options(5) = resetcovar;
options(14) = maxiters;

minerr = inf;
for replicate = 1:nreplicates,

  % initialize
  if isstruct(start),
    mix = start(replicate);
  elseif isnumeric(start),
    % start is just the mean
    mu = start(:,:,replicate);
    D = dist2(mu,X);
    [~,idx] = min(D,[],1);
    % create gmm structure
    mix = mygmminit(X,mu,idx,covartype);
  else
    [idx,C] = mykmeans(X,k,kmeansparams{:});
    for i = 1:k,
      if ~any(idx==k),
        keyboard;
      end
    end
    % create gmm structure
    mix = mygmminit(X,C,idx,covartype);
  end

  if replicate == 1,
    mixsave = mix;
  end
  
  if isempty(weights),
    [mix,options,errlog,post] = gmmem(mix,X,options);
  else
    [mix,options,errlog,post] = gmmem_weighted(mix,X,weights,options);
  end
  nllcurr = options(8);
  if nllcurr < minerr,
    minerr = nllcurr;
    mixsave = mix;
  end;

end;

mu = mixsave.centres;
S = mixsave.covars;
priors = mixsave.priors;
nll = minerr;