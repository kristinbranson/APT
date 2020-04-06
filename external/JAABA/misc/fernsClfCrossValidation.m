function [fracwrong_cv,fracwrong,hsPr_cv,probs_cv,ferns,hsPr,probs] = fernsClfCrossValidation( data, hs, varargin )
% Compute cross-validation error of random ferns classifier
%
% See "Fast Keypoint Recognition in Ten Lines of Code" by Mustafa Ozuysal,
% Pascal Fua and Vincent Lepetit, CVPR07.
%
% Dimensions:
%  M - number ferns
%  S - fern depth
%  F - number features
%  N - number input vectors
%  H - number classes
%
% USAGE
%  [ferns,hsPr] = fernsClfTrain( data, hs, [varargin] )
%
% INPUTS
%  data     - [NxF] N length F feature vectors
%  hs       - [Nx1] target output labels in [1,H]
%  varargin - additional params (struct or name/value pairs)
%   .S        - [10] fern depth (ferns are exponential in S)
%   .M        - [50] number of ferns to train
%   .thrr     - [0 1] range for randomly generated thresholds
%   .bayes    - [1] if true combine probs using bayes assumption
%   .ferns    - [] if given reuse previous ferns (recompute pFern)
%   .cv_sets  - [] if given, N x 1 vector containing the index of each 
%   .cv_nsets - [10] if cv_sets not specified, then the data is split into
%               cv_nsets randomly chosen sets. 
%
% OUTPUTS
%  fracwrong_cv  - [1x1] fraction of training examples predicted 
%               incorrectly in hsPr_cv
%  hsPr_cv    - [Nx1] predicted cross-validation-based classification of
%               each training example
%  probs_cv   - [NxH] predicted cross-validation-based output label
%               probabilities for each training example
%  ferns      - learned fern model from all training data with the
%               following fields 
%   .fids     - [MxS] feature ids for each fern for each depth
%   .thrs     - [MxS] threshold corresponding to each fid
%   .pFern    - [2^SxHxM] learned log probs at fern leaves
%   .bayes    - if true combine probs using bayes assumption
%   .inds     - [NxM] cached indices for original training data
%   .H        - number classes
%
% EXAMPLE
%  N=5000; H=5; d=2; [xs0,hs0,xs1,hs1]=demoGenData(N,N,H,d,1,1);
%  fernPrm=struct('S',4,'M',50,'thrr',[-1 1],'bayes',1);
%  tic, [ferns,hsPr0]=fernsClfTrain(xs0,hs0,fernPrm); toc
%  tic, hsPr1 = fernsClfApply( xs1, ferns ); toc
%  e0=mean(hsPr0~=hs0); e1=mean(hsPr1~=hs1);
%  fprintf('errors trn=%f tst=%f\n',e0,e1); figure(1);
%  subplot(2,2,1); visualizeData(xs0,2,hs0);
%  subplot(2,2,2); visualizeData(xs0,2,hsPr0);
%  subplot(2,2,3); visualizeData(xs1,2,hs1);
%  subplot(2,2,4); visualizeData(xs1,2,hsPr1);
%
% See also fernsClfTrain, fernsClfApply, fernsInds
%
% Kristin Branson 2011-06-09, based on fernsClfTrain from:
%
% Piotr's Image&Video Toolbox      Version 2.50
% Copyright 2010 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Lesser GPL [see external/lgpl.txt]

% get additional parameters and check dimensions
dfs={'S',10,'M',50,'thrr',[0 1],'bayes',1,'ferns',[],'cv_sets',[],'cv_nsets',10};
[S,M,thrr,bayes,ferns,cv_sets,cv_nsets]=getPrmDflt(varargin,dfs,1);
[N,F]=size(data); assert(length(hs)==N);
H=max(hs); assert(all(hs>0)); assert(S<=20);
edges = 1:2^S;

if( isempty(ferns) )
  % create ferns model and compute inds
  thrs=rand(M,S)*(thrr(2)-thrr(1))+thrr(1);
  fids=uint32(floor(rand(M,S)*F+1)); 
  inds=fernsInds(data,fids,thrs);
  ferns=struct('fids',fids,'thrs',thrs,'bayes',bayes,'H',H,'inds',inds);
  % compute unnormalized counts
  % KB: use hist to avoid inner loop
  counts = nan(2^S,H,M);
  for m = 1:M,
    for h = 1:H,
      counts(:,h,m) = histc(inds(hs==h,m),edges);
    end
  end
  counts = counts + bayes;
  % old code
  %   counts = bayes(ones(2^S,H,M));
  %   for n=1:N, h=hs(n);
  %     for m=1:M, ind=inds(n,m);
  %       counts(ind,h,m)=counts(ind,h,m)+1;
  %     end
  %   end
  ferns.counts = counts;
  % convert fern leaf class counts into probabilities
  if( bayes<=0 )
    norm = 1./sum(counts,2);
    pFern = bsxfun(@times,counts,norm);
  else
    norm = 1./sum(counts,1);
    pFern = bsxfun(@times,counts,norm);
    pFern=log(pFern);
  end

  % store pFern and compute output values
  ferns.pFern=pFern; 

else
  % re-use cached model, trained with all the training data in data, hs
  ferns.H=H; 
  inds=ferns.inds; 
  assert(size(inds,1)==N);
  counts = ferns.counts;
end

% randomly choose cv_sets if not input
if isempty(cv_sets),
  if cv_nsets >= N,
    cv_nsets = N;
    cv_sets = 1:N;
  else
    order = randperm(N);
    brks = round(linspace(1,N+1,cv_nsets+1));
    cv_sets = zeros(1,N);
    cv_sets(brks(1:end-1)) = 1;
    cv_sets = cumsum(cv_sets);
    cv_sets = cv_sets(order);
  end
else
  cv_sets = cv_sets(:);
  cv_nsets = max(cv_sets);
end
if cv_nsets <= 1,
  error('cv_nsets must be an integer > 1');
end

hsPr_cv = nan(N,1);
probs_cv = nan(N,H);
ferns_curr = ferns;
for seti = 1:cv_nsets,
  
  % delete this set from counts
  idx_curr = cv_sets==seti;
  if ~any(idx_curr),
    continue;
  end
  counts_curr = counts;
  for m = 1:M,
    for h = 1:H,
      counts_curr(:,h,m) = counts_curr(:,h,m) - histc(inds(idx_curr&hs==h,m),edges);
    end
  end
  
  % convert fern leaf class counts into probabilities
  if( bayes<=0 )
    norm_curr = 1./sum(counts_curr,2);
    pFern_curr = bsxfun(@times,counts_curr,norm_curr);
  else
    norm_curr = 1./sum(counts_curr,1);
    pFern_curr = bsxfun(@times,counts_curr,norm_curr);
    pFern_curr=log(pFern_curr);
  end
  ferns_curr.pFern = pFern_curr;
  ferns_curr.counts = counts_curr;
  [hsPr_cv(idx_curr),probs_cv(idx_curr,:)] = ...
    fernsClfApply([],ferns_curr,inds(idx_curr,:));
end

nwrong_cv = nnz(hsPr_cv(:) ~= hs(:));
fracwrong_cv = nwrong_cv / N;

[hsPr,probs] = fernsClfApply([],ferns,inds);
fracwrong = nnz(hsPr(:) ~= hs(:)) / N;


end
