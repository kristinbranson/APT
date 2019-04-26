function [ferns,hsPr] = fernsClfRemoveTrainingData( hs, remove_data_idx, ferns )
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
%dfs={'S',10,'M',50,'thrr',[0 1],'bayes',1,'ferns',[],'cv_sets',[],'cv_nsets',10};
%[S,M,thrr,bayes,ferns,cv_sets,cv_nsets]=getPrmDflt(varargin,dfs,1);
H=max(hs);
[M,S] = size(ferns.fids);
assert(all(hs>0)); assert(S<=20);

% grab inds for data we are removing
inds_remove = ferns.inds(remove_data_idx,:);

% remove inds from ferns
ferns.inds(remove_data_idx,:) = [];

% update counts
pFern = ferns.counts;
edges = 1:2^S;
for m = 1:M,
  for h = 1:H,
    pFern(:,h,m) = pFern(:,h,m) - reshape(histc(inds_remove(hs(remove_data_idx)==h,m),edges),[2^S,1]);
  end
end
ferns.counts = pFern;

% update pFern
if( ferns.bayes<=0 )
  norm = 1./sum(pFern,2);
  % KB: use bsxfun instead of loop
  pFern = bsxfun(@times,pFern,norm);
  %for h=1:H, pFern(:,h,:)=pFern(:,h,:).*norm; end
else
  norm = 1./sum(pFern,1);
  % KB: use bsxfun instead of loop
  pFern = bsxfun(@times,pFern,norm);
  %for s=1:2^S, pFern(s,:,:)=pFern(s,:,:).*norm; end
  pFern=log(pFern);
end

% store pFern and compute output values
ferns.pFern=pFern; clear pFern;
if(nargout==2), hsPr=fernsClfApply([],ferns,ferns.inds); end

end

