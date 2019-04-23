function [ferns,hsPr] = fernsClfReplaceFeature( data, hs, ferns, features_remove, varargin )
% Train random fern classifier.
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
%
% OUTPUTS
%  ferns    - learned fern model w the following fields
%   .fids     - [MxS] feature ids for each fern for each depth
%   .thrs     - [MxS] threshold corresponding to each fid
%   .pFern    - [2^SxHxM] learned log probs at fern leaves
%   .bayes    - if true combine probs using bayes assumption
%   .inds     - [NxM] cached indices for original training data
%   .H        - number classes
%  hsPr     - [Nx1] predicted output labels
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
% See also fernsClfApply, fernsInds
%
% Piotr's Image&Video Toolbox      Version 2.50
% Copyright 2010 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Lesser GPL [see external/lgpl.txt]

% get additional parameters and check dimensions
dfs={'features_replace',[],'doreplace',[],'fids_replace',[]};
[features_replace,doreplace,fids_replace]=getPrmDflt(varargin,dfs,1);
[N,F]=size(data); assert(length(hs)==N);
M=size(ferns.inds,2); S=log2(size(ferns.pFern,1));
H=max(hs); assert(all(hs>0)); assert(S<=20);
assert(size(ferns.inds,1)==N);

% indices we want to remove
if isempty(doreplace),
  doreplace = ismember(ferns.fids,features_remove);
else
  assert(numel(doreplace)==M*S);
end
nreplace = nnz(doreplace);

% features to replace
if isempty(fids_replace),
  % allowed features to replace with
  if isempty(features_replace),
    features_replace = uint32(setdiff(1:F,features_remove));
  else
    features_replace = uint32(features_replace);
  end
  fids_replace = features_replace(floor(rand(nreplace,1)*numel(features_replace)+1));
else
  assert(numel(fids_replace)==nreplace);
end

% replace features in fids
ferns.fids(doreplace) = fids_replace;

% update inds to reflect these replacements
ferns.inds = fernsInds_update(data,ferns.fids,ferns.thrs,ferns.inds,doreplace);

% get counts for each leaf for each class for each fern
% KB: use hist to avoid inner loop
pFern = nan(2^S,H,M);
edges = 1:2^S;
for m = 1:M,
  for h = 1:H,
    pFern(:,h,m) = histc(ferns.inds(hs==h,m),edges);
  end
end
pFern = pFern + ferns.bayes;

% KB: store the unnormalized counts
ferns.counts = pFern;

% convert fern leaf class counts into probabilities
if( ferns.bayes<=0 )
  pFern = bsxfun(@rdivide,pFern,sum(pFern,2));
else
  pFern = bsxfun(@rdivide,pFern,sum(pFern,1));
  pFern=log(pFern);
end

% store pFern and compute output values
ferns.pFern=pFern; clear pFern;
if(nargout==2), hsPr=fernsClfApply([],ferns,ferns.inds); end

end
