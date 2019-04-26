function [ferns,hsPr] = fernsClfChangeNFerns( data, hs, ferns, Mnew, varargin )
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
%   .ferns    - [] if given reuse previous ferns (recompute pFern)
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

dfs={'thrr',[0 1],'tmp',''};
[thrr,~]=getPrmDflt(varargin,dfs,1);

[Mold,S] = size(ferns.fids);
[N,F]=size(data); assert(length(hs)==N);
H=max(hs); assert(all(hs>0)); assert(S<=20);

if Mnew == Mold,
elseif Mnew < Mold,
  Mremove = Mold - Mnew;
  % randomly choose some ferns to remove
  idx_remove = randsample(Mold,Mremove);
  ferns.fids(idx_remove,:) = [];
  ferns.thrs(idx_remove,:) = [];
  ferns.counts(:,:,idx_remove) = [];
  ferns.pFern(:,:,idx_remove) = [];
  ferns.inds(:,idx_remove) = [];
else
  
  Madd = Mnew - Mold;
  % create some new ferns
  thrs_add=rand(Madd,S)*(thrr(2)-thrr(1))+thrr(1);
  fids_add=uint32(floor(rand(Madd,S)*F+1)); 
  inds_add=fernsInds(data,fids_add,thrs_add);

  % store new ferns
  ferns.fids(Mold+1:Mnew,:) = fids_add;
  ferns.thrs(Mold+1:Mnew,:) = thrs_add;
  ferns.inds(:,Mold+1:Mnew) = inds_add;
  
  % get counts for each leaf for each class for each new fern
  pFern_add = nan(2^S,H,Madd);
  edges = 1:2^S;
  for m = 1:Madd,
    for h = 1:H,
      pFern_add(:,h,m) = histc(inds_add(hs==h,m),edges);
    end
  end
  pFern_add = pFern_add + ferns.bayes;
  ferns.counts(:,:,Mold+1:Mnew) = pFern_add;

  % convert fern leaf class counts into probabilities
  if( ferns.bayes<=0 )
    norm = 1./sum(pFern_add,2);
    pFern_add = bsxfun(@times,pFern_add,norm);
  else
    norm = 1./sum(pFern_add,1);
    pFern_add = bsxfun(@times,pFern_add,norm);
    pFern_add=log(pFern_add);
  end
  ferns.pFern(:,:,Mold+1:Mnew) = pFern_add;

  clear pFern;
end

if(nargout==2),
  hsPr=fernsClfApply([],ferns,ferns.inds); 
end

end
