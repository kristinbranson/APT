function [ferns,hsPr] = fernsClfChangeFernDepth( data, hs, ferns, Snew, varargin )
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

% get additional parameters and check dimensions
dfs={'thrr',[0 1],'tmp',''};
[thrr,~]=getPrmDflt(varargin,dfs,1);

[M,Sold] = size(ferns.fids);
[N,F]=size(data); assert(length(hs)==N);
H=max(hs); assert(all(hs>0)); assert(Snew<=20);

if Snew == Sold,
elseif Snew < Sold,
  Sremove = Sold - Snew;
  
  % remove classifiers from ferns
  ferns.thrs(:,Snew+1:Sold) = [];
  ferns.fids(:,Snew+1:Sold) = [];
  
  % remove corresponding bits from inds
  ferns.inds = bitshift(ferns.inds,-Sremove);
  
  % accumulate counts into parent nodes
  ferns.counts = reshape(sum(reshape(ferns.counts,[2^Snew,2^Sremove,H,M]),2),[2^Snew,H,M]);
  
  % convert fern leaf class counts into probabilities
  if( ferns.bayes<=0 )
    norm = 1./sum(ferns.counts,2);
    ferns.pFern = bsxfun(@times,ferns.counts,norm);
  else
    norm = 1./sum(ferns.counts,1);
    ferns.pFern = bsxfun(@times,ferns.counts,norm);
    ferns.pFern=log(ferns.pFern);
  end
else
  
  Sadd = Snew - Sold;
  
  % generate some new classifiers for each fern
  ferns.thrs(:,Sold+1:Snew) = rand(M,Sadd)*(thrr(2)-thrr(1))+thrr(1);
  ferns.fids(:,Sold+1:Snew) = uint32(floor(rand(M,Sadd)*F+1)); 
  
  % new lower bits to inds
  indsadd=fernsInds(data,ferns.fids(:,Sold+1:Snew),ferns.thrs(:,Sold+1:Snew));
  ferns.inds = bitshift(ferns.inds,Sadd) + indsadd;
  
  % re-histogram
  ferns.counts = zeros(2^Snew,H,M); 
  edges = 1:2^Snew;
  for h=1:H, 
    inds1=ferns.inds(hs==h,:);
    for m=1:M, 
      ferns.counts(:,h,m)=histc(inds1(:,m),edges); 
    end
  end
  ferns.counts = ferns.counts + ferns.bayes;

  % convert fern leaf class counts into probabilities
  if( ferns.bayes<=0 )
    norm = 1./sum(ferns.counts,2);
    ferns.pFern = bsxfun(@times,ferns.counts,norm);
  else
    norm = 1./sum(ferns.counts,1);
    ferns.pFern = bsxfun(@times,ferns.counts,norm);
    ferns.pFern=log(ferns.pFern);
  end
end

if(nargout==2), hsPr=fernsClfApply([],ferns,inds); end

end
