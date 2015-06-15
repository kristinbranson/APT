function [use,ftrs1] = selectCorrFeat(S,pTar,ftrs,ftrPrm,...
    stdFtrs,dfFtrs)
% Selection of features based on their correlation, as proposed
% in "Face Alignment by Explicit Shape Regression", Cao et al, CVPR12.
%
% USAGE
%  [use,ftrs1] = selectCorrFeat(S,pTar,ftrs,ftrPrm,...
%    stdFtrs,dfFtrs)
%
% INPUTS
%  S        - Number of features to select
%  pTar     - [NxD] target output values
%  ftrs     - [NxF] computed features
%  ftrPrm   - type of features, shapeGt>ftrsGen struct
%  stdPrm   - [1xF] matrix with precomputed std for each feature
%  dfFtrs   - [NxF] differences between pair of features
%  class    - [1xF] feature class (if any)
%
% OUTPUTS
%  use    - [type x S] feature Ids
%  ftrs1  - [NxS] S selected features features
%
% See also 
%    regTrain
%
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia


[N,D]=size(pTar); 
% random projection of data
b=rand(D,S)*2-1; 
b = bsxfun(@rdivide,b,sqrt(sum(b.^2,1)));

if isfield(ftrPrm,'nsample_cor'),
  nsample = ftrPrm.nsample_cor;
else
  nsample = N;
end
if nsample < N,
  dosample = rand(N,1) <= nsample/N;
else
  dosample = true(N,1);
end

scalar = pTar(dosample,:)*b; 
%b=rand(S,D)*2-1; 
%for s=1:S, b(s,:)=b(s,:)./norm(b(s,:)); end
%scalar = pTar*b';

stdSc = std(scalar); muSc=mean(scalar);
% I think it is a bug that use is 4 x S, changed this to have a max value
type=ftrPrm.type;if(type>2),type=min(type-2,2);end
% type=ftrPrm.type;if(type>2),type=type-2;end
[use] = selectCorrFeat1(pTar(dosample,:),ftrs(dosample,:),type,...
    stdFtrs,dfFtrs(dosample,:),scalar,stdSc,muSc);
if(any(use(:)==0)) 
    %This can happen when selectCorrFeat1 does not find enough unique
    %combinations due to low number of features (e.g. when nzones==1)
    ind=find(use==0); F=size(ftrs,2); I=length(ind);
    if(F>=I), use(ind)=randSample(F,I);
    else use(ind)=randint2(1,I,[1 F]);
    end
end
clear scalar;
if(nargout>1)
    if(type==1), ftrs1 = ftrs(:,use);
    else ftrs1 = ftrs(:,use(1,:))-ftrs(:,use(2,:));
    end
end
end