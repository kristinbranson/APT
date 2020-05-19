function [use,ftrsSel] = selectCorrFeat(S,pTar,ftrs,ftrPrm,stdFtrs,dfFtrs)
% Selection of features based on their correlation, as proposed
% in "Face Alignment by Explicit Shape Regression", Cao et al, CVPR12.
%
% USAGE
%  [use,ftrsSel] = selectCorrFeat(S,pTar,ftrs,ftrPrm,stdFtrs,dfFtrs)
%
% INPUTS
%  S        - scalar int, number of features to select
%  pTar     - [NxD] target output values
%  ftrs     - [NxF] computed features
%  ftrPrm   - type of features, shapeGt>ftrsGen struct
%     .nsample_cor
%     .metatype
%  stdFtrs  - For ftrPrm.metatype=='single': [1xF] matrix with precomputed std for each feature
%           - For ftrPrm.metatype=='diff': [FxF] matrix (upper tri), std-of-differences, see stdFtrs1
%  dfFtrs   - [NxF] de-meaned features
%
% OUTPUTS
%  use      - Features indices (indices into cols of ftrs) for selected features.
%             for ftrPrm.metatype=='single': [S]. for .metatype='diff': [2xS].
%  ftrsSel  - [NxS] Selected (meta)feature values. 
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

% Modified by Allen Lee, Kristin Branson

[N,D] = size(pTar);
F = size(ftrs,2);

% random projection of data
b = rand(D,S)*2-1; 
b = bsxfun(@rdivide,b,sqrt(sum(b.^2,1))); % [DxS], each col is a D-dim unit vec


if isfield(ftrPrm,'corsamples') && ~isempty(ftrPrm.corsamples),
  nsample = ftrPrm.corsamples(2)-ftrPrm.corsamples(1)+1;
else
  assert(isfield(ftrPrm,'nsample_cor'));
  nsample = ftrPrm.nsample_cor;
end

if nsample < N
  if isfield(ftrPrm,'corsamples') && ~isempty(ftrPrm.corsamples),
    dosample = ftrPrm.corsamples(1):ftrPrm.corsamples(2);
  else
    dosample = rand(N,1) <= nsample/N;
  end
else
  dosample = true(N,1);
end

pTarSamp = pTar(dosample,:);
pTarSampMu = nanmean(pTarSamp,1); % centroid of each coordinate
assert(~any(isnan(pTarSampMu))); % nan would show up only if entire column is NaN

% We treat NaNs in pTar as missing data. For the purposes of feature
% selection, replace missing (NaN) values with the mean centroid for that
% coordinate, the idea being that the absence of a coordinate should not
% bias the 1D projections of pTar (see variable 'scalar' below).
tf = isnan(pTarSamp);
nnztf = nnz(tf);
if nnztf>0
  warningNoTrace('selectCorrFeat:nantar','%d/%d NaNs found in pTarSamp. Replacing with column means.',...
    nnztf,numel(tf));
  [~,jCol] = find(tf);
  iLin = find(tf(:));
  assert(numel(iLin)==numel(jCol));
  pTarSamp(iLin) = pTarSampMu(jCol);
end

Bsamp = pTarSamp*b; % [numel(dosample)xS], projections of pTar(dosample,:) onto S unit vecs
assert(nnz(isnan(Bsamp))==0);
BsampSD = std(Bsamp,[],1);
BsampMu = mean(Bsamp,1);

% AL20160310. It's a little weird that we are using ftrs(dosample,:) and
% dfFtrs(dosample,:) with stdFtrs. stdFtrs is the SD computed by sampling 
% all features, not just the ones included in dosample. So the 
% covariances-with-B that are calculated will not be quite right. 
% Hopefully if nsample_std and numel(dosample) are largish then should 
% basically be ok.


switch ftrPrm.metatype
  case 'single'
    dB = bsxfun(@minus,Bsamp,BsampMu);
    assert(isequal(size(stdFtrs),[1 F]));
    % Try a sanity check on stdFtrs vs std(ftrs(dosample,:))
    if false
      stdFtrsSamp1 = std(ftrs(dosample,1),[],1);
      fprintf(1,'stdFtrs sanity. stdFtrs: %.3g. stdFtrsSamp: %.3g\n',...
        stdFtrs(1),stdFtrsSamp1);
    end
    use = selectFeatSingle(dfFtrs(dosample,:),stdFtrs,dB,BsampSD);
    
    assert(numel(use)==S);
    use = use(:)';
    ftrsSel = ftrs(:,use);
    
    % added by KB 20181109 to try to get range to be similar to what you
    % would get in diff mode
    warningNoTrace('Feature computation for metatype=single has been changed so that the range of features is -1 to 1. If you trained pre-20181109, your model may not work.');
    ftrsSel = (ftrsSel-.5)*2;
    
  case 'diff'
    assert(isequal(size(stdFtrs),[F F]));    
    SELECTCORRFEATTYPE = 2;
    % - pTarSamp is [nsampxD], not used except for size
    % - ftrs is [nsampxF]
    % - dfFtrs is [nsampxF]
    % - Bsamp is [nsampxS]    
    use = selectCorrFeatAL(pTarSamp,ftrs(dosample,:),SELECTCORRFEATTYPE,stdFtrs,...
      dfFtrs(dosample,:),Bsamp,BsampSD,BsampMu);
    
    assert(isequal(size(use),[2 S]));
    iF1 = use(1,:);
    iF2 = use(2,:);
    ftrsSel = ftrs(:,iF1) - ftrs(:,iF2);
    
    %[use0,maxCo0] = selectCorrFeat1(args{:});
    % fprintf('### selectCorrFeat comparison:\n');
    % fprintf(' Old:\n');
    % disp(num2str(use0));
    % disp(num2str(maxCo0,3));
    % fprintf(' New:\n');
    % disp(num2str(use));
    % disp(num2str(maxCo,3));  
    
  otherwise
    assert(false,'Unknown ftrPrm.metatype.');
end

assert(~any(use(:)==0)); % AL20160310
  
% if any(use(:)==0)
%   %This can happen when selectCorrFeat1 does not find enough unique
%   %combinations due to low number of features (e.g. when nzones==1)
%   ind=find(use==0); F=size(ftrs,2); I=length(ind);
%   if(F>=I), use(ind)=randSample(F,I);
%   else use(ind)=randint2(1,I,[1 F]);
%   end
% end
