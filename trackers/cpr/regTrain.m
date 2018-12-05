function [regInfo,ysPr,timingInfo] = regTrain(data,ys,varargin)
% Train boosted regressor.
%
% USAGE
%  [regInfo,ysPr] = regTrain( data, ys, [varargin] )
%
% INPUTS
%  data     - [NxF] N length F feature vectors
%  ys       - [NxD] target output values
%  varargin - additional params (struct or name/value pairs)
%   .type     - [1] type of regression
%                   1=fern, 2=linear
%   .ftrPrm   - [REQ] prm struct (see shapeGt>ftrsGen)
%   .K        - [1] number of boosted regressors
%   .M        - [5] number of features used by each regressor
%   .R        - [0] number repetitions per fern (if =0 uses correlation
%                       selection instead of random optimization)
%   .loss     - ['L2'] loss function (used if R>0) for
%                      random step optimization
%                       options include {'L1','L2'}
%   .model    - [] optional, if special treatment is required for regression
%   .prm      - [REQ] regression parameters, relative to type
%   .occlD    - feature occlusion info, see shapeGt>ftrsCompDup
%   .occlPrm  - occlusion params for occlusion-centered (struct)
%                regression, output of shapeGt>ftrsComp
%       .nrows     - [3] number of rows into which divide face
%       .ncols     - [3] number of cols into which divide face
%       .nzones    - [1] number of face zone from which regressors draw features
%       .Stot      - [3] number of regressors to train at each round
%       .th        - [.5] occlusion threshold
%
% OUTPUTS
%  regInfo    - [K x Stot] cell with learnt regressors models
%     .ysFern  - [2^MxR] fern bin averages
%     .thrs    - [Mx1] thresholds
%     .fids    - [2xS] features used
%  ysPr       - [NxD] predicted output values
%
% See also
%           rcprTrain, regTrain>trainFern, regTrain>trainLin, regApply
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

% KB: added doshuffle = true, whether to shuffle the data for subsampling
% purposes. If data is already in an order such that we want to subsample
% contiguous chunks, then set doshuffle = false
% KB: added dosubsample = false, whether to train each fern on a subset of
% the data or not. This should probably be dependent on the number of
% training examples

starttime = tic;

dfs = {'type',1,'ftrPrm','REQ','K',1,...
  'loss','L2','R',0,'M',5,'model',[],'prm',{},...
  'occlD',[],'occlPrm',struct('Stot',1),'checkPath',false,...
  'doshuffle',true};
[regType,ftrPrm,K,loss,R,M,model,regPrm,occlD,occlPrm,checkPath,doshuffle] = ...
  getPrmDflt(varargin,dfs,0);
timingInfo = struct;
timingInfo.init = 0;
timingInfo.featureStat = nan;
timingInfo.selectFeatures = 0;
timingInfo.regress = 0;
timingInfo.iter = nan(K,1);

switch regType
  case 1, regFun = @trainFern;
  case 2, regFun = @trainLin;
  otherwise, error('unknown regressor type');
end

assert(any(strcmp(loss,{'L1','L2'})));

% KB 20180420: reorder data randomly so that we can sample by selecting
% from the top
N = size(data,1);
if doshuffle,
  dataorder = randperm(N);
  [~,datareorder] = sort(dataorder);
  data = data(dataorder,:);
  ys = ys(dataorder,:);
else
  datareorder = 1:N;
end

% KB 20180421: use intervals of samples for speed
[corsamplestarts,corsampleends] = SelectWrappingSampleSubsets(K,N,ftrPrm.nsample_cor);

timingInfo.init = toc(starttime);
inittime = tic;

% precompute feature stats to be used by selectCorrFeat
if R==0
  if checkPath
    sfPath = fileparts(which('SelectFeatures'));
    [~,sfBase] = fileparts(sfPath);
    if strcmpi(sfBase,'perframe')
      error('regTrain:path',...
        'Class definition ''SelectFeatures.m'' appears to be shadowed by JAABA. Please check your path.');
    end
  end
  [stdFtrs,dfFtrs] = SelectFeatures.statsFtrs(data,ftrPrm);
else %random step optimization selection
  assert(false,'AL unused');
  switch(loss)
    case 'L1',  lossFun=@(ys,ysGt) mean(abs(ys(:)-ysGt(:)));
    case 'L2',  lossFun=@(ys,ysGt) mean((ys(:)-ysGt(:)).^2);
  end
end
timingInfo.featureStat = toc(inittime);
featurestattime = tic;

Stot = occlPrm.Stot;
[N,D] = size(ys);
ysSum = zeros(N,D);
regInfo = cell(K,Stot);

%If using occlusion-centered approach, set up masks
if Stot>1 && ~isempty(occlD)
  assert(false,'AL');
  
  nGroups=occlPrm.nrows*occlPrm.ncols;
  masks=zeros(Stot,min(nGroups,occlPrm.nzones));
  for s=1:Stot
    masks(s,:)=randSample(nGroups,min(nGroups,occlPrm.nzones));
  end
  if(D>10),mg=median(occlD.group);
  else mg=occlD.group;
  end
  ftrsOccl=zeros(N,K,Stot);
end
timingInfo.init = timingInfo.init + toc(featurestattime);

%Iterate through K boosted regressors
for k=1:K
  
  iterkstarttime = tic;
  
  %Update regression target
  ysTar = ys-ysSum;
  %Train Stot different regressors
  ysPred = zeros(N,D,Stot);
  for s=1:Stot
    
    itersstarttime = tic;
    
    ftrPrm1 = ftrPrm;
    % KB: choose a different interval of samples each fern
    ftrPrm1.corsamples = [corsamplestarts(k),corsampleends(k)];
    
    %Select features from correlation score directly
    if R==0
      %If occlusion-centered approach, enforce feature variety
      if(s>1 && Stot>1 && ~isempty(occlD))
        keep=find(ismember(mg,masks(s-1,:)));
        if(~isempty(keep))
          data2=data(:,keep);dfFtrs2=dfFtrs(:,keep);
          stdFtrs2=stdFtrs(keep,keep);
          ftrPrm1.F=length(keep);
          [use,ftrs] = selectCorrFeat(M,ysTar,data2,...
            ftrPrm1,stdFtrs2,dfFtrs2);
          use=keep(use);
        else
          [use,ftrs] = selectCorrFeat(M,ysTar,data,...
            ftrPrm1,stdFtrs,dfFtrs);
        end
        %ow use all features
      else
        [use,ftrs] = selectCorrFeat(M,ysTar,data,ftrPrm1,stdFtrs,dfFtrs);
      end
      
      timingInfo.selectFeatures = timingInfo.selectFeatures+toc(itersstarttime);
      selectfeaturetime = tic;
            
      %Train regressor using selected features
      [reg1,ys1] = regFun(ysTar,ftrs,M,regPrm);
      reg1.fids = use;
      
      %fprintf(1,'Saving fern features in reg\n');
      %reg1.X = ftrs;
      
      best = {reg1,ys1};
      
      timingInfo.regress = timingInfo.regress+toc(selectfeaturetime);

      
    else
      assert(false,'codepath needs investigation; see ftrPrm.type below');
      %If occlusion-centered approach, enforce feature variety
      if(s>1 && Stot>1 && ~isempty(occlD))
        if(Stot==5),keep=find(ismember(mg,masks(s-1,:)));
        elseif(Stot==12), keep=find(ismember(mg,masks{s-1}));
        end
        data2=data(:,keep); F=length(keep);
        %ow use all features
      else
        data2=data;F=ftrPrm.F;keep=1:F;
      end
      %Select features with random step optimization
      e = lossFun(ysTar,zeros(N,D));
      type=ftrPrm.type;
      if(type>2)
        type=type-2;
      end
      for r=1:R
        if(type==1),
          use=randSample(F,M);ftrs = data2(:,use);
        else
          use=randSample(F,M*2);use=reshape(use,2,M);
          ftrs=data2(:,use(1,:))-data2(:,use(2,:));
        end
        %Train regressor using selected features
        [reg1,ys1]=regFun(ysTar,ftrs,M,regPrm);
        e1 = lossFun(ysTar,ys1);use=keep(use);
        %fprintf('%f - %f \n',e,e1);
        if(e1<=e), e=e1; reg1.fids=use; best={reg1,ys1}; end
      end
    end
    %Get output of regressor
    [regInfo{k,s},ysPred(:,:,s)] = deal(best{:});
    clear best;
    %If occlusion-centered, get occlusion averages by group
    if D>10 && Stot>1 && ~isempty(occlD)
      ftrsOccl(:,k,s)=sum(occlD.featOccl(:,regInfo{k,s}.fids),2)./K;
    end
        
  end
  %Combine S1 regressors to form prediction (Occlusion-centered)
  if D>10 && Stot>1 && ~isempty(occlD)
    assert(false,'AL');
    
    %(WEIGHTED MEAN)
    %ftrsOccl contains total occlusion of each Regressor
    % weight should be inversely proportional, summing up to 1
    weights=1-normalize(ftrsOccl(:,k,:));ss=sum(weights,3);
    weights=weights./repmat(ss,[1,1,Stot]);
    %when all are fully occluded, all get proportional weight
    % (regular mean)
    weights(ss==0,1,:)=1/Stot;
    weights=repmat(weights,[1,D,1]);
    %OLD
    for s=1:Stot
      ysSum=ysSum+ysPred(:,:,s).*weights(:,:,s);
    end
  else
    %Update output
    ysSum = ysSum+ysPred;
  end
  timingInfo.iter(k) = toc(iterkstarttime);
end
% create output struct
clear data ys; 
ysPr = ysSum(datareorder,:);
if R==0 
  clear stdFtrs dfFtrs; 
end
end

function [regSt,Y_pred] = trainFern(Y,X,M,prm)
% Train single random fern regressor.
%
% USAGE
%  [regSt,Y_pred]=trainFern(Y,X,M,prm)
%
% INPUTS
%  Y        - [NxD] target output values
%  X        - [NxF] data measurements (features)
%  M        - fern depth
%  prm      - additional parameters
%   .thrr     - fern bin thresholding
%   .reg      - fern regularization term
%
% OUTPUTS
%  regSt    - struct with learned regressors models
%    .ysFern    - average values for fern bins
%    .thrs      - thresholds used at each M level
%  ysPr       - [NxD] predicted output values
%
% See also

dfs = {'thrr',[-1 1]/5,'reg',.01,'useFern3',false};
[thrr,reg,useFern3] = getPrmDflt(prm,dfs,1);
N = size(Y,1);
assert(size(X,2)==M); % currently fern depth should always match total num ftrs
fids = uint32(1:M);
thrs = rand(1,M)*(thrr(2)-thrr(1))+thrr(1);
% inds(i) = 1+sum(2.^(M-1:-1:0).*(X(i,:)<=thrs))
% count = hist(inds,1:32)'
% ysFern(i,d) = sum(Y(inds==i,d)-mu(d))

if ~useFern3
  % orig
  [inds,mu,ysFern,count,~] = fernsInds2(X,fids,thrs,Y);
  ysFern = bsxfun(@plus,bsxfun(@rdivide,ysFern,max(count+reg*N,eps)),mu);
  
  dyFernCnt = [];
  dyFernSum = [];  
else
  mu = nanmean(Y);
  dY = bsxfun(@minus,Y,mu);
  [inds,dyFernSum,~,dyFernCnt] = Ferns.fernsInds3(X,fids,thrs,dY);
  ysFernCntUse = max(dyFernCnt+reg*N,eps); % [2^MxD], counts for each fernbin/coord
  ysFern = bsxfun(@plus,dyFernSum./ysFernCntUse,mu);
end

% S=size(count,1);
% cnts = repmat(count,[1,D]);
% for d=1:D
%     %ysFern(:,d) = ysFern(:,d) ./ max(cnts(:,d)+(1+1000/cnts(:,d))',eps) + mu(d);
%     ysFern(:,d) = ysFern(:,d) ./ max(count+reg*N,eps) + mu(d);
% end

Y_pred = ysFern(inds,:);
regSt = struct(...
  'N',N,... % scalar
  'fernSum',dyFernSum,... % [2^MxD]
  'fernCount',dyFernCnt,... % [2^MxD], counts for each fern bin/coord, treating NaNs in output vectors Y as missing
  'ysFern',ysFern,... % [2^MxD], fern predictions for each fern index
  'thrs',thrs,... % [1xM], fern thresholds
  'yMu',mu); % [1xD], (nan)mean of output vectors

end

function [regSt,Y_pred]=trainLin(Y,X,~,~)
% Train single linear regressor.
%
% USAGE
%  [regSt,Y_pred]=linFern(Y,X)
%
% INPUTS
%  Y        - [NxD] target output values
%  X        - [NxF] data measurements (features)
%
% OUTPUTS
%  regSt    - struct with learned regressors models
%    .W         - linear reg weights
%  Y_pred       - [NxD] predicted output values
%
% See also
W = X\Y; Y_pred = X*W;regSt = struct('W',W);
end

