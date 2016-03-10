function [regInfo,ysPr] = regTrain(data,ys,varargin)
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

% get/check parameters
dfs={'type',1,'ftrPrm','REQ','K',1,...
    'loss','L2','R',0,'M',5,'model',[],'prm',{},...
    'occlD',[],'occlPrm',struct('Stot',1)};
[regType,ftrPrm,K,loss,R,M,model,prm,occlD,occlPrm]=...
    getPrmDflt(varargin,dfs,0);
%Set base regression type
switch regType
  case 1, regFun = @trainFern;
  case 2, regFun = @trainLin;
  otherwise, error('unknown regressor type');
end
%Set loss type 
assert(any(strcmp(loss,{'L1','L2'})));
%precompute feature std to be used by selectCorrFeat
if R==0
  [stdFtrs,dfFtrs] = statsFtrs(data,ftrPrm); 
else%random step optimization selection
    switch(loss)
        case 'L1',  lossFun=@(ys,ysGt) mean(abs(ys(:)-ysGt(:)));
        case 'L2',  lossFun=@(ys,ysGt) mean((ys(:)-ysGt(:)).^2);
    end
end

Stot = occlPrm.Stot;
[N,D] = size(ys);
ysSum = zeros(N,D);
regInfo = cell(K,Stot);

%If using occlusion-centered approach, set up masks
if(Stot>1 && ~isempty(occlD))
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
%Iterate through K boosted regressors
for k=1:K
    %Update regression target
    ysTar = ys-ysSum; 
    %Train Stot different regressors
    ysPred = zeros(N,D,Stot); 
    for s=1:Stot
        %Select features from correlation score directly
        if(R==0)
            %If occlusion-centered approach, enforce feature variety
            if(s>1 && Stot>1 && ~isempty(occlD))
                keep=find(ismember(mg,masks(s-1,:)));
                if(~isempty(keep))
                    data2=data(:,keep);dfFtrs2=dfFtrs(:,keep);
                    stdFtrs2=stdFtrs(keep,keep);
                    ftrPrm1=ftrPrm;ftrPrm1.F=length(keep);
                    [use,ftrs] = selectCorrFeat(M,ysTar,data2,...
                    ftrPrm1,stdFtrs2,dfFtrs2);
                    use=keep(use);
                else
                    [use,ftrs] = selectCorrFeat(M,ysTar,data,...
                    ftrPrm,stdFtrs,dfFtrs);
                end
            %ow use all features    
            else
                [use,ftrs] = selectCorrFeat(M,ysTar,data,...
                  ftrPrm,stdFtrs,dfFtrs); % TO DO. edit this???
            end
            %Train regressor using selected features
            [reg1,ys1] = regFun(ysTar,ftrs,M,prm);
            reg1.fids = use;
            
            %%%%XXX
            fprintf(1,'Saving fern features in reg\n');
            reg1.X = ftrs;
            
            best = {reg1,ys1};
        %Select features using random step optimization            
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
                [reg1,ys1]=regFun(ysTar,ftrs,M,prm);
                e1 = lossFun(ysTar,ys1);use=keep(use);
                %fprintf('%f - %f \n',e,e1);
                if(e1<=e), e=e1; reg1.fids=use; best={reg1,ys1}; end
            end 
        end
        %Get output of regressor
        [regInfo{k,s},ysPred(:,:,s)]=deal(best{:});clear best;
        %If occlusion-centered, get occlusion averages by group
        if(D>10 && Stot>1 && ~isempty(occlD))
            ftrsOccl(:,k,s)=sum(occlD.featOccl(:,regInfo{k,s}.fids),2)./K;
        end
    end
    %Combine S1 regressors to form prediction (Occlusion-centered)
    if(D>10 && Stot>1 && ~isempty(occlD))
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
        ysSum=ysSum+ysPred;
    end
end
% create output struct
clear data ys; ysPr=ysSum;
if(R==0), clear stdFtrs dfFtrs; end
end

function [regSt,Y_pred]=trainFern(Y,X,M,prm)
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

% get/check parameters
dfs={'thrr',[-1 1]/5,'reg',.01}; 
[thrr,reg] = getPrmDflt(prm,dfs,1);
[N,D] = size(Y); 
fids = uint32(1:M);
thrs = rand(1,M)*(thrr(2)-thrr(1))+thrr(1);
% inds(i) = 1+sum(2.^(M-1:-1:0).*(X(i,:)<=thrs))
% count = hist(inds,1:32)'
% ysFern(i,d) = sum(Y(inds==i,d)-mu(d))

%[inds,mu,ysFern,count,~] = fernsInds2(X,fids,thrs,Y);
fprintf(1,'Using FernsInds3\n');
[inds,mu,ysFern,count,ysFernCnt] = fernsInds3(X,fids,thrs,Y);

% ysFern(i) = mean(Y(inds==i))

USEOLD = false;
if USEOLD
  ysFern = bsxfun(@plus,bsxfun(@rdivide,ysFern,max(count+reg*N,eps)),mu);
else
  ysFernCntUse = max(ysFernCnt+reg*N,eps);
  ysFern = bsxfun(@plus,ysFern./ysFernCntUse,mu);
end

% S=size(count,1);
% cnts = repmat(count,[1,D]);
% for d=1:D
%     %ysFern(:,d) = ysFern(:,d) ./ max(cnts(:,d)+(1+1000/cnts(:,d))',eps) + mu(d);
%     ysFern(:,d) = ysFern(:,d) ./ max(count+reg*N,eps) + mu(d);
% end
Y_pred = ysFern(inds,:); 
clear dfYs;
clear cnts vars inds mu;%conf
regSt = struct('ysFern',ysFern,'thrs',thrs);
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

%Compute std and diff between ftrs to be used by fast correlation selection
function [stdFtrs,dfFtrs]=statsFtrs(ftrs,ftrPrm)
N = size(ftrs,1);

if isfield(ftrPrm,'nsample_std'),
  nsample = ftrPrm.nsample_std;
else
  nsample = N;
end

if isnumeric(ftrPrm.type) && ftrPrm.type==1
    stdFtrs = std(ftrs); muFtrs = mean(ftrs);
    dfFtrs = bsxfun(@minus,ftrs,muFtrs);
else
    muFtrs = mean(ftrs);
    dfFtrs = bsxfun(@minus,ftrs,muFtrs);
    %dfFtrs = ftrs-repmat(muFtrs,[N,1]);
    if nsample < N,
      dosample = rand(N,1) <= nsample/N;
    else
      dosample = true(N,1);
    end
    stdFtrs=stdFtrs1(ftrs(dosample,:));
end
end