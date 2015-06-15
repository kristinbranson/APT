function ysSum = regApply(p,X,regInfo,regPrm)
% Apply boosted regressor.
%
% USAGE
%  ysSum = regApply(p,X,regInfo,regPrm)
%
% INPUTS
%  p        - [NxD] initial pose
%  X        - [NxF] N length F feature vectors
%  regInfo  - structure containing regressor info, output of regTrain
%  regPrm
%   .type     - [1] type of regression
%                   1=fern, 2=linear
%   .model    - [] optional, model to use (see shapeGt)
%   .ftrPrm   - [REQ] prm struct (see shapeGt>ftrsGen)
%   .K        - [1] number of boosted regressors
%   .Stot     - [1] number of regressors trained at each round
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
%  ysSum       - [NxD] predicted output values
%
% See also
%          demoRCPR, FULL_demoRCPR, rcprTrain, regTrain
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

type=regPrm.type;K=regPrm.K;Stot=regPrm.occlPrm.Stot;occlD=regPrm.occlD;
%Set up reg function
[N,D]=size(p);
switch(type)
    case 1, regFun=@applyFern;
    case 2, regFun=@applyLin;
end

%Prepare data
ysSum=zeros(N,D);
if(D>10 && Stot>1 && ~isempty(occlD))
    ftrsOccl=zeros(N,K,Stot);
end
%For each boosted regressor
for k=1:K
    %Occlusion-centered weighted mean
    if(D>10 && Stot>1 && ~isempty(occlD))
        ysPred = zeros(N,D,Stot);
        for s=1:Stot
            ysPred(:,:,s)=regFun(X,regInfo{k,s},regPrm);
            ftrsOccl(:,k,s)=sum(occlD.featOccl(:,regInfo{k,s}.fids),2)./K;
        end
        %(WEIGHTED MEAN)
          %ftrsOccl contains total occlusion of each Regressor
          % weight should be inversely proportional, summing up to 1
          weights=1-normalize(ftrsOccl(:,k,:));ss=sum(weights,3);
          weights=weights./repmat(ss,[1,1,Stot]);
          %when all are fully occluded, all get proportional weight 
          % (regular mean)
          weights(ss==0,1,:)=1/Stot;
          weights=repmat(weights,[1,D,1]);
        for s=1:Stot
            ysSum=ysSum+ysPred(:,:,s).*weights(:,:,s);
        end
    %Normal
    else
        ysPred=regFun(X,regInfo{k,:},regPrm);
        ysPred = median(ysPred,3);
        ysSum=ysSum+ysPred;
    end
end
end

function Y_pred=applyFern(X,regInfo,regPrm)
% Apply single random fern regressor.
%
% USAGE
%  Y_pred=applyFern(X,regInfo,regPrm)
%
% INPUTS
%  X        - [NxF] data measurements (features)
%  regInfo  - structure containing trained fern
%  regPrm   - regression parameters used
%   .M      - fern depth
%
% OUTPUTS
%  Y_pred   - [NxD] predicted output values
%
% See also

type=size(regInfo.fids,1);M=regPrm.M;
if(type==1), ftrs=X(:,regInfo.fids);
else ftrs=X(:,regInfo.fids(1,:))-X(:,regInfo.fids(2,:));
end
inds = fernsInds(ftrs,uint32(1:M),regInfo.thrs);
Y_pred=regInfo.ysFern(inds,:);
end

function Y_pred=applyLin(X,regInfo,~)
% Apply single linear regressor.
%
% USAGE
%  [Y_pred,Y_conf]=applyLin(X,regInfo,~)
%
% INPUTS
%  X        - [NxF] data measurements (features)
%  regInfo  - structure containing trained linear regressor
%
% OUTPUTS
%  Y_pred   - [NxD] predicted output values
%
% See also

type=size(regInfo.fids,1);
if(type==1), ftrs=X(:,regInfo.fids);
else ftrs=X(:,regInfo.fids(1,:))-X(:,regInfo.fids(2,:));
end
Y_pred=ftrs*regInfo.W;
end