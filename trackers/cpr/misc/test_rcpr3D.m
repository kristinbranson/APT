% Test function for 3D recosntruction
%       + bboxesT: bounding boxes
%       + IsT: images
%       + regModel: regressor model (obtained from training)
%       + regParam: regression parameters
%       + prunePrm: prune parameters
%       + piT: inital label position (optional)
function [pT3D,pRTT3D]=test_rcpr3D(bboxesT,IsT,regModel,regPrm,prunePrm,piT)
if nargin==5
    % Setup parameters
    RT1=5;
    %Initialize randomly using RT1 shapes drawn from training
    piT=shapeGt('initTest',IsT,bboxesT,regModel.model,regModel.pStar,regModel.pGtN,RT1);
else
    RT1=size(piT,3);
end

%% TEST on TRAINING data
%Create test struct
testPrmT = struct('RT1',RT1,'pInit',bboxesT,...
    'regPrm',regPrm,'initData',piT,'prunePrm',prunePrm,...
    'verbose',0);
%Test
t=clock;[pT3D,pRTT3D] = rcprTest(IsT,regModel,testPrmT);t=etime(clock,t);
%Round up the pixel positions
