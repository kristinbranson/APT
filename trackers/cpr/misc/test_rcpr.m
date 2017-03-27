% Test function 
%       + phisT: labels to compute the loss (empty if there are no labels) 
%       + bboxesT: bounding boxes
%       + IsT: images
%       + regModel: regressor model (obtained from training)
%       + regParam: regression parameters
%       + prunePrm: prune parameters
%       + piT: inital label position (optional)

function [pT,pRTT,lossT,fail,p_t] = ...
  test_rcpr(phisT,bboxesT,IsT,regModel,regPrm,prunePrm,piT)

if nargin<7 || isempty(piT)
    RT1 = prunePrm.numInit;
    % Initialize randomly using RT1 shapes drawn from training
    if ~isfield(prunePrm,'dorotate')
      prunePrm.dorotate = false;
    end
    piT = shapeGt('initTest',[],bboxesT,regModel.model,regModel.pStar,...
      regModel.pGtN,RT1,prunePrm.dorotate);
else
    RT1 = size(piT,3);
end

%% TEST 
testPrmT = struct('RT1',RT1,'pInit',bboxesT,...
    'regPrm',regPrm,'initData',piT,'prunePrm',prunePrm,...
    'verbose',0);
[pT,pRTT,p_t,fail] = rcprTest(IsT,regModel,testPrmT);

%Compute loss
if ~isempty(phisT)
    lossT = shapeGt('dist',regModel.model,pT,phisT);
    fprintf('--------------DONE\n');
else
    lossT = [];
end
