cpr_type=2;
nTr=40;
nT=10;


[file,folder] = uigetfile('*.mat');
File=fullfile(folder,file);


load(File,'phis','Is','bboxes');
bboxes=round(bboxes);

phisNaN = find(any(isnan(phis),2));
phis(phisNaN,:) = [];
Is(phisNaN) = [];
bboxes(phisNaN,:) = [];

%%
idxrand=randperm(numel(Is)); 

% Traning data
idxTr=idxrand(1:nTr);
phisTr=phis(idxTr,:);
bboxesTr=bboxes(idxTr,:);
IsTr=Is(idxTr);
nfids=size(phisTr,2)/2;

% Test data
idxT=idxrand(nTr+1:nTr+nT);
phisT=phis(idxT,:);
bboxesT=bboxes(idxT,:);
IsT=Is(idxT);

% Setup parameters
%Create larva model (30 landmarks without visibility)
model = shapeGt('createModel','larva');
%RCPR(features+restarts) PARAMETERS
%(type 4, points relative to any 2 landmarks)
T=100;K=50;L=20;RT1=5;
if cpr_type==1
    ftrPrm = struct('type',2,'F',400,'nChn',1,'radius',1);
    prunePrm=struct('prune',0,'maxIter',2,'th',0.1,'tIni',10);
elseif cpr_type==2
    ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',1.5);
    %smart restarts are enabled
    prunePrm=struct('prune',1,'maxIter',2,'th',0.1,'tIni',10);
end
prm=struct('thrr',[-1 1]/5,'reg',.01);
occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',1,'th',.5);
regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',model,'prm',prm);

%% TRAIN
%Initialize randomly L shapes per training image
[pCur,pGt,pGtN,pStar,imgIds,N,N1]=shapeGt('initTr',...
    IsTr,phisTr,model,[],bboxesTr,L,10);
initData=struct('pCur',pCur,'pGt',pGt,'pGtN',pGtN,'pStar',pStar,...
    'imgIds',imgIds,'N',N,'N1',N1);
%Create training structure
trPrm=struct('model',model,'pStar',[],'posInit',bboxesTr,...
    'T',T,'L',L,'regPrm',regPrm,'ftrPrm',ftrPrm,...
    'pad',10,'verbose',1,'initData',initData);
%Train model
[regModel,~] = rcprTrain(IsTr,phisTr,trPrm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST
%Initialize randomly using RT1 shapes drawn from training
pi=shapeGt('initTest',IsT,bboxesT,model,pStar,pGtN,RT1);
%Create test struct
testPrm = struct('RT1',RT1,'pInit',bboxesT,...
    'regPrm',regPrm,'initData',pi,'prunePrm',prunePrm,...
    'verbose',1);
%Test
t=clock;[p,pRT] = rcprTest(IsT,regModel,testPrm);t=etime(clock,t);
%Round up the pixel positions
p(:,1:nfids*2)=round(p(:,1:nfids*2));
%Compute loss
loss = shapeGt('dist',regModel.model,p,phisT);
fprintf('--------------DONE\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST on TRAINING data
%Initialize randomly using RT1 shapes drawn from training
piTr=shapeGt('initTest',IsTr,bboxesTr,model,pStar,pGtN,RT1);
%Create test struct
testPrmTr = struct('RT1',RT1,'pInit',bboxesTr,...
    'regPrm',regPrm,'initData',piTr,'prunePrm',prunePrm,...
    'verbose',1);
%Test
t=clock;[pTr,pRTTr] = rcprTest(IsTr,regModel,testPrmTr);t=etime(clock,t);
%Round up the pixel positions
pTr(:,1:nfids*2)=round(pTr(:,1:nfids*2));
%Compute loss
lossTr = shapeGt('dist',regModel.model,pTr,phisTr);
fprintf('--------------DONE\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUALIZE Example results on a test image
figure(3),clf,
nimage=55;
%Ground-truth
subplot(1,2,1),
shapeGt('draw',model,IsT{nimage},phisT(nimage,:),{'lw',20});
title('Ground Truth');
%Prediction
subplot(1,2,2),shapeGt('draw',model,IsT{nimage},p(nimage,:),...
    {'lw',20});
title('Prediction');