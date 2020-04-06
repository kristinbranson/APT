% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For pre-requisites and compilation, see CONTENTS.m
%
% This code tests a pre-trained RCPR (data/rcpr.mat) on COFW dataset. 
%  COFW test is composed of one file (data/COFW_test.mat) 
%  which contains:  
%    -phisT - ground truth shapes 
%    -IsT - images 
%    -bboxesT - face bounding boxes 
%  If you change path to folder containing training/testing files, change
%  this variable here:
COFW_DIR='./data/';
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD COFW dataset (test only)
% testing images and ground truth
testFile=[COFW_DIR 'COFW_test.mat'];
load(testFile,'phisT','IsT','bboxesT');bboxesT=round(bboxesT);
nfids=size(phisT,2)/3;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD PRE-TRAINED RCPR model
load('models/rcpr_COFW.mat','regModel','regPrm','prunePrm');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST
%Initialize randomly using RT1 shapes drawn from training
RT1=5;
p=shapeGt('initTest',IsT,bboxesT,regModel.model,...
    regModel.pStar,regModel.pGtN,RT1);
%Create test struct
testPrm = struct('RT1',RT1,'pInit',bboxesT,...
    'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
    'verbose',1);
%Test
t=clock;[p,pRT] = rcprTest(IsT,regModel,testPrm);t=etime(clock,t);
%Round up the pixel positions
p(:,1:nfids*2)=round(p(:,1:nfids*2));
% Use threshold computed during training to 
% binarize occlusion
occl=p(:,(nfids*2)+1:end);
occl(occl>=regModel.th)=1;occl(occl<regModel.th)=0;
p(:,(nfids*2)+1:end)=occl;
%Compute loss
loss = shapeGt('dist',regModel.model,p,phisT);
fprintf('--------------DONE\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DISPLAY Standard histogram of errors
figure(1),clf,
mu1=mean(loss(loss<0.1));muAll=mean(loss);
fail=100*length(find(loss>0.1))/length(loss);
bins=log10(min(loss)):0.1:log10(max(loss));ftsz=20;
[n,b]=hist(log10(loss),bins); n=n./sum(n);
semilogx(10.^b,n,'b','LineWidth',3);
hold on,plot(zeros(10,1)+2.5,linspace(0,max(n),10),'--k');
ticks=[0 linspace(min(loss),max(loss)/4,5) ...
    linspace((max(loss)/3),max(loss),3)];
ticks=round(ticks*100)/100;
set(gca,'XTick',ticks,'FontSize',ftsz);
xlabel('error','FontSize',ftsz);ylabel('probability','FontSize',ftsz),
title(['Mean error=' num2str(muAll,'%0.2f') '   ' ...
    'Mean error (<0.1)=' num2str(mu1,'%0.2f') '   ' ...
    'Failure rate (%)=' num2str(fail,'%0.2f')],'FontSize',ftsz);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUALIZE Example results on a test image
figure(3),clf,
nimage=12;
%Ground-truth
subplot(1,2,1),
shapeGt('draw',regModel.model,IsT{nimage},phisT(nimage,:),{'lw',20});
title('Ground Truth');
%Prediction
subplot(1,2,2),shapeGt('draw',regModel.model,IsT{nimage},p(nimage,:),...
    {'lw',20});
title('Prediction');