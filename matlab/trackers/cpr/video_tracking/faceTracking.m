function faceTracking(seqVideo,rcprFile,detFile)
%
% Demo code to track face landmarks from scratch in video. 
%
% Video processing is as follows:
% 1- Run face detector frame-by-frame 
%      for more info, see acfTrain.m, bbDetect.m in Piotr's Toolbox
%       "Fast Feature Pyramids for Object Detection"
%       P. Dollár, R. Appel, S. Belongie and P. Perona 
%       PAMI 2014
%
% 2- Run RCPR landmark estimator frame-by-frame from detector's output
%      for more info, see ../demoRCPR.m
%      "Robust face landmark estimation under occlusion" 
%      X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%      ICCV'13, Sydney, Australia
%      http://vision.caltech.edu/xpburgos/ICCV13/
%
% 3- Merge landmarks estimates in space and time (tracking)
%      for more info, see poseNMS.m
%      "Merging Pose Estimates Across Space and Time". 
%      X.P. Burgos-Artizzu, D.Hall, P.Perona, P.Dollar. 
%      BMVC'13, Bristol, UK.
%      http://vision.caltech.edu/~dhall/projects/MergingPoseEstimates/
%
% USAGE
%  faceTracking(seqVideo,rcprFile,detFile)
%
% INPUTS
%  seqVideo   - input seq video
%  rcprFile   - pre-trained RCPR model 
%  detFile    - pre-trained face detector 
%
% OUTPUTS
%
% EXAMPLE
% 
%    %Track using COFW model with occlusion estimation
%    %(retrained from real face detector output)
%    seqVideo='Zurich_03193_03286.seq';
%    rcprFile='models/rcpr_COFW_detector.mat';
%    detFile='models/frontal_faceDetector.mat';
%    faceTracking(seqVideo,rcprFile,detFile);
%    
%    %Track using 300-Faces in the wild model
%    seqVideo='Zurich_03193_03286.seq';
%    rcprFile='models/rcpr_300W_detector.mat';
%    detFile='models/frontal_faceDetector.mat';
%    faceTracking(seqVideo,rcprFile,detFile);
%
% For full function example usage, see individual function help and papers
%
% Copyright 2013 
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our papers if you use the code:
%  - Landmark estimation
%      "Robust face landmark estimation under occlusion" 
%      X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%      ICCV'13, Sydney, Australia
%  - Pose tracker
%      "Merging Pose Estimates Across Space and Time". 
%      X.P. Burgos-Artizzu, D.Hall, P.Perona, P.Dollar. 
%      BMVC'13, Bristol, UK.
%  - Object detector
%      "Fast Feature Pyramids for Object Detection"
%      P. Dollár, R. Appel, S. Belongie and P. Perona 
%      PAMI 2014

%% LOAD DATA
%Load rcpr model
load(rcprFile,'regModel','regPrm');RT1=5;
prunePrm=struct('prune',1,'maxIter',2,'th',0.1,'tIni',10);
testPrm = struct('RT1',RT1,'pInit',[],...
    'regPrm',regPrm,'initData',[],'prunePrm',prunePrm,...
    'verbose',1);

%Load face detector
d=load(detFile); d=d.detector;
pNms=struct('overlap',.75,'separate',1);
detector=acfModify(d,'pNms',pNms,'cascThr',0,'cascCal',0);%-1,0

info=seqIo(seqVideo,'getinfo'); T=info.numFrames;
if(T<300)
    %Load entire video into memory (small clip)
    Is=seqIo(seqVideo,'toImgs'); 
    [good,pInit,bbs,pRT]=runDetector_RCPR(Is,detector,testPrm,regModel);
else
    partSize=300; 
    good=zeros(1,T);bbs=cell(1,T);pInit=zeros(T,4);
    pRT=zeros(T,regModel.model.D,RT1);
    for t=1:partSize:T
        frames= t:min(T,t+partSize-1);
        Is=seqIo(seqVideo,'toImgs',[],1,frames(1)-1,frames(end)-1); 
        [good(frames),pInit(frames,:),...
         bbs(frames),pRT(frames,:,:)]=...
            runDetector_RCPR(Is,detector,testPrm,regModel);
    end
end
if(strcmp(regModel.model.name,'cofw')), th=regModel.th;
else th=-1;
end
Y=performTracking(pRT,bbs,good,th);
save([seqVideo(1:end-4) '_RES.mat'],'Y','pInit');
showResults(seqVideo,Y,pInit,regModel.model)
end

function [good,pInit,bbs,pRT]=runDetector_RCPR(Is,detector,testPrm,regModel)
%% RUN FACE DETECTOR
T=size(Is,4); Isc=cell(1,T);bbs=cell(1,T); good=zeros(1,T); detTH=5; 
pInit=zeros(T,4);
for t=1:T
    im=Is(:,:,:,t); Isc{t}=rgb2gray(im);
    bb=acfDetect(im,detector); [h,w,nC]=size(im);
    if(any(bb(:,5)>detTH)),        
        bb=bb(:,1:5); bb=bb(bb(:,5)>detTH,:);
        if(bb(:,1)<1), bb(:,3)=bb(:,1)+bb(:,3)-1; bb(:,1)=1; end
        if(bb(:,2)<1), bb(:,4)=bb(:,2)+bb(:,4)-1; bb(:,2)=1; end
        
        if(bb(:,1)+bb(:,3)>w), bb(:,3)=w-bb(:,1); end
        if(bb(:,2)+bb(:,4)>h), bb(:,4)=h-bb(:,2); end
        bbs{t}=bb;good(t)=1; pInit(t,:)=bb(1,1:4);
    end
end
%% RUN RCPR 
pRT=zeros(T,regModel.model.D,testPrm.RT1);
keep=find(good); testPrm.pInit=pInit(keep,:);
testPrm.initData=shapeGt('initTest',Isc(keep),bbs(keep),regModel.model,...
    regModel.pStar,regModel.pGtN,testPrm.RT1);
[~,pRT(keep,:,:)] = rcprTest(Isc(keep),regModel,testPrm);
%Round up the pixel positions
pRT(:,1:regModel.model.nfids*2,:)=round(pRT(:,1:regModel.model.nfids*2,:));
end

function Y=performTracking(pRT,bbs,good,th)
%% MERGE ESTIMATES
[T,D,RT1]=size(pRT);X=cell(1,T);S=cell(1,T);R=cell(1,T);
for t=1:T
    if(good(t))
        X{t}= permute(pRT(t,:,:),[3 2 1]);
        S{t}= repmat(mean(bbs{t}(:,5)),RT1,1);
    end
end
%%Track positions
prmTrack=struct('norm',100,'th',1,'lambda',.25,'lambda2',0,...
  'nPhase',4,'window',1000,'symmetric',1,'isAng',zeros(1,D),...
  'ms',[],'bnds',[]); 
Y = poseNMS( X, S, R, 1, prmTrack );
Y=permute(Y,[3 2 1]);
%If using COFW model, binarize occlusion according to learnt threshold
if(th~=-1)
    occl=Y(:,59:end); 
    occl(occl<th)=0; occl(occl>=th)=1;
    Y(:,59:end)=occl; 
end
end

function showResults(seqVideo,Y,pInit,model)
%% Show results
T=size(Y,1); 
sr=seqIo(seqVideo,'reader');
for t=1:T
    sr.seek(t-1); im=sr.getframe();
    clf,imshow(im), hold on,
    if(any(pInit(t,:)>0)), bbApply('draw',pInit(t,:),'b'); end
    shapeGt('draw',model,im,Y(t,:),{'drawIs',0,'lw',15});
    pause(.02)
end
sr.close();
end