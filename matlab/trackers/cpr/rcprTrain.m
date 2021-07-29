function [regModel,pAll] = rcprTrain( Is, pGt, varargin )
% Train multistage robust cascaded shape regressor
%
% USAGE
%  [regModel,pAll] = rcprTrain( Is, pGt, varargin )
%
% INPUTS
%  Is       - cell [N] input images, optionally with 'channels' in 3rd dim
%  pGt      - [NxR] ground truth shape for each image
%  varargin - additional params (struct or name/value pairs)
%   .model    - [REQ] shape model
%   .pStar    - [] initial shape
%   .posInit  - [] known object position (e.g. tracking output)
%   .T        - [REQ] number of stages
%   .L        - [1] data augmentation factor
%   .regPrm   - [REQ] param struct for regTrain
%   .ftrPrm   - [REQ] param struct for shapeGt>ftrsGen
%   .regModel - [Tx1] previously learned single stage shape regressors
%   .pad      - amount of padding around bbox
%   .verbose  - [0] method verbosity during training
%   .initData - initialization parameters (see shapeGt>initTr)
%
% OUTPUTS
%  regModel - learned multi stage shape regressor:
%   .model    - shape model
%   .pStar    - [1xR] average shape
%   .pDstr    - [NxR] ground truth shapes
%   .T        - number of stages
%   .pGtN     - [NxR] normalized ground truth shapes
%   .th       - threshold for occlusion detection
%   .regs     - [Tx1] struct containing learnt cascade of regressors
%      .regInfo  - [KxStot] regressors
%         .ysFern  - [2^MxR] fern bin averages
%         .thrs    - [Mx1] thresholds
%         .fids    - [2xM] features used
%      .ftrPos   - feature information
%         .type    - type of features
%         .F       - number of features
%         .nChn    - number of channels used
%         .xs      - [Fx3] features position
%         .pids    - obsolete
%
%  pAll     - shape estimation at each iteration T
%
% EXAMPLE
%
% See also  demoRCPR, FULL_demoRCPR
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

dfs = {...
  'model','REQ',...
  'regPrm','REQ',...
  'ftrPrm','REQ',...
  'regModel',[],... % previously learned regressors
  'bboxes',[],...
  'initData',[],... % if empty, initial conditions generated using initPrm 
  'initPrm',[],...
  'pStar',[],...    
  'verbose',0,...
  };
[model,regPrm,ftrPrm,regModel,bboxes,initData,initPrm,pStar,verbose] = ...
  getPrmDflt(varargin,dfs,1);

if isempty(initData)
  fprintf('Generating initData...\n');  
  
  initData = struct();
  [initData.pCur,...
    initData.pGt,...
    initData.pGtN,...
    initData.pStar,...
    initData.imgIds,...
    initData.N,...
    initData.N1] = shapeGt('initTr',[],pGt,model,pStar,bboxes,...
      initPrm.Naug,initPrm.augpad,initPrm.augrotate);  
end

fprintf('train. USE_AL_CORRECTION=%d\n',regPrm.USE_AL_CORRECTION);

pause(3.0);

[regModel,pAll] = rcprTrain1(Is,initData,model,bboxes,regPrm,ftrPrm,...
  regModel,verbose);
end

function [regModel,pAll] = rcprTrain1(Is,initData,model,bbs,regPrm,ftrPrm,...
  regModel,verbose)

pCur = initData.pCur;
pGt = initData.pGt; % note: initD.pGt contains replicates
pGtN = initData.pGtN;
pStar = initData.pStar; 
imgIds = initData.imgIds;
N = initData.N;
N1 = initData.N1;
D = size(pGt,2);
assert(D==model.D);

T = regPrm.T;

% remaining initialization, possibly continue training from
% previous model
pAll = zeros(N1,D,T+1);
regs = repmat(struct('regInfo',[],'ftrPos',[]),T,1);
if isempty(regModel)
  t0 = 1;
  pAll(:,:,1) = pCur(1:N1,:);
else % not working for mouse_paw3D
  assert(false,'Unsupported codepath (what is cprApply)');
  %   t0=regModel.T+1; regs(1:regModel.T)=regModel.regs;
  %   [~,pAll1]=cprApply(Is,regModel,'imgIds',imgIds,'pInit',pCur);
  %   pAll(:,:,1:t0)=pAll1(1:N1,:,:); pCur=pAll1(:,:,end);
end

loss = mean(shapeGt('dist',model,pCur,pGt));
if verbose
  fprintf('  t=%i/%i       loss=%f     \n',t0-1,T,loss);
end
tStart = clock;%pCur_t=zeros(N,D,T+1);
bboxes = bbs(imgIds,:);
ftrPrmRadiusOrig = ftrPrm.radius;

for t=t0:T
  if regPrm.USE_AL_CORRECTION
    pCurN_al = shapeGt('projectPose',model,pCur,bboxes);
    pGtN_al = shapeGt('projectPose',model,pGt,bboxes);
    assert(isequal(size(pCurN_al),size(pGtN_al)));
    pDiffN_al = Shape.rotInvariantDiff(pCurN_al,pGtN_al,1,3); % XXXAL HARDCODED HEAD/TAIL
    pTar = pDiffN_al;
  else
    % get target value for shape
    pTar = shapeGt('inverse',model,pCur,bboxes); % pCur: absolute. pTar: normalized
    pTar = shapeGt('compose',model,pTar,pGt,bboxes); % pTar: normalized
  end
  
  if numel(ftrPrmRadiusOrig)>1
    ftrPrm.radius = ftrPrmRadiusOrig(min(t,numel(ftrPrmRadiusOrig)));
  end
  
  % XXXAL understand this codepath and best way to specify it
  if false %tfFidsSpeced && t > 10,
    
    % added by KB: only update some of the outputs
    % AL: meanErrPerPt created/updated below (note t>10 condition above)
    fidupdate = randsample(model.nfids,1,true,meanErrPerPt/sum(meanErrPerPt));
    %fidupdate = randint2(1,1,[1 model.nfids]);
    disp(fidupdate);
    %fidsupdate = sort([fidupdate,ftrPrm.neighbors{fidupdate}]);
    ftrPrm.fids = fidsupdate;
    ftrPos = shapeGt('ftrsGenDup2',model,ftrPrm);
    [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup2',...
      model,pCur,Is,ftrPos,...
      imgIds,pStar,posInit,regPrm.occlPrm,Prm3D);
    regPrm.ftrPrm=ftrPrm;
    pTarCurr = reshape(pTar,[N,model.nfids,model.d]);
    pTarCurr = pTarCurr(:,fidsupdate,:);
    pTarCurr = reshape(pTarCurr,[N,numel(fidsupdate)*model.d]);
    [regInfo0,pDel0] = regTrain(ftrs,pTarCurr,regPrm);
    regInfo = regInfo0;
    for i = 1:numel(regInfo0),
      ysFern = zeros(size(regInfo0{i}.ysFern,1),model.nfids,model.d);
      ysFernCurr = reshape(regInfo0{i}.ysFern,[size(regInfo0{i}.ysFern,1),numel(fidsupdate),model.d]);
      ysFern(:,fidsupdate,:) = ysFernCurr;
      ysFern = reshape(ysFern,[size(regInfo0{i}.ysFern,1),D]);
      regInfo{i}.ysFern = ysFern;
    end
    pDel = zeros([N,model.nfids,model.d]);
    pDelCurr = reshape(pDel0,[N,numel(fidsupdate),model.d]);
    pDel(:,fidsupdate,:) = pDelCurr;
    pDel = reshape(pDel,[N,D]);
    
  else
    switch ftrPrm.type
      case {'kborig_hack'}
        ftrPos = shapeGt('ftrsGenKBOrig',model,ftrPrm);
        ftrs = shapeGt('ftrsCompKBOrig',...
          model,pCur,Is,ftrPos,...
          imgIds,pStar,bbs,regPrm.occlPrm);
      case {'1lm' '2lm' '2lmdiff'}
        ftrPos = shapeGt('ftrsGenDup2',model,ftrPrm);
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup2',...
          model,pCur,Is,ftrPos,...
          imgIds,pStar,bbs,regPrm.occlPrm);
      case {3 4}
        assert(false,'Unsupported new Is');
        ftrPos = shapeGt('ftrsGenDup',model,ftrPrm);
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup',...
          model,pCur,Is,ftrPos,...
          imgIds,pStar,bbs,regPrm.occlPrm);
      otherwise
        assert(false,'Unsupported new Is');
        ftrPos = shapeGt('ftrsGenIm',model,pStar,ftrPrm);
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompIm',...
          model,pCur,Is,ftrPos,...
          imgIds,pStar,bbs,regPrm.occlPrm);
    end
    
    %Regress
    regPrm.ftrPrm = ftrPrm;
    [regInfo,pDel] = regTrain(ftrs,pTar,regPrm);
  end
  
  % Apply pDel
  if regPrm.USE_AL_CORRECTION
    pCur = Shape.applyRIDiff(pCurN_al,pDel,1,3); %XXXAL HARDCODED HEAD/TAIL
    pCur = shapeGt('reprojectPose',model,pCur,bboxes);
  else
    pCur = shapeGt('compose',model,pDel,pCur,bboxes);
    pCur = shapeGt('reprojectPose',model,pCur,bboxes);
  end
  
  %assert(size(pCur,1)==N1); % AL 20160314 this is not true, pAll holds
  %just first replicate
  pAll(:,:,t+1) = pCur(1:N1,:);
  %loss scores
  [errPerEx,errPerPt] = shapeGt('dist',model,pCur,pGt);
  meanErrPerPt = mean(errPerPt,1);
  loss = mean(errPerEx);
  % store result
  regs(t).regInfo = regInfo;
  regs(t).ftrPos = ftrPos;
  
  if verbose
    msg = tStatus(tStart,t,T);
    fprintf(['  t=%i/%i       loss=%f     ' msg],t,T,loss);
  end
  
  if loss<1e-5
    T=t; 
    break; 
  end
end

ftrPrm.radius = ftrPrmRadiusOrig;

% create output structure
regs = regs(1:T); 
pAll = pAll(:,:,1:T+1);
regModel = struct('model',model,'pStar',pStar,...
  'pDstr',pGt(1:N1,:),'T',T,'regs',regs);
if ~strcmp(model.name,'ellipse')
  regModel.pGtN = pGtN(1:N1,:); 
end

% Compute precision recall curve for occlusion detection and find
% desired occlusion detection performance (default=90% precision)
if(strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF1'))
  assert(false,'AL');
  
  nfids=D/3;
  occlGt=pGt(:,(nfids*2)+1:end);
  op=pCur(:,(nfids*2)+1:end);
  indO=find(occlGt==1);
  
  th=0:.01:1;
  prec=zeros(length(th),1);
  recall=zeros(length(th),1);
  for i=1:length(th)
    indPO=find(op>th(i));
    prec(i)=length(find(occlGt(indPO)==1))/numel(indPO);
    recall(i)=length(find(op(indO)>th(i)))/numel(indO);
  end
  %precision around 90% (or closest)
  pos=find(prec>=0.9);
  if(~isempty(pos)),pos=pos(1);
  else [~,pos]=max(prec);
  end
  %maximum f1score
  % f1score=(2*prec.*recall)./(prec+recall);
  % [~,pos]=max(f1score);
  regModel.th=th(pos);
end
end

function msg = tStatus(tStart,t,T)
elptime = etime(clock,tStart);
fracDone = max( t/T, .00001 );
esttime = elptime/fracDone - elptime;
if( elptime/fracDone < 600 )
  elptimeS  = num2str(elptime,'%.1f');
  esttimeS  = num2str(esttime,'%.1f');
  timetypeS = 's';
else
  elptimeS  = num2str(elptime/60,'%.1f');
  esttimeS  = num2str(esttime/60,'%.1f');
  timetypeS = 'm';
end
msg = ['[elapsed=' elptimeS timetypeS ...
  ' / remaining~=' esttimeS timetypeS ']\n' ];
end
