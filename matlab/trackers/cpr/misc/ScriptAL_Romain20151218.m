%% Path
APTPATH = 'c:\data.not.os\bitbucket\apt';
RCPR = 'c:\data.not.os\bitbucket\rcpr.kb';
JCTRAX = 'c:\data.not.os\bitbucket\jctrax';
PIOTR = 'c:\data.not.os\bitbucket\piotr_toolbox';

addpath(APTPATH);
APT.setpath;
addpath(RCPR);
addpath(fullfile(RCPR,'video_tracking'));
addpath(fullfile(RCPR,'misc'));
addpath(fullfile(JCTRAX,'misc'));
addpath(fullfile(JCTRAX,'filehandling'));
addpath(genpath(PIOTR));

%%
ROOTDIR = 'f:\cpr\data\romain\LabeledData';
MOVFILE = 'flyOnABoardApril15.avi';
LBLFILE = 'flyOnABoardApril15.lbl';
movFileFull = fullfile(ROOTDIR,MOVFILE);
lblFileFull = fullfile(ROOTDIR,LBLFILE);

%%
% doeq = false;
% loadH0 = false;
% cpr_type = 2;
% docacheims = false;
maxNTr = 10000;
radius = 100;
winrad = 50;
% docurate = false;

%% read in the training images
XBOUNDARY = 600;
YBOUNDARY = 275;
LBLS0 = [1:18 55];
LBLS1 = [19:36 56];
LBLS2 = [37:54 57];

mr = MovieReader();
mr.open(movFileFull);
lbls = load(lblFileFull,'-mat');
lbls = lbls.labeledpos{1}; % npts x 2 x nframes

[npts,d,~] = size(lbls);
D = 2*19; % d*npts;
frms = find(~isnan(lbls(1,1,:)));
nFrm = numel(frms);
IsTr = cell(nFrm,3); % IsTr{iF,i} is training image for frame frms(iF), index i (i in 1..3)
pGT0 = nan(nFrm,D,3); % pGT0(iF,:,i) is GT shape for frms(iF), index i

for iFrm = 1:nFrm  
  f = frms(iFrm);
  im = mr.readframe(f);
  assert(size(im,3)==3);
  im = rgb2gray(im);
  
  im0 = im(:,1:XBOUNDARY);
  im1 = im(1:YBOUNDARY,XBOUNDARY+1:end);
  im2 = im(YBOUNDARY+1:end,XBOUNDARY+1:end);
  IsTr{iFrm,1} = im0;
  IsTr{iFrm,2} = im1;
  IsTr{iFrm,3} = im2;
  
  lblsFrm = lbls(:,:,f);
  lblsFrm0 = lblsFrm(LBLS0,:);
  lblsFrm1 = lblsFrm(LBLS1,:);
  lblsFrm2 = lblsFrm(LBLS2,:);
  pGT0(iFrm,:,1) = Shape.xy2vec(lblsFrm0);
  lblsFrm1(:,1) = lblsFrm1(:,1)-XBOUNDARY;
  pGT0(iFrm,:,2) = Shape.xy2vec(lblsFrm1);
  lblsFrm2(:,1) = lblsFrm2(:,1)-XBOUNDARY;
  lblsFrm2(:,2) = lblsFrm2(:,2)-YBOUNDARY;
  pGT0(iFrm,:,3) = Shape.xy2vec(lblsFrm2);  
end
  
%%
outdir = sprintf('f:\\cpr\\data\\romain');
[tmp1,tmp2] = fileparts(outdir);
if exist(outdir,'dir')==0
  mkdir(tmp1,tmp2);
end

%% create/save training data
sz = cellfun(@(x)size(x'),IsTr,'uni',0);
bb = cellfun(@(x)[[1 1] x],sz,'uni',0);

for i = 1:3
  td = TrainData(IsTr(:,i),pGT0(:,:,i),cat(1,bb{:,i}));

  % partition data
  rng(7); % seed for determinism

  nTr = 80;
  assert(nTr<td.N);
  idxTrn = randSample(td.N,nTr);
  idxTst = setdiff(1:td.N,idxTrn);

  fprintf('Training set, %d/%d exps: %s...\n',numel(idxTrn),td.N,num2str(idxTrn(1:5)));
  fprintf('Test set, %d/%d exps: %s...\n',numel(idxTst),td.N,num2str(idxTst(1:5)));

  td.iTrn = idxTrn;
  td.iTst = idxTst;

  td.Name = ['reg' num2str(i)];
  TrainDataFile = fullfile(outdir,sprintf('td_%s_20160104.mat',td.Name));
  fprintf('Saving training data to: %s\n',TrainDataFile);
  save(TrainDataFile,'td','frms');
end
%% Visualize training set
td = load('td_reg2_20160104.mat'); 
td = td.td;
td.viz('fig',gcf,'nr',5,'nc',3,'labelpts',true);
  
%% Create/save parameters
tp = TrainParams;
tp.USE_AL_CORRECTION = 0;
tp.model_type = 'Romain';
tp.Name = 'reg_NOcorrect';
tp.model_nfids = 19;
paramsfile2 = fullfile(outdir,sprintf('tp_%s_%s.mat',tp.Name,datestr(now,'yyyymmdd')));
fprintf('Saving training params to: %s\n',paramsfile2);
save(paramsfile2,'tp');

%% Train on training set
TRAINNOTE = 'e78c';
TrainDataFile = 'f:\cpr\data\romain\td_reg2_20160104.mat';
td = load(TrainDataFile);
td = td.td;
trName = sprintf('%s__%s__%s__%s',td.Name,tp.Name,TRAINNOTE,datestr(now,'yyyymmddTHHMMSS'));
trainresfile = fullfile(outdir,trName);
fprintf('Training and saving results to: %s\n',trainresfile);
diary([trainresfile '.dry']);
[regModel,regPrm,prunePrm,phisPr,err] = train(TrainDataFile,paramsfile2,trainresfile);
diary off;

%% Test on test set
td = load(TrainDataFile);
td = td.td;
tr = load(trainresfile);
mdl = tr.regModel.model;
pGTTstN = shapeGt('projectPose',mdl,td.pGTTst,td.bboxesTst);
DOROTATE = false;
pIni = shapeGt('initTest',[],td.bboxesTst,mdl,[],pGTTstN,50,DOROTATE);
[~,~,~,~,pTstT] = test_rcpr([],td.bboxesTst,td.ITst,tr.regModel,tr.regPrm,tr.prunePrm,pIni);
pTstT = reshape(pTstT,[22 50 38 101]);
  
%% Select best preds for each time
[N,R,D,Tp1] = size(pTstT);
pTstTRed = nan(N,D,Tp1);
prunePrm = tr.prunePrm;
prunePrm.prune = 1;
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
  pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,tr.regPrm,prunePrm);
end

%% Visualize loss over time
[ds,dsAll] = shapeGt('dist',mdl,pTstTRed,td.pGTTst);
ds = squeeze(ds)';
dsmu = mean(ds,2);
logds = log(ds);
logdsmu = mean(logds,2);

figure('WindowStyle','docked');
hax = createsubplots(2,1,[.1 0;.1 .01],gcf);
x = 1:size(ds,1);
plot(hax(1),x,ds)
hold(hax(1),'on');
plot(hax(1),x,dsmu,'k','linewidth',2);
grid(hax(1),'on');
set(hax(1),'XTickLabel',[]);
ylabel(hax(1),'meandist from pred to gt (px)','interpreter','none');
tstr = sprintf('NTest=%d, numIter=%d, final mean ds = %.3f',N,Tp1,dsmu(end));
title(hax(1),tstr,'interpreter','none','fontweight','bold');
plot(hax(2),x,logds);
hold(hax(2),'on');
plot(hax(2),x,logdsmu,'k','linewidth',2);
grid(hax(2),'on');
ylabel(hax(2),'log(meandist) from pred to gt (px)','interpreter','none');

%% Viz: overall
figure(5);
Shape.vizDiff(td.ITst,td.pGTTst,pTstTRed(:,:,end),tr.regModel.model,...
  'fig',gcf,'nr',4,'nc',2,'idxs',4:11);

%%
figure(4);
iTrl = 3;
Shape.vizRepsOverTimeDensity(td.ITst,pTstT,iTrl,tr.regModel.model,'fig',gcf,'smoothsig',20);




%%
figure(4);
iTrl = 4;
Shape.vizRepsOverTime(td.ITst,pTstT,iTrl,tr.regModel.model,'fig',gcf,...
  'pGT',td.pGTTst,'regs',tr.regModel.regs,'nr',4,'nc',2);


%%
figure(3);
Shape.vizRepsOverTimeTracks(td.ITst,pTstT,iTrl,tr.regModel.model,'fig',gcf,...
  'nr',2,'nc',1);
%%
iTrl = 1;
figure(4);
Shape.vizRepsOverTimeDensity(td.ITst,pTstT,iTrl,tr.regModel.model,'fig',gcf);
%%
figure(5);
Shape.viz(td.ITst,pTst0,mdl,'fig',gcf,'nr',4,'nc',6,'idxs',1:22);

%%
figure(5);
Shape.viz(td.ITst,pTst0,mdl,'fig',gcf,'nr',4,'nc',6,'idxs',1:22);


%% Test set: visualize all features for a single trial/replicate/iteration
iTrl = 1;
iRT = 1;
t = 1;
mdl = tr.regModel.model;
hFig = figure('windowstyle','docked');
hax = createsubplots(1,2,.01);

imagesc(td.ITst{iTrl},'Parent',hax(1),[0,255]);
axis(hax(1),'image','off');
hold(hax(1),'on');
colormap gray;

reg = tr.regModel.regs(t);
p = reshape(pTstT(iTrl,iRT,:,t),1,mdl.D); % absolute shape for trl/rep/it

axes(hax(2));
shapeGt('draw',mdl,td.ITst{iTrl},p,'lw',20);

[xF,yF,infoF] = Features.compute2LM(reg.ftrPos.xs,p(1:mdl.nfids),p(mdl.nfids+1:end));
nfeat = size(xF,2);
fprintf(1,'%d features\n',nfeat);
for iF = 1:nfeat
  Features.visualize2LM(hax(1),xF,yF,infoF,1,iF);
  input(num2str(iF));
end

%% Test set: visualize selected (fern) features for a single trial/replicate over time/minitime
iTrl = 1;
iRT = 1;
mdl = tr.regModel.model;
hFig = figure('windowstyle','docked');
hax = createsubplots(1,2,.1);

imagesc(td.ITst{iTrl},'Parent',hax(1),[0,255]);
axis(hax(1),'image','off');
hold(hax(1),'on');
colormap gray;

hFids = [];
for t = 1:101  
  reg = tr.regModel.regs(t);
  p = reshape(pTstT(iTrl,iRT,:,t),1,mdl.D); % absolute shape for trl/rep/it
  im = td.ITst{iTrl};
  
  axes(hax(2));
  shapeGt('draw',mdl,im,p,'lw',20);
  title(num2str(t));

  [xF,yF,infoF] = Features.compute2LM(reg.ftrPos.xs,p(1:mdl.nfids),p(mdl.nfids+1:end));
  assert(isrow(xF));
  assert(isrow(yF));
  nMini = numel(reg.regInfo);
  for iMini = 1:nMini
    fids = reg.regInfo{iMini}.fids;
    nfids = size(fids,2);
    colors = jet(nfids);    
    
    deleteValidGraphicsHandles(hFids);
    for iFid = 1:nfids
      hFids(end+1) = plot(hax(1),xF(fids(1,iFid)),yF(fids(1,iFid)),'^','markerfacecolor',colors(iFid,:));
      hFids(end+1) = plot(hax(1),xF(fids(2,iFid)),yF(fids(2,iFid)),'o','markerfacecolor',colors(iFid,:));      
    end
    
    title(hax(1),sprintf('it %d mini %d\n',t,iMini));
    input('hk');
  end  
end
  


