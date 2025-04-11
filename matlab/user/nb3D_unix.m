%% Agg all TD to form tMFP
TDFILES = {
  'oct2916-1420Romain_trnData_20161221T122734.mat'
  'romainJun22NewLabels_trnData_20161221T115407.mat'
  'sep1316-1606Romain_trnData_20161221T115759.mat'
  'sep1616-1531Romain_trnData_20161221T121140.mat'};
NTD = numel(TDFILES);

EXPNAMES = {
  'oct29'
  'jun22'
  'sep13'
  'sep16'};

EXPDIRS = {
  'f:\Dropbox\MultiViewFlyLegTracking\oct2916'
  'f:\Dropbox\MultiViewFlyLegTracking\trackingJun22-11-02'
  'f:\Dropbox\MultiViewFlyLegTracking\sep1316'
  'f:\Dropbox\MultiViewFlyLegTracking\sep1616'
  };

assert(isequal(NTD,numel(EXPNAMES),numel(EXPDIRS)));

tdifo = cellfun(@load,TDFILES);
tFPcell = cell(NTD,1);
for iTD=1:NTD
  tFP = tdifo(iTD).tFPtrn;
  nrow = size(tFP,1);
  tTmp = table(repmat(EXPNAMES(iTD),nrow,1),repmat(EXPDIRS(iTD),nrow,1),'VariableNames',{'expID' 'exp'});
  tFPcell{iTD} = [tTmp tFP];
end

%% Shape checks.

nview = 3;
phi2D = cell(NTD,1); % Each el is [nrows x nview]
exts = cell(NTD,1); % each el is [nrowsx3]
grp = cell(NTD,1); % grp vec
for iTD = 1:NTD
  tFP = tFPcell{iTD};
  nrow = size(tFP,1);
  p = tFP.p;
  p = reshape(p,[nrow 18 3 2]); % row, ipt, ivw, xy
  
  phi2D{iTD} = nan(nrow,nview);
  exts{iTD} = nan(nrow,3);  
  grp{iTD} = repmat(iTD,nrow,1);
  for ivw=1:nview
    % 1. Check 2D fly orientations across/within datasets. Is it consistent?
    xyVw = squeeze(p(:,:,ivw,:));
    szassert(xyVw,[nrow 18 2]);
    for irow=1:nrow
      [~,~,~,phi2D{iTD}(irow,ivw)] = RF.flyOrientation2D(squeeze(xyVw(irow,:,:)));
    end
    
    % 2. Check 2D fly dimensions.
  end

  % 3. Check 3D fly EXTENTS.
  for irow=1:nrow
    XL = tFP.XL{irow}';
    szassert(XL,[18 3]);
    maxs = max(XL,[],1);
    mins = min(XL,[],1);
    exts{iTD}(irow,:) = maxs-mins;
  end
end

NBIN = 80;
edges = linspace(-pi,pi,NBIN);
ctrs = (edges(1:end-1)+edges(2:end))/2;
phiDists = cellfun(@(x)histcounts(x(:,3),edges)/numel(x(:,3)),phi2D,'uni',0);
phiDists = cat(1,phiDists{:});

figure('windowstyle','docked');
ax = axes;
plot(ax,ctrs,phiDists','linewidth',2);
grid on;
legend(EXPNAMES);
title('2D phi in bottom view','fontweight','bold');
xlabel('phi (rad)','fontweight','bold');
  
NBIN = 20;
DIMNAMES = {'x' 'y' 'z'};
extscat = cat(1,exts{:});
minexts = min(extscat,[],1);
maxexts = max(extscat,[],1);
szassert(minexts,[1 3]); % minimum extent x, y, z
szassert(maxexts,[1 3]);
edges = arrayfun(@(x1,x2)linspace(x1,x2,NBIN),minexts,maxexts,'uni',0);
ctrs = cellfun(@(x) (x(1:end-1)+x(2:end))/2, edges,'uni',0);
% edges/ctrs: [1x3], indexed by dim (x/y/z)
figure('windowstyle','docked');
ax = createsubplots(2,2,0.08);
for iDim=1:3
  extCountsDim = cellfun(@(x)histcounts(x(:,iDim),edges{iDim})/numel(x(:,iDim)),exts,'uni',0);
  extCountsDim = cat(1,extCountsDim{:})';
  szassert(extCountsDim,[NBIN-1 NTD]);
  plot(ax(iDim),ctrs{iDim},extCountsDim,'linewidth',2);
  grid(ax(iDim),'on');
  legend(ax(iDim),EXPNAMES);
  title(ax(iDim),sprintf('Shape extents: %s',DIMNAMES{iDim}),'fontweight','bold');
end


%% Try single global 3D bboxes. Compare TDsets.

XLmats = cellfun(@(x) cat(3,x.XL{:}), tFPcell, 'uni',0);
XLmatsCat = cat(3,XLmats{:});
szassert(XLmatsCat,[3 18 size(XLmatsCat,3)]);
XLmatsCat = permute(XLmatsCat,[3 2 1]);
bboxes = RF.generateBBoxes(XLmatsCat);

% plot shapes in bboxes by set
figure('windowstyle','docked');
axs = createsubplots(2,2,0.08);
axs = reshape(axs,[2 2]);
for iTD=1:NTD  
  ax = axs(iTD);
  RF.plotShapes3D(ax,permute(XLmats{iTD},[3 2 1]));
  ax.XLim = [bboxes(1) bboxes(1)+bboxes(4)];
  ax.YLim = [bboxes(2) bboxes(2)+bboxes(5)];
  ax.ZLim = [bboxes(3) bboxes(3)+bboxes(6)];
  grid(ax,'on');
  title(ax,EXPNAMES{iTD});
  xlabel(ax,'x','fontweight','bold');
  ylabel(ax,'y','fontweight','bold');
  zlabel(ax,'z','fontweight','bold');  
end

%% Change views

% X
arrayfun(@(x)view(x,0,90),axs);
% Y
arrayfun(@(x)view(x,0,0),axs);
% Z
arrayfun(@(x)view(x,90,0),axs);


%% Think about how abs/norm shapes work in training/propagation.

%% Compute TD-specific bboxes.
XLmats = cellfun(@(x) cat(3,x.XL{:}), tFPcell, 'uni',0);
XLmats = cellfun(@(x) permute(x,[3 2 1]), XLmats, 'uni',0);
bboxesExp = cellfun(@RF.generateBBoxes,XLmats,'uni',0);
bboxesExp = cat(1,bboxesExp{:});

%% Viz TD-specific bboxes with pGT.
figure('windowstyle','docked');
axs = createsubplots(2,2,0.08);
axs = reshape(axs,[2 2]);
for iTD=1:NTD  
  ax = axs(iTD);
  RF.plotShapes3D(ax,XLmats{iTD});
  ax.XLim = [bboxesExp(iTD,1) bboxesExp(iTD,1)+bboxesExp(iTD,4)];
  ax.YLim = [bboxesExp(iTD,2) bboxesExp(iTD,2)+bboxesExp(iTD,5)];
  ax.ZLim = [bboxesExp(iTD,3) bboxesExp(iTD,3)+bboxesExp(iTD,6)];
  ax.DataAspectRatio = [1 1 1];
  grid(ax,'on');
  title(ax,EXPNAMES{iTD});
  xlabel(ax,'x','fontweight','bold');
  ylabel(ax,'y','fontweight','bold');
  zlabel(ax,'z','fontweight','bold');  
end
%% Change views
% X
arrayfun(@(x)view(x,0,90),axs);
% Y
arrayfun(@(x)view(x,0,0),axs);
% Z
arrayfun(@(x)view(x,90,0),axs);

%% Viz pGTN across TDs
figure('windowstyle','docked');
axs = createsubplots(2,2,0.08);
axs = reshape(axs,[2 2]);
mdl = struct('D',18*3,'d',3,'nfids',18);
pNTD = cell(NTD,1);
for iTD=1:NTD  
  ax = axs(iTD);
  xl = XLmats{iTD};
  n = size(xl,1);
  szassert(xl,[n 18 3]);
  p = reshape(xl,[n 18*3]);
  pN = shapeGt('projectPose',mdl,p,repmat(bboxesExp(iTD,:),n,1));
  pNTD{iTD} = pN;
  xlN = reshape(pN,[n 18 3]);
  
  RF.plotShapes3D(ax,xlN);
  ax.XLim = [-1 1];
  ax.YLim = [-1 1];
  ax.ZLim = [-1 1];
  ax.DataAspectRatio = [1 1 1];
  grid(ax,'on');
  title(ax,sprintf('%s: %d normalized shapes',EXPNAMES{iTD},n));
  xlabel(ax,'x','fontweight','bold');
  ylabel(ax,'y','fontweight','bold');
  zlabel(ax,'z','fontweight','bold');
end
%% Change views
% X
arrayfun(@(x)view(x,0,90),axs);
% Y
arrayfun(@(x)view(x,0,0),axs);
% Z
arrayfun(@(x)view(x,90,0),axs);

%% TrnDataSel (coords?)
pNTDcat = cat(1,pNTD{:});
tblP = table(pNTDcat,'VariableNames',{'p'});
[grps,ffd,ffdiTrl] = CPRData.ffTrnSet(tblP,[]);
hFig1 = CPRData.ffTrnSetSelect(tblP,grps,ffd,ffdiTrl);
% Grand total of 777/799 (97%) shapes selected for training.
%%
iTrlSel = ffdiTrl{1}(1:777);
tFPcat = cat(1,tFPcell{:});
tFPtrn = tFPcat(iTrlSel,:);
[tf,loc] = ismember(tFPtrn.expID,EXPNAMES);
assert(all(tf));
bboxesTrn = bboxesExp(loc,:);

%% check training shapes (normalized)
pGT3dTrn = cellfun(@(x)reshape(x',[1 18*3]),tFPtrn.XL,'uni',0);
pGT3dTrn = cat(1,pGT3dTrn{:});
pGT3dTrnN = shapeGt('projectPose',mdl,pGT3dTrn,bboxesTrn);
tblTmp = table(pGT3dTrnN,'VariableNames',{'p'});
[grps,ffd,ffdiTrl] = CPRData.ffTrnSet(tblTmp,[]);
hFig1 = CPRData.ffTrnSetSelect(tblTmp,grps,ffd,ffdiTrl);

%% no nans in p
nnz(isnan(pGT3dTrn))
nnz(isnan(pGT3dTrnN))

%%
save intTrain3DRes20170105T6pm.mat tFPtrn bboxesTrn;

%%
aviCell = cellfun(@(x)dir(fullfile(x,'*.avi')),EXPDIRS,'uni',0);
aviCell = cellfun(@(x,y)fullfile(x,{y.name}),EXPDIRS,aviCell,'uni',0);
for iTD=1:NTD
  avis = aviCell{iTD};
  assert(numel(avis)==3);
  assert(~isempty(regexp(avis{1},'_cam_0_','once')));
  assert(~isempty(regexp(avis{2},'_cam_1_','once')));
  assert(~isempty(regexp(avis{3},'_cam_2_','once')));
end

[tf,loc] = ismember(tFPtrn.exp,EXPDIRS);
assert(all(tf));
movSet = aviCell(loc);
tFPtrn = [tFPtrn table(movSet)];

%% Get I
I = MFTable.fetchImages(tFPtrn);

%% pp
td = CPRData(I,tFPtrn,bboxesTrn);

%%
%bpp = load('/groups/branson/home/leea30/rf/rfBlurPreProc.mat');
bpp = load('f:\romain\rfBlurPreProc.mat');
bpp = bpp.bpp;
%td.computeIpp([],[],[],'romain',bpp,'iTrl',1:td.N);
td.computeIpp([],[],[],'romain',bpp,'iTrl',1:60);

%% Get the crigFull
[tf,loc] = ismember(tFPtrn.expID,EXPNAMES);
assert(all(tf));
crigUn = [tdifo.crig2];
[EXPNAMES {crigUn.roiDir}']
crigAll = crigUn(loc);

%% montage I/pGT
GAMMA = .3;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);

Imontage = I;
Nmont = 3;
figure('windowstyle','docked');
axs = createsubplots(Nmont,3);
axs = reshape(axs,Nmont,3);
arrayfun(@(x)colormap(x,mgray2),axs);

nrow = size(tFPtrn,1);
randrows = randint2(1,Nmont,[1 nrow]);
for iMont=1:Nmont
  iRow = randrows(iMont);
  crRow = crigAll(iRow);
  for iView=1:3
    ax = axs(iMont,iView);
    axes(ax);
    imagesc(Imontage{iRow,iView});
    colormap(ax,mgray2);
    axis(ax,'equal')
    
    if iView==1
      title(ax,num2str(iRow),'fontweight','bold');
    end
    if ~(iView==3 && iMont==1)
      ax.XTick = [];
      ax.YTick = [];
    end
  end
  
  X = cell(1,3);
  X{1} = tFPtrn.XL{iRow};
  szassert(X{1},[3 18]);
  X{2} = crRow.camxform(X{1},'lr');
  X{3} = crRow.camxform(X{1},'lb');
  
  MARKERS = {'o' 's' 'v'};
  COLORS = {[1 0 0] [0 1 0] [0 1 1] [1 1 0] [0 0 1] [1 0 1]};
  for leg=1:6
    for j=1:3
      ipt = leg+(j-1)*6;
      for iView=1:3
        ax = axs(iMont,iView);
        hold(ax,'on');
        Xtmp = X{iView}(:,ipt);
        [r,c] = crRow.projectCPR(Xtmp,iView);
        plot(ax,c,r,[MARKERS{j}],'markersize',8,'color',COLORS{leg},'markerfacecolor',COLORS{leg});
      end
    end
  end
end

%% Train init/params
PARAMFILE = 'f:\romain\20161214MoreData\tp@18pts@3d_jitterForeMore@2lmdiff.yaml';
sPrm = yaml.ReadYaml(PARAMFILE);
sPrm.Model.nviews = 3;
sPrm.Model.Prm3D.iViewBase = 1;
rc = RegressorCascade(sPrm);
rc.init();

%% TRAIN
trnData = load('trnData3D_20170106.mat');
%%
% Ntrk = td.N;
% [IsTrk,nChan] = td.getCombinedIs(1:td.N);
% bboxes = td.bboxes;
pGT3dTrn = cellfun(@(x)reshape(x',[1 18*3]),trnData.tFPtrn(1:60,:).XL,'uni',0);
pGT3dTrn = cat(1,pGT3dTrn{:});

[pAll,pIidx,p0,p0info] = rc.trainWithRandInit(IsTrn,bboxes,pGT3dTrn,...
  'initUseFF',true,'calrig',trnData.crigAll(1:60));

pAll = reshape(pAll,Ntrk,50,18*3,rc.nMajor+1);

%% Browse p0
p0 = reshape(trkResFull.p0,[2e3 50 54]);
TESTROWIDX = 2e3; % Pick any training row to view convergence
NPTS = 18;
frame = trkResFull.tMFPtrack.frm(TESTROWIDX);
if exist('lObj','var')==0
  lObj = Labeler;
  lObj.projLoadGUI(LBL);
end
lObj.setFrameGUI(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end
hLine = gobjects(3,NPTS);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = {[1 0 0] [1 0 0] [1 0 0]; ...
          [1 1 0] [1 1 0] [1 1 0]; ...
          [0 1 0] [0 1 0] [0 1 0]; ...
          [0 1 1] [0 1 1] [0 1 1]; ...
          [0 0 1] [0 0 1] [0 0 1]; ...
          [1 0 1] [1 0 1] [1 0 1];};
  clrs = cat(1,clrs{:});
  mrks = {...
    '^' 's' 'o'; ...
    '^' 's' 'o'; ...
    '^' 's' 'o'; ...
    '^' 's' 'o'; ...
    '^' 's' 'o'; ...
    '^' 's' 'o';};
  mrks = mrks(:);
          
  for iPt = 1:NPTS
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',4,...
      'Color',clrs(iPt,:),...
      'Marker',mrks{iPt});
  end
end

p0Row = squeeze(p0(TESTROWIDX,:,:));
p0Row = reshape(p0Row,50,18,3); % (iRep,iPt,iDim)

%crTestRow = trnData.crigAll(TESTROWIDX);
crTestRow = trkResFull.crigTrack;
for iVw=1:3
  for iPt=1:18
    X = squeeze(p0Row(:,iPt,:)); % [50x3]
    Xvw = crTestRow.viewXformCPR(X',1,iVw); % iViewBase==1
    [r,c] = crTestRow.projectCPR(Xvw,iVw);
    
    h = hLine(iVw,iPt);
    set(h,'XData',c,'YData',r);
  end
end
  
%% Browse propagated replicates
TESTROWIDX = 222; % Pick any training row to view convergence
NPTS = 18;
frame = tFPtrn.frm(TESTROWIDX);
if exist('lObj','var')==0
  lObj = Labeler;
  lObj.projLoadGUI(LBL);
end
lObj.setFrameGUI(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end
hLine = gobjects(3,NPTS);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = {[1 0 0] [1 0 0] [1 0 0]; ...
          [1 1 0] [1 1 0] [1 1 0]; ...
          [0 1 0] [0 1 0] [0 1 0]; ...
          [0 1 1] [0 1 1] [0 1 1]; ...
          [0 0 1] [0 0 1] [0 0 1]; ...
          [1 0 1] [1 0 1] [1 0 1];};
  clrs = cat(1,clrs{:});
          
  for iPt = 1:NPTS
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

pRepTrow = squeeze(pAll(TESTROWIDX,:,:,:));
szassert(pRepTrow,[50 18*3 rc.nMajor+1]);

crTestRow = crigAll(TESTROWIDX);
for t=1:rc.nMajor+1
  pRep = pRepTrow(:,:,t);
  pRep = reshape(pRep,50,18,3); % (iRep,iPt,iDim)
  for iVw=1:3
    for iPt=1:18 
      X = squeeze(pRep(:,iPt,:)); % [50x3]
      Xvw = crTestRow.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crTestRow.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('t=%d',t));
end

%% TRACK Propagate on labeled, nontraining data
TRACKEXPDIR = '/groups/branson/home/leea30/rf/exps/trackingJun22-11-02';
FRMTEST = 1:2e3;

avisTrack = dir(fullfile(TRACKEXPDIR,'*.avi'));
avisTrack = fullfile(TRACKEXPDIR,{avisTrack.name});
assert(numel(avisTrack)==3);
assert(~isempty(regexp(avisTrack{1},'_cam_0_','once')));
assert(~isempty(regexp(avisTrack{2},'_cam_1_','once')));
assert(~isempty(regexp(avisTrack{3},'_cam_2_','once')));
movsetTrack = avisTrack(:)';

nFrmTest = numel(FRMTEST);
tMFPtrack = table(repmat({movsetTrack},nFrmTest,1),FRMTEST(:),nan(numel(FRMTEST),1),...
  'VariableNames',{'movSet' 'frm' 'p'});

Itrack = MFTable.fetchImages(tMFPtrack);

%% bboxes
trnData = load('trnData3D_20170106.mat');
assert(strcmp(trnData.tFPtrn.expID{end},'jun22'));
bboxesJun22 = trnData.bboxesTrn(end,:);
bboxesTrack = bboxesJun22;
crigTrack = trnData.crigAll(end);

%% pp
Ntrack = size(Itrack,1);
tdTrack = CPRData(Itrack,tMFPtrack,repmat(bboxesTrack,Ntrack,1));
%%
bpp = load('/groups/branson/home/leea30/rf/rfBlurPreProc.mat');
bpp = bpp.bpp;
tdTrack.computeIpp([],[],[],'romain',bpp,'iTrl',1:tdTrack.N);

%%
trnRes = load('trnRes3D_pp21_20170106.mat');
Ntrk = tdTrack.N;
[IsTrk,nChan] = tdTrack.getCombinedIs(1:tdTrack.N);

%%
rc = trnRes.rc;
[pAllTest,pIidxTest] = rc.propagateRandInit(IsTrk,repmat(bboxesTrack,Ntrk,1),...
  sPrm.TestInit,'calrig',repmat(crigTrack,Ntrack,1));
%%
pAllTestSingle = single(pAllTest);
save trackResAll_20170110.mat pAllTestSingle bboxesTrack Ntrk crigTrack sPrmTest tMFPtrack

%%
trkD = rc.prmModel.D;
Tp1 = rc.nMajor+1;
nTestAug = sPrm.TestInit.Nrep;
pTstT = reshape(pAllTest,Ntrack,nTestAug,trkD,Tp1);
pTstTRed = nan(Ntrack,trkD,Tp1);
assert(sPrm.Prune.prune==1);
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
  pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,sPrm.Model,sPrm.Prune);
end
pTstTRedFinalT = pTstTRed(:,:,end);
pTstTRedSingle = single(pTstTRed);
%%
save trackResPruned_20170110.mat pTstTRedSingle
