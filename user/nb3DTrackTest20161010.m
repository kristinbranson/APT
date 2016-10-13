%%
ROOTDIR = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling';
%assert(strcmp(pwd,ROOTDIR));

LBL = 'romainJun22NewLabels.lbl';
LBL = fullfile(ROOTDIR,LBL);

CRIG = 'crig2Optimized_calibjun2916_roiTrackingJun22_20160810_2.mat';
CRIG = fullfile(ROOTDIR,CRIG);

NVIEW = 3;
NREALPT = 57/3;

%%
lbl = load(LBL,'-mat');
crig2 = load(CRIG,'-mat');
crig2 = crig2.crig2Mod;

%%
lpos = lbl.labeledpos{1};
lpostag = lbl.labeledpostag{1};
nfrm = size(lpos,3);
lpos = reshape(lpos,[NREALPT NVIEW 2 nfrm]);
lpostag = reshape(lpostag,[NREALPT NVIEW nfrm]);

%% Generate MD table
% 
% Fields
% frm
% npts2VwLblNO. scalar, number of pts that have >= 2 views labeled (nonOcc)
% ipts2VwLblNO. vector with npts2VwLblNO els. pt indices.
% iVwsLblNO. cell vector with npts2VwLblNO els. view indices for each pt in
%   ipts2VwLblNO.
frm = nan(0,1);
npts2VwLblNO = nan(0,1);
ipts2VwLblNO = cell(0,1);
iVwsLblNO = cell(0,1);
iVwsLblNOCode = cell(0,1);
% yL = nan(0,2); % GT pts: (row,col) cropped coords in L view
% yR = nan(0,2);
% yB = nan(0,2);
VIEWCODES = 'lrb';
for f=1:nfrm
  
  iVwLbledNonOccPt = cell(NREALPT,1);
  tf2VwLbledNonOcc = false(NREALPT,1);
  for ipt = 1:NREALPT
    lposptfrm = squeeze(lpos(ipt,:,:,f));
    ltagptfrm = squeeze(lpostag(ipt,:,f));
    ltagptfrm = ltagptfrm(:);
    assert(isequal(size(lposptfrm),[NVIEW 2]));
    assert(isequal(size(ltagptfrm),[NVIEW 1]));
    tfVwLbled = ~any(isnan(lposptfrm),2);
    tfVwNotOcc = cellfun(@isempty,ltagptfrm);    
    iVwLbledNonOccPt{ipt} = find(tfVwLbled & tfVwNotOcc);      
    tf2VwLbledNonOcc(ipt) = numel(iVwLbledNonOccPt{ipt})>=2;
  end

  if any(tf2VwLbledNonOcc)
    frm(end+1,1) = f;
    npts2VwLblNO(end+1,1) = nnz(tf2VwLbledNonOcc);
    ipts2VwLblNO{end+1,1} = find(tf2VwLbledNonOcc);
    iVwsLblNO{end+1,1} = iVwLbledNonOccPt(tf2VwLbledNonOcc);
    iVwsLblNOCode{end+1,1} = cellfun(@(x)VIEWCODES(x),...
      iVwLbledNonOccPt(tf2VwLbledNonOcc),'uni',0);
  end
  
%   tf2VwLbledNonOcc(end+1,1) = nnz(tfVwLbled & tfVwNotOcc)>=2;
%   yL(end+1,:) = lpos(ipt,1,[2 1],f);
%   yR(end+1,:) = lpos(ipt,2,[2 1],f);
%   yB(end+1,:) = lpos(ipt,3,[2 1],f);  
end

nPtsLRCode = cellfun(@(x)nnz(strcmp('lr',x)),iVwsLblNOCode);
tFrmPts = table(frm,npts2VwLblNO,ipts2VwLblNO,iVwsLblNO,iVwsLblNOCode,nPtsLRCode);
% nGood = numel(iPtGood);
% fprintf('Found %d labeled pts.\n',nGood);

%% 
for i=1:size(tFrmPts,1)
  codes = tFrmPts.iVwsLblNOCode{i};
  for iV = 1:numel(iVws)
    if strcmp(codes{iV},'lr')
      fprintf(1,'%d: %d\n',tFrmPts.frm(i),tFrmPts.ipts2VwLblNO{i}(iV));
    end
  end
end


%% Add reconstruct/err stats
%
% For points labeled in all three views ('lrb'):
%  * Use each viewpair to recon/project in 3rd view and compute error.
%  * Recon 3D pt.
% For all points labeled in only two views:
%  * Recon 3D pt.
%
% New fields:
%  * X [3x19]. 3D recon pt in certain frame, say 'l'.
%  * errReconL. [npts2VwlblNOx1]. For points with 'lrb'. L2 err in L view (recon vs gt)
%  * errReconR. etc
%  * errReconB.

t19 = tFrmPts(tFrmPts.npts2VwLblNO==19,:);
nRows = size(t19,1);
XL = cell(nRows,1);
XLlr = cell(nRows,1);
errReconL = nan(nRows,19);
errReconR = nan(nRows,19);
errReconB = nan(nRows,19);
errReconL_lr = nan(nRows,19);
errReconR_lr = nan(nRows,19);
errReconB_lr = nan(nRows,19);
for iRow = 1:nRows
  frm = t19.frm(iRow);
  XLrow = nan(3,19);
  XLlrrow = nan(3,19);
  for iPt = 1:19
    code = t19.iVwsLblNOCode{iRow}{iPt};
    
    lposPt = squeeze(lpos(iPt,:,:,frm));
    assert(isequal(size(lposPt),[3 2]));
    yL = lposPt(1,[2 1]);
    yR = lposPt(2,[2 1]);
    yB = lposPt(3,[2 1]);
    switch code
      case 'lr'
        %assert(all(isnan(yB(:))));
        XLrow(:,iPt) = crig2.stereoTriangulateCropped(yL,yR,'L','R');        
      case 'lb'
        %assert(all(isnan(yR(:))));
        XLrow(:,iPt) = crig2.stereoTriangulateLB(yL,yB);        
      case 'rb'
        %assert(all(isnan(yL(:))));
        XBbr = crig2.stereoTriangulateBR(yB,yR);
        XLrow(:,iPt) = crig2.camxform(XBbr,'bl');
      case 'lrb'
        [~,~,~,...
          errReconL(iRow,iPt),errReconR(iRow,iPt),errReconB(iRow,iPt),...
          ~,~,~,...
          XLrow(:,iPt),~,~,...
          errReconL_lr(iRow,iPt),errReconR_lr(iRow,iPt),errReconB_lr(iRow,iPt),...
          XLlrrow(:,iPt)] = ...
            crig2.calibRoundTripFull(yL,yR,yB);
    end
  end
  XL{iRow} = XLrow;
  XLlr{iRow} = XLlrrow;
end

tAug = table(XL,errReconL,errReconR,errReconB,...
  errReconL_lr,errReconR_lr,errReconB_lr,XLlr);
t19aug = [t19 tAug];

%% Make an expanded frm/pt err browsing table
s19expanded = struct(...
  'frm',cell(0,1),...
  'ipt',[],...
  'code',[],...
  'XL',[],...
  'errReconL',[],...
  'errReconR',[],...
  'errReconB',[],...
  'errReconL_lr',[],...
  'errReconR_lr',[],...
  'errReconB_lr',[],...
  'XLlr',[]);
%%
nRows = size(t19aug,1);
ERRFLDS = {'errReconL' 'errReconR' 'errReconB' 'errReconL_lr' 'errReconR_lr' 'errReconB_lr'};
for iRow=1:nRows
  assert(t19aug.npts2VwLblNO(iRow)==19);
  assert(isequal(t19aug.ipts2VwLblNO{iRow},(1:19)'));
  frm = t19aug.frm(iRow);
  for iPt = 1:19
    s19expanded(end+1,1).frm = frm;
    s19expanded(end,1).ipt = iPt;
    s19expanded(end,1).code = t19aug.iVwsLblNOCode{iRow}{iPt};
    s19expanded(end,1).XL = t19aug.XL{iRow}(:,iPt)';
    s19expanded(end,1).XLlr = t19aug.XLlr{iRow}(:,iPt)';
    for f=ERRFLDS,f=f{1}; %#ok<FXSET>
      s19expanded(end,1).(f) = t19aug.(f)(iRow,iPt);
    end
  end
end
t19expanded = struct2table(s19expanded);
t19expandedLRB = t19expanded(strcmp(t19expanded.code,'lrb'),:);
%% 
hFig = figure('windowstyle','docked');
for f = ERRFLDS,f=f{1};
  clf(hFig);
  z = t19expandedLRB.(f);
  hist(z,50);
  iptsbig = unique(t19expandedLRB.ipt(z>20));
  fprintf('\n%s:\n',f);
  disp(iptsbig);
  input('hk');
end

%%
t19expandedLRB = t19expandedLRB(t19expandedLRB.ipt~=19,:);
%% Browse original/recon labels for given frame/pt
% CONC: first one is mislabel in side view. second one is mislabel in
% bottom view. 
% IDEA: just recon from two pts that lead to most consistency?
%tRow = [1711 2032]; bad errReconR
tRow = [120 1711 1889 2289];
TROWIDX = 3;
frame = t19expandedLRB.frm(tRow);
lObj.setFrame(frame(TROWIDX));
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
X = cell(1,3);
Xlr = cell(1,3);
X{1} = t19expandedLRB.XL(tRow(TROWIDX),:)';
X{2} = crig2.camxform(X{1},'lr');
X{3} = crig2.camxform(X{1},'lb');
Xlr{1} = t19expandedLRB.XLlr(tRow(TROWIDX),:)';
Xlr{2} = crig2.camxform(Xlr{1},'lr');
Xlr{3} = crig2.camxform(Xlr{1},'lb');
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  [r,c] = crig2.projectCPR(Xlr{iAx},iAx);
  hLine(iAx) = plot(ax,c,r,'wx','markersize',8);
end

%%
hold(ax,'on');
clrs = parula(19);
for iF=1:numel(frms)
  lposF = squeeze(lpos(:,3,:,frms(iF)));
  for iPt=1:19
    plot(ax,lposF(iPt,1),lposF(iPt,2),'o','Color',clrs(iPt,:));
  end
  input(num2str(iF));
end

%% gen I, bboxes, pGT

% take codes 'lrb' 'lb' 'br'
tfNoLR = t19aug.nPtsLRCode==0;
fprintf(1,'%d/%d rows have at least one ''lr'' code. Taking remaining %d rows.\n',...
  nnz(~tfNoLR),size(t19aug,1),nnz(tfNoLR));
t19AugNoLR = t19aug(tfNoLR,:);

tbl = t19AugNoLR;

codes = cat(1,tbl.iVwsLblNOCode{:});
codesUn = unique(codes);
codesUnCnt = cellfun(@(x)nnz(strcmp(x,codes)),codesUn);
fprintf(1,'Distro of codes:\n');
[codesUn num2cell(codesUnCnt)]

% accum pGT
% NOTE: for p-vectors or shapes, there are two flavors:
% * "concatenated-projected", ie you take the projected labels and
% concatenate. numel here is npts x nView x 2, raster order is pt, view,
% coord (x vs y).
% * absolute/3d, numel here is npts x 3, raster order is pt, coord (x vs y
% vs z).

nRows = size(tbl,1);
pGT = nan(nRows,19*3);
for i=1:nRows
  x = tbl.XL{i}; % 3x19
  x = x';
  szassert(x,[19 3]);
  pGT(i,:) = x(:);
end

% I
MOVDIR = 'F:\Dropbox\MultiViewFlyLegTracking\trackingJun22-11-02';
MOVS = {
  'bias_video_cam_0_date_2016_06_22_time_11_02_02_v001.avi'
  'bias_video_cam_1_date_2016_06_22_time_11_02_13_v001.avi'
  'bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi'
  };
movsFull = fullfile(MOVDIR,MOVS);
nView = 3;
assert(numel(movsFull)==nView);
for iView=1:nView
  mr(iView) = MovieReader();
  mr(iView).open(movsFull{iView});
  mr(iView).forceGrayscale = true;
end
I = cell(nRows,nView);
for iRow=1:nRows
  frm = tbl.frm(iRow);
  for iView=1:nView
    I{iRow,iView} = mr(iView).readframe(frm);
  end
  if mod(iRow,10)==0
    fprintf(1,'Read row %d\n',iRow);
  end
end

%% viz pGT in 3d
pGT2 = reshape(pGT_1_7_13,nRows,3,3);
clrs = parula(3);
hFig = figure('windowstyle','docked');
ax = axes;
hold(ax,'on');
for iRow = 1:nRows
  for iPt=1:3
    x = pGT2(iRow,iPt,1);
    y = pGT2(iRow,iPt,2);
    z = pGT2(iRow,iPt,3);
    plot3(ax,x,y,z,'o','MarkerFaceColor',clrs(iPt,:));
    text(x,y,z,num2str(iPt),'parent',ax,'Color',[0 0 0],'fontsize',12);
  end
end
ax.XLim = [bboxes(1) bboxes(1)+bboxes(4)];
ax.YLim = [bboxes(2) bboxes(2)+bboxes(5)];
ax.ZLim = [bboxes(3) bboxes(3)+bboxes(6)];

%% bboxes
pGT2mins = nan(1,3);
pGT2maxs = nan(1,3);
for i=1:3
  x = pGT2(:,:,i); % x-, y-, or z-coords for all rows, pts
  pGT2mins(i) = min(x(:));
  pGT2maxs(i) = max(x(:));
end
dels = pGT2maxs-pGT2mins;
% pad by 50% in every dir
pads = dels/2;
widths = 2*dels; % del (shapes footprint) + 2*pads (one on each side)
bboxes = [pGT2mins-pads widths];

[pGT2mins; pGT2maxs; dels] 
bboxes

%% pGT for 1-7-13
iPts_1_7_13 = [1 7 13];
pGT_1_7_13 = pGT(:,[iPts_1_7_13 iPts_1_7_13+19 iPts_1_7_13+38]);


%%
PARAMFILE = 'f:\romain\tp@3pts.yaml';
sPrm = ReadYaml(PARAMFILE);
sPrm.Model.nviews = 3;
sPrm.Model.Prm3D.iViewBase = 1;
sPrm.Model.Prm3D.calrig = crig2;
rc = RegressorCascade(sPrm);
rc.init();

%%
N = size(I,1);
pAll = rc.trainWithRandInit(I,repmat(bboxes,N,1),pGT_1_7_13);
pAll = reshape(pAll,197,50,9,31);

%% Browse trained replicates
pIidx = repmat((1:197)',50,1); % labels rows of pAll; indices into rows of tbl, I
TROWIDX = 1;
frame = tbl.frm(TROWIDX);
lObj.setFrame(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
deleteValidHandles(hLine);
hLine = gobjects(3,3);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 1 0;0 1 0];
  for iPt = 1:3
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

pRepTrow = squeeze(pAll(TROWIDX,:,:,:));
szassert(pRepTrow,[50 9 31]);

for t=1:31
  pRep = pRepTrow(:,:,t);
  pRep = reshape(pRep,50,3,3); % (iRep,iPt,iDim)
  for iVw=1:3
    for iPt=1:3
      X = squeeze(pRep(:,iPt,:)); % [50x3]
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('t=%d',t));
end

%% Propagate on labeled, nontraining data

% find all labeled frames not in tbl
frmTest = setdiff(tFrmPts.frm,tbl.frm);

[ITest,tblTest] = Labeler.lblCompileContentsRaw(...
  lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,1,{frmTest},...
  'hWB',waitbar(0));
% NOTE: tblTest.p is projected/concatenated
%%
nTest = size(ITest,1);
[p_t,pIidx] = rc.propagateRandInit(ITest,repmat(bboxes,nTest,1),sPrm.TestInit);
p_t = reshape(p_t,nTest,50,9,31);

%% Browse propagated replicates
TESTROWIDX = 155;
frame = tblTest.frm(TESTROWIDX);
lObj.setFrame(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidHandles(hLine);
end
hLine = gobjects(3,3);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 1 0;0 1 0];
  for iPt = 1:3
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

pRepTrow = squeeze(p_t(TESTROWIDX,:,:,:));
szassert(pRepTrow,[50 9 31]);

for t=1:31
  pRep = pRepTrow(:,:,t);
  pRep = reshape(pRep,50,3,3); % (iRep,iPt,iDim)
  for iVw=1:3
    for iPt=1:3
      X = squeeze(pRep(:,iPt,:)); % [50x3]
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('t=%d',t));
end

%% PRUNE PROPAGATED REPLICATES
trkD = rc.prmModel.D;
Tp1 = rc.nMajor+1;
nTestAug = sPrm.TestInit.Nrep;
pTstT = reshape(p_t,[nTest nTestAug trkD Tp1]);

%% Select best preds for each time
pTstTRed = nan(nTest,trkD,Tp1);
assert(sPrm.Prune.prune==1);
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
  pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,sPrm.Model,sPrm.Prune);
end
pTstTRedFinalT = pTstTRed(:,:,end);

%% Browse test frames
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidHandles(hLine);
end
hLine = gobjects(3,3);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 1 0;0 1 0];
  for iPt = 1:3
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',30,...
      'Color',clrs(iPt,:));
  end
end

pTstTRedFinalT = reshape(pTstTRedFinalT,274,3,3);
for iF=1:numel(frmTest)
  f = frmTest(iF);
  lObj.setFrame(f);

  pTstBest = squeeze(pTstTRedFinalT(iF,:,:));
  for iVw=1:3
    for iPt=1:3
      X = pTstBest(iPt,:);
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('frame=%d',f));
end