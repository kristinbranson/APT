function trkppAssess(varargin)
% trkppAssess(varargin)
% Assess tracking via various metrics correlated to tracking err
% 
% resFile: string, results file
% PVs:
% - res. Results structure
% - resFile. If res not supplied, file to load for res
% - td. CPRData object
% - tdFile. If td is not supplied, file to load for td
% - tdiFile. 
% - tdiVarTrn.
% - tdiVarTst.
% - saveAssess. Filename to save assess results
% - savePlots. Filename to save figs

% RES = 'F:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\13@he@for_150730_02_002_07_v2@iTrn@lotsa1__13@he@for_150730_02_002_07_v2@iTstLbl__0225T1028\res.mat';
% TD = 'f:\cpr\data\jan\td@13@he@0217';
% TDI = 'f:\cpr\data\jan\tdI@13@for_150730_02_002_07_v2@0224';
% TDIVARTRN = 'iTrn';
% TDIVARTST = 'iTstLbl';

[res,resFile,td,tdFile,tdiFile,tdiVarTrn,tdiVarTst,saveAssess,savePlots] = ...
  myparse(varargin,...
  'res',[],...
  'resFile',[],...
  'td',[], ...
  'tdFile',[],...
  'tdiFile',[], ...
  'tdiVarTrn','iTrn',...
  'tdiVarTst','iTstLbl',...
  'saveAssess','',...
  'savePlots','');

% fprintf(1,'Loading res\n');
% res = load(resFile);
if ~isempty(res)
  fprintf(1,'Res supplied.\n');
else
  fprintf(1,'Loading res file %s...\n',resFile);
  res = load(resFile);
end

if ~isempty(td)
  fprintf(1,'TD supplied.\n');
else
  fprintf(1,'Loading TD file %s...\n',tdFile);
  td = load(tdFile);
  td = td.td;
end

tdi = load(tdiFile);
assert(all(isfield(tdi,{tdiVarTrn;tdiVarTst})));

resAssess = struct();

%%
pTrk = res.pTstTRed(:,:,end);
pGTTst = td.pGT(tdi.(tdiVarTst),:);
pGTTrn = td.pGT(tdi.(tdiVarTrn),:);
assert(isequal(size(pTrk),size(pGTTst)));
nTest = size(pTrk,1);
nTrn = size(pGTTrn,1);

%% distance from tracking to pGTTest
dTrk = Shape.distP(pGTTst,pTrk);
dTrk47 = dTrk(:,4:7);
dTrk47Av = mean(dTrk47,2);

%% distance from test/track to nearest trn

% XXX SEE JANPOSTPROC.minDistToTrainingSet

iTrns4iTsts = nan(nTest,1); % argmin(dist-to-trning set) for each iTst
dTrnMin = nan(nTest,1); % min distance to training set from each Test (GT) pt
dTrnMinTrk = nan(nTest,1); % min distance to training set from each Test (Tracked) pt
warnst = warning('off','Shape:distP');
for iTest = 1:nTest
  p = pGTTst(iTest,:);
  d = Shape.distP(pGTTrn,repmat(p,nTrn,1)); % [nTrnx7]
  d = mean(d(:,4:7),2);
  [dminDistToTrnDShape,iTrn] = min(d);
  
  iTrns4iTsts(iTest) = iTrn;
  dTrnMin(iTest) = dminDistToTrnDShape;
  
  p = pTrk(iTest,:);
  d = Shape.distP(pGTTrn,repmat(p,nTrn,1)); % [nTrnx7]
  d = mean(d(:,4:7),2);
  dminDistToTrnDShape = min(d);
  
  %iTrns4iTsts(iTest) = iTrn;
  dTrnMinTrk(iTest) = dminDistToTrnDShape;
  
  if mod(iTest,10)==0
    fprintf(1,'iTest=%d/%d\n',iTest,nTest);
  end
end
warning(warnst);

hFig = gobjects(0,1);
hFig(end+1,1) = figure('windowstyle','docked');
clear ax;
ax(1) = subplot(1,2,1);
scatter(dTrnMin,dTrk47Av);
[r,p] = corrcoef(dTrnMin,dTrk47Av);
tstr = sprintf('tracking err vs dist-GT-from-training: r = %.3g,p = %.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('dTrnMin','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');
grid on;
ax(2) = subplot(1,2,2);
scatter(dTrnMinTrk,dTrk47Av);
[r,p] = corrcoef(dTrnMinTrk,dTrk47Av);
tstr = sprintf('tracking err vs dist-Trk-from-training: r = %.3g,p = %.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('dTrnMinTrk','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');

resAssess.dTrnMin = dTrnMin;
resAssess.dTrnMinTrk = dTrnMinTrk;
resAssess.dTrk47Av = dTrk47Av;

%% replicate dispersion analysis
hFig(end+1,1) = figure('windowstyle','docked');
[~,~,~,~,~,~,~,xyRepMad47Av] = janResults(res);
assert(isequal(size(xyRepMad47Av),size(dTrk47Av)));
scatter(xyRepMad47Av,dTrk47Av);
[r,p] = corrcoef(xyRepMad47Av,dTrk47Av);
tstr = sprintf('tracking err vs repMad: r = %.3g,p = %.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('xyRepMad47Av','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');

resAssess.xyRepMad47Av = xyRepMad47Av;

%% dShape analysis
% For this analysis, we consider each dShape and consider how different it
% is from GT dShapes.
% To form the training set, we have selected a "good"/variable subset of
% all available labeled GT data. Thus the training set will not contain
% every consecutive labeled frame. 
% So in this analysis, we use the entire set of GT/labeled data available,
% *from all lbls/movs in the training set*. Thus we do not include dShapes
% in the test movie itself, but we do include all available dShapes from
% all training movies.

iTst = tdi.(tdiVarTst);
assert(issorted(iTst));
mdtrk = td.MD(iTst,:);
lblFileTrk = unique(mdtrk.lblFile);
iMovTrk = unique(mdtrk.iMov);
assert(isscalar(lblFileTrk) && isscalar(iMovTrk),...
  'Currently expect only one lblfile/movie in tracking results.');
assert(issorted(mdtrk.frm));

dfrms = diff(mdtrk.frm);
dfrmsUn = unique(dfrms);
dfrmsUnCnt = arrayfun(@(x)nnz(dfrms==x),dfrmsUn);
[dfrmCnt,idx] = max(dfrmsUnCnt);
dfrm = dfrmsUn(idx);
fprintf(1,'Tracking dshapes: use dfrm=%d, nTrkDShapes=%d\n',dfrm,dfrmCnt);
iTstiGoodDF = 1+find(dfrms==dfrm);
tmp0 = iTst(iTstiGoodDF-1);
tmp1 = iTst(iTstiGoodDF);
trkDShapesiTrl = [tmp0(:) tmp1(:)];
assert(size(trkDShapesiTrl,1)==dfrmCnt);

trnDShapesiTrl = zeros(0,2); % ith dShape is td.pGT(trnDShapes(i,2),:)-td.pGT(trnDShapes(i,1),:)
if ~isscalar(dfrm)
  fprintf(2,'nonscalar dfrm. skipping dShape analysis.\n');  
  
  % none; tfTrnDShapeSet is correct
else
  % figure out space of allowed delta-shapes (Allowed Motions)
  
  iTrn = tdi.(tdiVarTrn); % UNSORTED
  lblFileTrnUn = unique(td.MD.lblFile(iTrn));
  nLblFileTrnUn = numel(lblFileTrnUn); 
  for iLblFile = 1:nLblFileTrnUn
    
    lbl = lblFileTrnUn{iLblFile};
    tfLblFileAndLbled = strcmp(lbl,td.MD.lblFile) & td.isFullyLabeled;
    
    % current trials: from current lblFile, that are labeled
    iTrls = find(tfLblFileAndLbled);
    assert(all(td.MD.iMov(tfLblFileAndLbled)==1),...
      'Expected all iMov==1 for single lblfile.');
    assert(issorted(td.MD.frm(iTrls)));
    tfTrlDFMatch = [false; diff(td.MD.frm(iTrls))==dfrm];
    iGoodiTrls = find(tfTrlDFMatch); % indices into iTrls that are good
    
    fprintf(1,' Lblfile %s: %d labeled training frames with matching dfrm=%d.\n',...
      lbl,numel(iGoodiTrls),dfrm);
    
    trnDShapesiTrl = [trnDShapesiTrl; [iTrls(iGoodiTrls-1) iTrls(iGoodiTrls)]]; %#ok<AGROW>
  end
end

nTrnDShapeSet = size(trnDShapesiTrl,1);
if nTrnDShapeSet==0
  fprintf(2,'no training dShape set.\n');  
else
  fprintf(1,'Training dShape set size: %d\n',nTrnDShapeSet);
    
  lclCheckDShapeSet(trkDShapesiTrl,td.MD,dfrm);
  lclCheckDShapeSet(trnDShapesiTrl,td.MD,dfrm);
    
  dpTrk = td.pGT(trkDShapesiTrl(:,2),:)-td.pGT(trkDShapesiTrl(:,1),:);
  dpTrn = td.pGT(trnDShapesiTrl(:,2),:)-td.pGT(trnDShapesiTrl(:,1),:);
  ndpTrk = size(dpTrk,1);
  %assert(ndpTrk==nTest-1);
  
  dminDistToTrnDShape = nan(ndpTrk,1);
  warnst = warning('off','Shape:distP');
  fprintf(1,'Computing distances to training dShape set...\n');
  for i = 1:ndpTrk
    % for each dShapeTrk, find the nearest dpGTTst (most similar movement)

    dp = dpTrk(i,:);
    d = Shape.distP(dpTrn,repmat(dp,nTrnDShapeSet,1)); % [nTrnDShapeSetx7]
    d = mean(d(:,4:7),2);
    dminDistToTrnDShape(i) = min(d);
  end
  warning(warnst);

  hFig(end+1,1) = figure('windowstyle','docked');
  % - rows of dminDistToTrnDShape correspond to dpTrk.
  % - rows of dpTrk are iTrls labeled by trkDShapesiTrl(:,2).
  % - rows of dTrk47Av are iTrls labeled by tdi.(tdiVarTst)
  [tf,loc] = ismember(trkDShapesiTrl(:,2),tdi.(tdiVarTst));
  assert(all(tf));
  assert(issorted(loc));  
  dTrk47AvSubsetForDShape = dTrk47Av(loc,:);
  
  scatter(dminDistToTrnDShape,dTrk47AvSubsetForDShape);
  [r,p] = corrcoef(dminDistToTrnDShape,dTrk47AvSubsetForDShape);
  tstr = sprintf('tracking err vs dist-from-training dShape (nTrnDShape=%d): r = %.3g,p = %.3g',...
    nTrnDShapeSet,r(1,2),p(1,2));
  title(tstr,'fontweight','bold','interpreter','none');
  grid on;
  xlabel('dminDP','fontweight','bold');
  ylabel('dTrk47Av','fontweight','bold');
  
  resAssess.dminDistToTrnDShape = dminDistToTrnDShape;
  resAssess.dTrk47AvSubsetForDShape = dTrk47AvSubsetForDShape;
end

%%
fprintf(2,'TODO: Maximum correlation of test image to training image?\n');

%%
tfSaveAssess = ~isempty(saveAssess);
if tfSaveAssess
  assert(exist(saveAssess,'file')==0,'Assess file ''%s'' already exists.',...
    saveAssess);
  fprintf(1,'Saving assess results to:\n%s\n',saveAssess);
  save(saveAssess,'-mat','-struct','resAssess');
  resAssess; %#ok<VUNUS>
end

%%
tfSavePlots = ~isempty(savePlots);
if tfSavePlots
  fprintf(1,'Saving plots\n');
  SaveFigLotsOfWays(hFig,savePlots,{'fig'});
end

function lclCheckDShapeSet(dss,tMD,dfrm)

n = size(dss,1);
assert(size(dss,2)==2);
for i = 1:n
  md0 = tMD(dss(i,1),:);
  md1 = tMD(dss(i,2),:);
  assert(strcmp(md0.lblFile,md1.lblFile));
  assert(md0.iMov==md1.iMov);
  assert(md1.frm-md0.frm==dfrm);
end
