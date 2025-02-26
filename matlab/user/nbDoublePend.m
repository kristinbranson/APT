%% Notes

% frame-sets
% 0. The *labeled or GT set* is the set of labeled frames.
% 1. The *training set* is a subset of the labeled set used for training.
% 2. The *tracked set* is typically not a subset of either the labeled or
% training sets.
% 3. The tracked-not-train set is setdiff(2,1).
% 4. The *GT test set* is intersect(3,0) or some subset thereof.

%% Train
LBLS = { % lbl trk id ntrn tfSupervised
'dpend_trn30.lbl' 'dpend_trn30.trk' 'trn30' 30 'reg'
'dpend_trn30_sv11.lbl' 'dpend_trn30_sv11.trk' 'trn30_sv11' 41 'sv'
'dpend_trn30_sv11_sv20.lbl' 'dpend_trn30_sv11_sv20.trk' 'trn30_sv11_sv20' 61 'sv'
'dpend_trn30_sv11_sv20_sv15.lbl' 'dpend_trn30_sv11_sv20_sv15.trk' 'trn30_sv11_sv20_sv15' 76 'sv'
'dpend_trn30_sv11_sv20_sv15_sv16.lbl' 'dpend_trn30_sv11_sv20_sv15_sv16.trk' 'trn30_sv11_sv20_sv15_sv16' 92 'sv'
'dpend_trn60.lbl' 'dpend_trn60.trk' 'trn60' 60 'reg'
'dpend_trn90.lbl' 'dpend_trn90.trk' 'trn90' 90 'reg'
'dpend_trn120.lbl' 'dpend_trn120.trk' 'trn120' 120 'reg'
'dpend_trn150.lbl' 'dpend_trn150.trk' 'trn150' 150 'reg'
};
NLBL = size(LBLS,1);
tbl = cell2table(LBLS,'variablenames',{'lbl' 'trk' 'id' 'ntrn' 'grp'});
lbls = cellfun(@(x)load(x,'-mat'),tbl.lbl);
%%
for iLbl=1:NLBL
  lfile = LBLS{iLbl};
  lObj.projLoadGUI(lfile);
  fprintf('Loaded %s. Training\n',lfile);
  lObj.trainIncremental();
  fprintf('Saving %s\n',lObj.projectfile);
  lObj.projSaveSmart();
end
 
%% Track
tm = TrackMode.CurrMovCustomFrames;
tm.info = 1200:1800;
for iLbl=1:NLBL
  lfile = tbl.lbl{iLbl};
  tfile = tbl.trk{iLbl};
  lObj.projLoadGUI(lfile);
  fprintf('Loaded %s. Tracking to %s.\n',lfile,tfile);
  lObj.trackAndExportGUI(tm,'trkFilename',tfile,'trackArgs',{'stripTrkPFull' 0});
end  
%%
LBLGT = 'dpend_gt3.lbl';
frmTest = setdiff(1297:1659,[1303 1320 1341 1342 1343 1487]);
makeTrndataPlot(LBLGT,tbl.lbl,tbl.trk,'frmsTest',frmTest,'g',categorical(tbl.grp));
  
%% Transduction

% Starting from td30: Def had to skip many candidates. With such bad
% tracking not all candidates were good.

% Tried a few ideas for how to computationally find good tracking results.
% - velpred err, ie the tracking should in line with vel prediction on
% nearby training data. Velpred err didn't look very correlated with
% tracking err. One point here is that the dpend framerate is relatively
% coarse
% - replicate dispersion. Instead of previous "dmuRms" measure,
% "neighborscore" using a guassian kernel seems to work better. This counts
% how many replcates are nearby. The nborscore on pt4 is anticorrelated 
% with pt4 tracking error. (Most of the correlation may be driven by very
% bad tracking for low neighborscore replicate clouds). That said, a fixed
% threshold on neighborscore4 seemed to fail b/c the threshold depends on
% the amount of training data seen (more training data => better tracking
% generally => tighter replicate clouds).
% - nearness to training data. Points with tracking results (or gt labels)
% closer to the training set also tended to be better tracked (again noisy/
% very weak correlation).
%
% Rather than coming up with thresholds etc, we simply create a composite
% score by summing rescaled replicate dispersion and nearness-to-training-
% data. These candidates are given to the user in order of decreasing
% quality/confidence and the user can accept or reject.
 
NBORSIGMA = 40;
% %NBORSCORE_PT4_THRESH = 18.6; % AL20170216 ~90%tile of pt4 nbor scores; above this tracking pretty consistent
% NBORSCORE_PT4_THRESH = 16; 
% DIST_TRNSET = 14.2; % AL20170216 ~7.5%tile of eq distances

TRK = 'f:\romain\transduce20170206\dpend_trn30_sv11_sv20_sv15.trk';
trk = load(TRK,'-mat');
LBLTRN = 'dpend_trn30_sv11_sv20_sv15.lbl';
lbltrn = load(LBLTRN,'-mat');

% DAMPFAC = 0.85;
% VELPREDERR_PT4_THRESH = 26; % AL20170216 ~50%tile of pt4 velpred error with DAMPFAC=0.85

% frame lists
nptsTmp = Labeler.labelPosNPtsLbled(trk.pTrk);
frmsTrk = find(nptsTmp==4);
frmsTrn = sort(lbltrn.trackerData.trnDataTblP.frm);
frmsTrkNotTrn = setdiff(frmsTrk,frmsTrn);
nTrk = numel(frmsTrk);
nTrn = numel(frmsTrn);

lposTrn = lbltrn.labeledpos{1}(:,:,frmsTrn);
vTrn = reshape(lposTrn,[8 nTrn])'; % [nTrnx8]

frmsCand = frmsTrkNotTrn;
nCand = numel(frmsCand);
fprintf('\n');
fprintf('Start: %d nTrk, %d nTrn, %d candidates.\n',nTrk,nTrn,nCand);
fprintf('Cands: %s\n',mat2str(frmsCand));

% plot: two stats
nborscore = replicateTightness(trk.pTrk(:,:,frmsCand),...
  trk.pTrkFull(:,:,:,frmsCand),NBORSIGMA);
szassert(nborscore,[nCand 4]);
nbors4 = nborscore(:,4);
nbors4inv = 50-nbors4;
nbors4invSD = std(nbors4inv);
nbors4invzs = zscore(nbors4inv);

pCandTrk = trk.pTrk(:,:,frmsCand);
vCandTrk = reshape(pCandTrk,[8 nCand])'; % [nCandx8]
DTrn_CandTrk = pdist2(vTrn,vCandTrk,'euclidean');
minDtrn_CandTrk = min(DTrn_CandTrk,[],1);
minDtrn_CandTrk = minDtrn_CandTrk(:);
szassert(minDtrn_CandTrk,[nCand 1]);
minDtrn_CandTrkSD = std(minDtrn_CandTrk);
minDtrn_CandTrkzs = zscore(minDtrn_CandTrk);

figure;
hScatter = scatter(nbors4inv,minDtrn_CandTrk);
% hold on
% plot(0,minDtrn_CandTrkSD,'or');
% plot(nbors4invSD,0,'or');
grid on
xlabel('nbors4 inverted (smaller better)','fontweight','bold');
ylabel('minDtrn_CandTrk (smaller better)','fontweight','bold');

netscore = nbors4invzs+minDtrn_CandTrkzs;
%%

% tfOK = nborscore(:,4)>=NBORSCORE_PT4_THRESH;
% frmsFail = frmsCand(~tfOK);
% frmsCand = frmsCand(tfOK);
% nCand = numel(frmsCand);
% fprintf('Stage, repl: %d candidates remain.\n',nCand);
% fprintf('Failed: %s\n',mat2str(frmsFail));

% Stage: distance to training set must be small

if 0
  % minD_trk = minD_trk(:); % smallest distance to training set from tracked shapes on test set
  tfOK = minDtrn_CandTrk(:)<=DIST_TRNSET;
  frmsFail = frmsCand(~tfOK);
  frmsCand = frmsCand(tfOK);
  nCand = numel(frmsCand);
  fprintf('Stage, trnset dist: %d candidates remain.\n',nCand);
  fprintf('Failed: %s\n',mat2str(frmsFail));
else
  % sort by distance to training set
  [~,idx] = sort(netscore);
  %frmsCand = frmsCand(idx);
  fprintf('Sorted %d candidates by net score:\n',numel(frmsCand));
  %disp(frmsCand(:))
end

fprintf('Loading proj: %s\n',LBLTRN); 
lObj.projLoadGUI(LBLTRN);
fprintf('Importing trk results into labels2: %s\n',TRK); 
lObj.labels2ImportTrk(1,{TRK});

frmsCandAccept = [];
for iF=1:nCand
  idxFrmsCand = idx(iF);
  
  f = frmsCand(idxFrmsCand);
  lObj.setFrameGUI(f);
  hScatter.CData = zeros(nCand,3);
  hScatter.CData(idxFrmsCand,:) = [1 0 0];
  fprintf('frame %d (%d out of %d)\n',f,iF,nCand);
  ans = input('1 to accept, 2 to reject');
  switch ans
    case 1
      frmsCandAccept(end+1) = f;
      fprintf('Accepted %d',f);
  end
end

%%
Save to new project
%%
nCandAcc = numel(frmsCandAccept);
for iF=1:nCandAcc
  f = frmsCandAccept(iF);
  lObj.setFrameGUI(f);
  lObj.labelPosSetFromLabeledPos2();
  fprintf('Set labels for frame %d (%d out of %d)\n',f,iF,nCandAcc);
end

%%
TRKS = { % lbl trkfile id nTrnFrm tfSupervis
  'dp_1205_more_trned_40td_rot.lbl' 'doubpend_td40_rot.trk' 'td40_r' 40 0
  'dp_1205_trned_60td_rot.lbl' 'dp_1205_trned_60td_rot.trk' 'td60_r' 60 0
  'dp_1205_trned_60td_rot_15sv.lbl' 'dp_1205_trned_60td_rot_15sv.trk' 'td60_r_15sv' 75 1
  'dp_1205_trned_80td_rot.lbl' 'doubpend_td80_rot.trk' 'td80_r' 80 0
  'dp_1205_trned_120td_rot.lbl' 'doubpend_td120_rot.trk' 'td120_r' 120 0
  'dp_1205_trned_80td_rot_plus15supervis.lbl' 'dp_1205_trned_80td_rot_plus15supervis.trk' 'td80_r_15spv' 95 1
  'dp_1205_trned_80td_rot_plus15supervis.lbl' 'dp_1205_trned_80td_rot_plus15supervis.trk' 'td80_r_15spv' 95 1
  'dp_1205_trned_80td_rot_plus15supervis_plus25sv.lbl' 'dp_1205_trned_80td_rot_plus15supervis_plus25sv.trk' 'td80_r_15_25spv' 120 1
  };
TRKS = cell2table(TRKS,'variablenames',{'lbl' 'trkfile' 'id' 'nTrnFrm' 'tfSupervis'});
NTRKS = size(TRKS,1);
LBLGT = 'dp_1205_more2.lbl';


  













%% Rotation Correction Helps
% (Re)generate trkfiles

tm = TrackMode.CurrMovCustomFrames;
tm.info = 1200:1700;
for iLbl=1:NTRKS
  lblfile = TRKS.lbl{iLbl};
  fprintf('Load %s\n',lblfile);
  lObj.projLoadGUI(lblfile);
  trkfile = TRKS.trkfile{iLbl};
  [p,f,e] = fileparts(trkfile);
  fprintf('Tracking to %s\n',f);
  lObj.trackAndExportGUI(tm,'trkFilename',f);
end

%%

TRKS = { % lbl trkfile id nTrnFrm tfRotCorrect
  'dp_1205_more_trned_40td_norot.lbl' 'doubpend_td40_norot.trk' 'td40_nr' 40 0
  'dp_1205_more_trned_40td_rot.lbl' 'doubpend_td40_rot.trk' 'td40_r' 40 1
  'dp_1205_trned_80td_norot.lbl' 'doubpend_td80_norot.trk' 'td80_nr' 80 0 
  'dp_1205_trned_80td_rot.lbl' 'doubpend_td80_rot.trk' 'td80_r' 80 1
  'dp_1205_trned_120td_norot.lbl' 'doubpend_td120_norot.trk' 'td120_nr' 120 0 
  'dp_1205_trned_120td_rot.lbl' 'doubpend_td120_rot.trk' 'td120_r' 120 1
  'dp_1205_trned_160td_norot.lbl' 'doubpend_td160_norot.trk' 'td160_nr' 160 0 
  'dp_1205_trned_160td_rot.lbl' 'doubpend_td160_rot.trk' 'td160_r' 160 1
  };
TRKS = cell2table(TRKS,'variablenames',{'lbl' 'trkfile' 'id' 'nTrnFrm' 'tfRotCorrect'});
NTRKS = size(TRKS,1);
LBLGT = 'dp_1205_more2.lbl';

lbls = cellfun(@(x)load(x,'-mat'),TRKS.lbl);
trks = cellfun(@(x)load(x,'-mat'),TRKS.trkfile);
lblgt = load(LBLGT,'-mat');

nptsGT = Labeler.labelPosNPtsLbled(lblgt.labeledpos{1});
frmsGT = find(nptsGT==4);
frmsGTTest = frmsGT;
fprintf('Starting with %d GT frames.\n',numel(frmsGTTest));
for iLbl=1:NTRKS
  frmsTrn = lbls(iLbl).trackerData.trnDataTblP.frm;
  fprintf('%s: %d training frames.\n',TRKS.lbl{iLbl},numel(frmsTrn));
  frmsGTTest = setdiff(frmsGTTest,frmsTrn);
end
nfrmsGTTest = numel(frmsGTTest);
fprintf('Using %d GT frames for testing.\n',nfrmsGTTest);

lposGTTest = lblgt.labeledpos{1}(:,:,frmsGTTest);
dmat = nan(nfrmsGTTest,4,NTRKS);
for iTrk=1:NTRKS
  ptrk = trks(iTrk).pTrk(:,:,frmsGTTest);
  d = squeeze(sqrt(sum((ptrk-lposGTTest).^2,2)))';
  dmat(:,:,iTrk) = d;
end

figure;
dmatPt4 = squeeze(dmat(:,4,:));
boxplot(dmatPt4,TRKS.id)
grid on
title('pt 4 eqdist trking err','fontweight','bold');

dmatmu = squeeze(mean(dmat,1));
dmatmu = dmatmu'; % rows: iTrk. cols: iPt
format short
dmatmu

figure;
nTrnFrms = TRKS.nTrnFrm;
tfRC = logical(TRKS.tfRotCorrect);
dmatmuPt4Rot = dmatmu(tfRC,4);
dmatmuPt4NR = dmatmu(~tfRC,4);
frmsRot = nTrnFrms(tfRC);
frmsNR = nTrnFrms(~tfRC);
plot(frmsRot,dmatmuPt4Rot,'.-',frmsNR,dmatmuPt4NR,'.-');
grid on;
xlabel('num training frames','fontweight','bold');
ylabel('mean tracking err, pt 4 (px)','fontweight','bold');
legend('rot correct','no rot correct');
axis([0 200 0 200]);
tstr = sprintf('Rot correction helps: %d test frames\n',nfrmsGTTest);
title(tstr,'fontweight','bold');


TRKS = { % lbl trkfile id nTrnFrm tfIncremental
    'dp_1205_first40_inctrain10.lbl' 'doubpend_tdfirst40_rot_inctrain10.trk' 'td40_10' 50 1
    'dp_1205_first40_inctrain20.lbl' 'doubpend_tdfirst40_rot_inctrain20.trk' 'td40_20' 60 1
    'dp_1205_first40_inctrain30.lbl' 'doubpend_tdfirst40_rot_inctrain29.trk' 'td40_29' 69 1
    'dp_1205_first40_inctrain40.lbl' 'doubpend_tdfirst40_rot_inctrain39.trk' 'td40_39' 79 1
    'dp_1205_first40.lbl' 'doubpend_tdfirst40_rot_inctrain0.trk' 'td40' 40 0
    'dp_1205_first70.lbl' 'doubpend_tdfirst70_rot.trk' 'td69' 69 0
    'dp_1205_first80.lbl' 'doubpend_tdfirst80_rot.trk' 'td79' 79 0
    };
TRKS = cell2table(TRKS,'variablenames',{'lbl' 'trkfile' 'id' 'nTrnFrm' 'tfIncremental'});
NTRKS = size(TRKS,1);
LBLGT = 'dp_1205_more2.lbl';

lbls = cellfun(@(x)load(x,'-mat'),TRKS.lbl);
trks = cellfun(@(x)load(x,'-mat'),TRKS.trkfile);
lblgt = load(LBLGT,'-mat');

nptsGT = Labeler.labelPosNPtsLbled(lblgt.labeledpos{1});
frmsGT = find(nptsGT==4);
frmsGTTest = frmsGT;
fprintf('Starting with %d GT frames.\n',numel(frmsGTTest));
for iLbl=1:NTRKS
  frmsTrn = lbls(iLbl).trackerData.trnDataTblP.frm;
  fprintf('%s: %d training frames.\n',TRKS.lbl{iLbl},numel(frmsTrn));
  frmsGTTest = setdiff(frmsGTTest,frmsTrn);
end
nfrmsGTTest = numel(frmsGTTest);
fprintf('Using %d GT frames for testing.\n',nfrmsGTTest);

lposGTTest = lblgt.labeledpos{1}(:,:,frmsGTTest);
dmat = nan(nfrmsGTTest,4,NTRKS);
for iTrk=1:NTRKS
  ptrk = trks(iTrk).pTrk(:,:,frmsGTTest);
  d = squeeze(sqrt(sum((ptrk-lposGTTest).^2,2)))';
  dmat(:,:,iTrk) = d;
end

figure;
dmatPt4 = squeeze(dmat(:,4,:));
boxplot(dmatPt4,TRKS.id)
grid on
tstr = sprintf('pt4 eqdist trking err: %d test frames\n',nfrmsGTTest);
title(tstr,'fontweight','bold');

dmatmu = squeeze(mean(dmat,1));
dmatmu = dmatmu'; % rows: iTrk. cols: iPt
format short
dmatmu

figure;
nTrnFrms = TRKS.nTrnFrm;
tfInc = logical(TRKS.tfIncremental);
dmatmuPt4inc = dmatmu(tfInc,4);
dmatmuPt4Ninc = dmatmu(~tfInc,4);
frmsInc = nTrnFrms(tfInc);
frmsNinc = nTrnFrms(~tfInc);
plot(frmsInc,dmatmuPt4inc,'.-',frmsNinc,dmatmuPt4Ninc,'.-');
grid on;
xlabel('num training frames','fontweight','bold');
ylabel('mean tracking err, pt 4 (px)','fontweight','bold');
legend('incremental','not');
axis([0 200 0 200]);
tstr = sprintf('pt4 eqdist trking err: %d test frames\n',nfrmsGTTest);
title(tstr,'fontweight','bold');




%% Explore tracking DoubPend via transduction (expanding training set)
%% Load stuff, preprocess

TRK = 'f:\romain\transduce20170206\doubpend_td80_rot_full.trk';
trk = load(TRK,'-mat');
LBLTRN = 'dp_1205_trned_80td_rot.lbl';
lbltrn = load(LBLTRN,'-mat');
LBLGT = 'f:\romain\transduce20170206\dp_1205_more2.lbl';
lblgt = load(LBLGT,'-mat');
nptsTmp = Labeler.labelPosNPtsLbled(lblgt.labeledpos{1});
frmGT = find(nptsTmp==4);
nptsTmp = Labeler.labelPosNPtsLbled(trk.pTrk);
frmTrk = find(nptsTmp==4);
frmTrn = sort(lbltrn.trackerData.trnDataTblP.frm);

frmTrkNotTrn = setdiff(frmTrk,frmTrn);
frmTrkGTNotTrn = setdiff(intersect(frmTrk,frmGT),frmTrn);

trkposTrkNotTrn = trk.pTrk(:,:,frmTrkNotTrn);
trkposFTrkNotTrn = trk.pTrkFull(:,:,:,frmTrkNotTrn);

% trk, gt, not trn
trkposTrkGTNotTrn = trk.pTrk(:,:,frmTrkGTNotTrn);
trkposFTrkGTNotTrn = trk.pTrkFull(:,:,:,frmTrkGTNotTrn);
nTrkGTNotTrn = numel(frmTrkGTNotTrn);
lposTrkGTNotTrn = lblgt.labeledpos{1}(:,:,frmTrkGTNotTrn);
dposTrkGTNotTrn = sqrt(sum((trkposTrkGTNotTrn-lposTrkGTNotTrn).^2,2));
dposTrkGTNotTrn = squeeze(dposTrkGTNotTrn)';
szassert(dposTrkGTNotTrn,[nTrkGTNotTrn 4]);

%% Replicate Spreads

%[trkposFVdmuRms,trkposFmad] = lclReplicateDmuRms(trkposFTrkNotTrn);
% medoid-mad not doing a good job b/c the replicates are quite spread out even when 
% tracking results are good/decent.

% figure out sigma
NBORSIGMA = 40;
[nborscore,d2all] = replicateTightness(trkposTrkGTNotTrn,trkposFTrkGTNotTrn,NBORSIGMA);
dall = sqrt(d2all);
dall = permute(dall,[1 3 2]);
nrep = 50;
szassert(dall,[nTrkGTNotTrn 50 4]);
dall = reshape(dall,[nTrkGTNotTrn*50 4]);

hFig = figure;
axs = createsubplots(1,4,.1);
for ipt=1:4
  ax = axs(ipt);
  axes(ax);
  h(1) = histogram(dall(:,ipt));    
  grid(ax,'on');
  if ipt==2
    title(ax,'dist to pruned trk, all frms/reps (px)','fontweight','bold');
  else
    ax.XTickLabel = [];
    ax.YTickLabel = [];
  end
  ylabel(sprintf('pt %d',ipt),'fontweight','bold');
end
linkaxes(axs);
xlim(axs(1),[0 500]);
ptileDall = prctile(dall,[10 25 50 75 90])

% nbor score
hFig = figure;
axs = createsubplots(4,1,.1);
for ipt=1:4
  ax = axs(ipt);
  axes(ax);
  h(1) = histogram(nborscore(:,ipt));    
  grid(ax,'on');
  if ipt==1
    tstr = sprintf('nborscore (sigma=%.2f)',NBORSIGMA);
    title(ax,tstr,'fontweight','bold');
  else
    ax.XTickLabel = [];
    ax.YTickLabel = [];
  end
  ylabel(sprintf('pt %d',ipt),'fontweight','bold');
end
linkaxes(axs);
%xlim(axs(1),[0 500]);
ptileDall = prctile(nborscore,[10 25 50 75 90])

%% correlation with gterr

%[trkposFVdmuRmsTrkGTNotTrn,trkposFmadTrkGTNotTrn] = lclReplicateDmuRms(trkposFTrkGTNotTrn);
npts = 4;  
dposComb = sqrt(sum(dposTrkGTNotTrn.^2,2));
figure;
axs = createsubplots(2,2,.1);
for ipt=1:npts
  axes(axs(ipt));
  x = nborscore(:,ipt);
  y = dposTrkGTNotTrn(:,ipt);
  scatter(x,y);
  grid on;
  
  xlabel('nborscore','fontweight','bold');
  ylabel('tracking err (px)','fontweight','bold');
  tfOut = false(size(y));
  xNoOut = x(~tfOut);
  yNoOut = y(~tfOut);
  [r,p] = corrcoef(xNoOut,yNoOut);
  title(sprintf('trking vs replSpread, pt %d. n=%d, r=%.2f,p=%.2g',ipt,nTrkGTNotTrn,r(1,2),p(1,2)),...
    'fontweight','bold');
  ax = gca;
end
%ax.XLim = [0 380];
%ax.YLim = [0 50];

%% velpred

DAMPS = 0:.05:1;
nDamps = numel(DAMPS);

frm3 = nan(0,1); % third frame in consecutive seq
frm3Off = nan(0,1); % frm3-11e3+1
ipts = nan(0,1); % point index
pAct = nan(0,2); % labeled [x y] of pt
pPrv = nan(0,2); % previous labeled [x y]
pPrdD = cell(0,1); % predicted based on pAct, pPrv, dampvel. each el is [nDampsx2]
ePrv = nan(0,1);
ePrdD = cell(0,1);
pTrk = nan(0,2); % tracked [x y] of pt
eTrk = nan(0,1); 
%nfrmGT = numel(frmGT);
lposGT = lblgt.labeledpos{1};
for i=3:nTrkGTNotTrn
  f3 = frmTrkGTNotTrn(i);  
  tfConsec = isequal(frmTrkGTNotTrn(i-2:i),(f3-2:f3)');
  if tfConsec
    p123 = lposGT(:,:,f3-2:f3);
    for iPt = 1:npts
      p = p123(iPt,:,:);
      p = squeeze(p)';
      szassert(p,[3 2]); % rows: [f1 f2 f3]. cols: [x y]
      if nnz(isnan(p)|isinf(p))==0 %&& size(unique(p,'rows'),1)==3
        frm3(end+1,1) = f3;
        ipts(end+1,1) = iPt;
        pAct(end+1,:) = p(3,:);
        pPrv(end+1,:) = p(2,:);
        ePrv(end+1,1) = sqrt(sum((p(3,:)-p(2,:)).^2));
        tmp = arrayfun(@(x)p(2,:)+x*(p(2,:)-p(1,:)),DAMPS,'uni',0);
        pPrdDthis = cat(1,tmp{:});
        pPrdD{end+1,1} = pPrdDthis;
        szassert(pPrdDthis,[nDamps 2]);
        ePrdD{end+1,1} = sqrt(sum((p(3,:)-pPrdDthis).^2,2));
        
        pTrkf3 = trk.pTrk(iPt,:,f3);
        pTrk(end+1,:) = pTrkf3;
        eTrk(end+1,1) = sqrt(sum((p(3,:)-pTrkf3).^2));
      end
    end
  end
end
tblConsec3 = table(frm3,ipts,pAct,pPrv,pPrdD,ePrv,ePrdD,pTrk,eTrk);

%
npts = 4;
errPrvMu = nan(nDamps,npts);
errPrvMdn = nan(nDamps,npts);
errPrdDMu = nan(nDamps,npts);
errPrdDMdn = nan(nDamps,npts);
for iDamp=1:nDamps
  for iPt=1:npts
    tf = tblConsec3.ipts==iPt;
    
    pActThis = tblConsec3.pAct(tf,:);
    pPrvThis = tblConsec3.pPrv(tf,:);
    pPrdDThis = tblConsec3.pPrdD(tf,:);
    pPrdDThis = cellfun(@(x)x(iDamp,:),pPrdDThis,'uni',0);
    pPrdDThis = cat(1,pPrdDThis{:});
    
    ePrvThis = sqrt(sum((pActThis-pPrvThis).^2,2));
    ePrdDThis = sqrt(sum((pActThis-pPrdDThis).^2,2));        
    errPrvMu(iDamp,iPt) = mean(ePrvThis);
    errPrvMdn(iDamp,iPt) = median(ePrvThis);
    errPrdDMu(iDamp,iPt) = mean(ePrdDThis);
    errPrdDMdn(iDamp,iPt) = median(ePrdDThis);
  end
  fprintf(1,'Damp %d\n',iDamp);
end
errPrdDMuTot = sum(errPrdDMu,2);
errPrdDMdnTot = sum(errPrdDMdn,2);
errPrvMuTot = sum(errPrvMu,2);
errPrvMdnTot = sum(errPrvMdn,2);

figure;
ax = createsubplots(1,4,.08); % rows: view. cols: [errPrvMu errPrdDMu errPrvMdn errPrdDMdn]
ax = reshape(ax,[1 4]);
arrayfun(@(x)hold(x,'on'),ax);
arrayfun(@(x)grid(x,'on'),ax);

for iPt=1:npts
  x = errPrvMu(:,iPt);
  szassert(x,[nDamps 1]);
  hVw1(iPt) = plot(ax(1),DAMPS(:),x);
  
  x = errPrdDMu(:,iPt);
  szassert(x,[nDamps 1]);
  plot(ax(2),DAMPS(:),x);
  
  x = errPrvMdn(:,iPt);
  szassert(x,[nDamps 1]);
  plot(ax(3),DAMPS(:),x);
  
  x = errPrdDMdn(:,iPt);
  szassert(x,[nDamps 1]);
  plot(ax(4),DAMPS(:),x);
end

legend(ax(1),{'pt1' 'pt2' 'pt3' 'pt4'});

title(ax(1),'errPrvMu','fontweight','bold');
title(ax(2),'errPrdDMu','fontweight','bold');
title(ax(3),'errPrvMdn','fontweight','bold');
title(ax(4),'errPrdDMdn','fontweight','bold');

xlabel(ax(1),'dampfac','fontweight','bold');
ylabel(ax(1),'error(px)','fontweight','bold');

ax(2).XTickLabel = [];
ax(2).YTickLabel = [];
ax(4).XTickLabel = [];
ax(4).YTickLabel = [];

linkaxes(ax(:,1:2));
linkaxes(ax(:,3:4));
arrayfun(@(x)grid(x,'on'),ax);

figure;
ax = createsubplots(1,4,.08); % rows: view. cols: [errPrvMu errPrdDMu errPrvMdn errPrdDMdn]
ax = reshape(ax,[1 4]);
arrayfun(@(x)hold(x,'on'),ax);
arrayfun(@(x)grid(x,'on'),ax);

plot(ax(1),DAMPS(:),errPrvMuTot);
plot(ax(2),DAMPS(:),errPrdDMuTot);
plot(ax(3),DAMPS(:),errPrvMdnTot);
plot(ax(4),DAMPS(:),errPrdDMdnTot);

title(ax(1),'errPrvMuTot','fontweight','bold');
title(ax(2),'errPrdDMuTot','fontweight','bold');
title(ax(3),'errPrvMdnTot','fontweight','bold');
title(ax(4),'errPrdDMdnTot','fontweight','bold');
    
xlabel(ax(1),'dampfac','fontweight','bold');
ylabel(ax(1),'error(px)','fontweight','bold');
linkaxes(ax);

% CONCLUSION AL 20170215: dampfac=0.85
%% Pick a dampfac, focus on it

DAMPFAC = 0.85;
iDampUse = find(DAMPS==DAMPFAC)

errPrdDPts = cell(npts,1); % el i contains vector of observed errPrdD on gt data for pt i
errPrdDPtsImprov = cell(npts,1); % el i contains vector of observed errPrdD-errPrv on gt data for pt i
for ipt=1:npts
  tf = ipt==tblConsec3.ipts;
  errPrdDPts{ipt} = cellfun(@(x)x(iDampUse,:),tblConsec3.ePrdD(tf));
  errPrdDPtsImprov{ipt} = tblConsec3.ePrv(tf)-errPrdDPts{ipt};
end

% plot hists of errPrdDPts/improvement
figure;
axs = createsubplots(1,4,.1);
clear h;
for ipt=1:npts
  axes(axs(ipt));
  h(1) = histogram(errPrdDPts{ipt});
  grid on;
  if ipt==1
    xlabel('errPrd (px)','fontweight','bold');
  end
end
linkaxes(axs);
axs(1).XLim = [0 120];

errPrdDPtsMat = cat(2,errPrdDPts{:});
ptile50ErrPrdPts = prctile(errPrdDPtsMat,[10 25 50 75 90])

figure;
histogram(errPrdDPts{4});
grid on;
xlabel('errPrd (px)','fontweight','bold');
title('errPrd for pt4','fontweight','bold');


% correlation between ePrd and ePrv for pt 4
tfpt4 = tblConsec3.ipts==4;
ePrd = cellfun(@(x)x(iDampUse,:),tblConsec3.ePrdD);
figure;
scatter(tblConsec3.ePrv(tfpt4),ePrd(tfpt4));
grid on;
xlabel('err prev, px','fontweight','bold');
ystr = sprintf('err pred (px, damp=%.2f)',DAMPFAC);
ylabel(ystr,'fontweight','bold');
[r,p] = corrcoef(tblConsec3.ePrv(tfpt4),ePrd(tfpt4));
tstr = sprintf('velpred err tracks eprv (pt4), r=%.3f,p=%.2g',r(1,2),p(1,2));
title(tstr,'fontweight','bold');

% correlation between errPrdD and eTrk for pt4
x = ePrd(tfpt4);
y = tblConsec3.eTrk(tfpt4);
[r,p] = corrcoef(x,y);

figure;
axs = createsubplots(1,2,.1);
axes(axs(1));
scatter(x,y);
grid on;
xstr = sprintf('err pred (px, damp=%.2f)',DAMPFAC);
xlabel(xstr,'fontweight','bold');
ylabel('err trk, px','fontweight','bold');
tstr = sprintf('errtrk vs errvelpred, pt4 (n=%d), r=%.3f,p=%.2g',nnz(tfpt4),r(1,2),p(1,2));
title(tstr,'fontweight','bold');
set(gca,'XLim',[0 150],'YLim',[0 50]);
axes(axs(2));
scatter(x,y);
grid on;
set(gca,'XScale','log','YScale','log');

%% find tracked shapes close to training shape
frmTrn = sort(lbltrn.trackerData.trnDataTblP.frm);
nTrn = numel(frmTrn);
lposTrn = lbltrn.labeledpos{1}(:,:,frmTrn);
vTrn = reshape(lposTrn,[8 nTrn])'; % [nTrnx8]

frmTest = frmTrkGTNotTrn;
nTest = numel(frmTest);
pTestTrk = trk.pTrk(:,:,frmTest);
vTestTrk = reshape(pTestTrk,[8 nTest])'; % [nTrkNotTrnx8]

lposTest = lblgt.labeledpos{1}(:,:,frmTest);
vTestGT = reshape(lposTest,[8 nTest])'; % [nTest x 8]
dTest = sqrt(sum((vTestTrk-vTestGT).^2,2));

DTrnTest_Trk = pdist2(vTrn,vTestTrk,'euclidean');
minD_trk = min(DTrnTest_Trk,[],1);
minD_trk = minD_trk(:); % smallest distance to training set from tracked shapes on test set

DTrnTest_GT = pdist2(vTrn,vTestGT,'euclidean');
minD_gt = min(DTrnTest_GT,[],1);
minD_gt = minD_gt(:); % smallest distance to training set from tracked shapes on test set

figure;
scatter(minD_gt,minD_trk);
grid on;
xlabel('mindist from gt shape to training set','fontweight','bold');
ylabel('mindist from trked shape to training set','fontweight','bold');
[r,p] = corrcoef(minD_gt,minD_trk);
tstr = sprintf('r=%.3f,p=%.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold');

figure;
scatter(minD_gt,dTest);
grid on;
xlabel('mindist from gt shape to training set','fontweight','bold');
ylabel('tracking err','fontweight','bold');
[r,p] = corrcoef(minD_gt,dTest);
tstr = sprintf('r=%.3f,p=%.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold');

figure;
scatter(minD_trk,dTest);
grid on;
xlabel('mindist from tracked shape to training set','fontweight','bold');
ylabel('tracking err','fontweight','bold');
[r,p] = corrcoef(minD_trk,dTest);
tstr = sprintf('r=%.3f,p=%.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold');

figure
histogram(minD_trk)
grid on;
xlabel('mindist from tracked shape to trn set','fontweight','bold');
ptileminD_trk = prctile(minD_trk,[5 7.5 10 25 50])


%% browse reps

hLine = RF.addLinesToLabelerAxis(lObj,'npts',4);

