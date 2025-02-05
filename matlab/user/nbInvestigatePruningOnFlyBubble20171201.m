% CONCLUSIONS 20171206
%
% Objective was to investigate "static" cpr pruning methods (median vs kde 
% vs "global kde"), and ChooseBest... traj smoothing.
%
% Data used was i) FlyBubble (multitarget_bubble_n2195.lbl), this is prior 
% to expanded grooming/behaviors, so mainly just isolated flies walking etc.
% Currently tracking is quite strong for this (<1px typical XV err). ii)
% mousereach data, N~8000
%
% For bub, pull out test set of size Ntst~300, train on remainder. For
% mouse, existing tracker trained on all data, Use test set of ~200 so this
% overlaps with training set. Test pruning methods on test set vs GT lbls.
%
% For kde/glbl pruning, did titration of sigma parameter. For
% trajsmoothing, did titration of sigma parameter along with some others.
% 
% FlyBub: Bubble with Traj: Median won, not sure it reaches significance. 
%
% Britton: kde/glbl/traj tie, these three beat median but seem same-ish


%%%% BUB DATA PREP %%%%%
%% 
% lObj = ... flybubble proj ...

t = lObj.labelGetMFTableLabeled;
%%
g = categorical(t.mov).*categorical(t.iTgt);
summary(g)
iMov2 = find(t.mov==2);
NTEST = 300;
iTst = randsample(iMov2,NTEST);
iTrn = setdiff(1:height(t),iTst);
%% Train on a subset
tObj = lObj.tracker;
sPrm = tObj.sPrm;
roiRadius = sPrm.PreProc.TargetCrop.Radius;
tTrn = t(iTrn,:);
tTrn = lObj.labelMFTableAddROITrx(tTrn,roiRadius);
tTrn.pAbs = tTrn.p;
tTrn.p = tTrn.pRoi;
%%
tObj.retrain('tblPTrn',tTrn);

%% Traj smoothing: need consecutive frames tracked. For each row of tTst, 
% track from -75 frames previous to +50 frames after
tTst = t(iTst,:);

PREFRMS = 75;
POSTFRMS = 50;

tblTrkTraj = [];
for i=1:height(tTst)
  row = tTst(i,:);
  frmschunk = row.frm-PREFRMS:row.frm+POSTFRMS;
  nchunk = numel(frmschunk);
  tblchunk = table(repmat(row.mov,nchunk,1),frmschunk(:),...
    repmat(row.iTgt,nchunk,1),'VariableNames',{'mov' 'frm' 'iTgt'});
  tblTrkTraj = [tblTrkTraj;tblchunk];
end

tblTrkTraj = unique(tblTrkTraj);
fprintf('%d rows to be tracked.\n',height(tblTrkTraj));
%%
lObj.trackTbl(tblTrkTraj);

% exported to movie_trk9272fullForTrajSmoothing.trk

%%% END DATA PREP %%%

%%
%%%% BRITTON DATA PREP %%%%%

% lObj = <load reach_all_mice_including_BPN_perturb_TRAINING_AL_trained.lbl>
% This is a project trained on ~8K rows
%
% full track and export movie 7

t = lObj.labelGetMFTableLabeled;
tfMov7 = t.mov==7;
tTst = t(tfMov7,:);
nTst = height(tTst);

%% Load tracked data: Bub
trk = load('f:\pathMacros20170731\localdata\cx_GMR_SS00030_CsChr_RigC_20150826T144616\movie_trk9272fullForTrajSmoothing.trk','-mat');
%% Load tracked data: Britton
trk = load('F:\aptSmoothingAndSereInitOnTopOfTrx20171128\M195_20160504_v202\movie_comb_reach_all_mice_TRAINING.trk','-mat');
%%
[npts,d,nRep,nTrkFull]  = size(trk.pTrkFull);
D = d*npts;
pTrkFull = reshape(trk.pTrkFull,[D,nRep,nTrkFull]);

pTrkFull = permute(pTrkFull,[3 2 1]); % [nTrkFull x nRep x D]
pTrkMD = trk.pTrkFullFT;
pTrkMD = [table(mov) pTrkMD];

%% 
[tf,loc] = tblismember(tTst,pTrkMD,MFTable.FLDSID);
assert(all(tf));
pTrkFullStatic = pTrkFull(loc,:,:);
pGT = tTst.p;
errGT = @(pTrk,pGT) sqrt(sum(reshape(pTrk-pGT,[nTst npts d]).^2,3));

%% Static prune: median
[pTrk_med,score_med,info_med] = Prune.median(pTrkFullStatic);
eTrk_med = errGT(pTrk_med,pGT);
mednerrpt = median(eTrk_med)
mean(mednerrpt)

figure;
scatter(score_med,mean(eTrk_med,2));
grid on;
title('prune.median, mean trkerr vs -(repl mad)','fontweight','bold','interpreter','none');

% Bub: trkerr dereases with increasing score as expected

% Britton: trkerr dereases with increasing score as expected

%% Static prune: kde calibration
[~,~,info] = Prune.maxdensity(pTrkFullStatic,'sigma',0);

kde_d2 = cat(1,info{:}); % [nrep*(nrep-1)/2*nTst x npt] pairwise dist^2 for each replicate/pt
h = figure;
axs = createsubplots(1,2);
%axs = createsubplots(3,6);
%bctrs = 0:1:25;
bctrs = 0:4:400;
for i=1:npts
  ax = axs(i);
  axes(ax);
  hist(kde_d2(:,i),bctrs);
  title(num2str(i));
  grid on;
  if i>1
    ax.XTickLabel = [];
    ax.YTickLabel = [];
  end
end
linkaxes(axs,'x');
%xlim(axs(1),[0 30]);
xlim(axs(1),[0 450]);

% For FlyBub leg (tips), looks like scale of d2 is ~2

% Britton, scale of d2 is ~50-100

%% Static prune: kde
%SIGMAS = [sqrt(0.1) sqrt(0.33) 1 sqrt(2) sqrt(5)];
SIGMAS = [sqrt(5) sqrt(20) sqrt(50) sqrt(100) sqrt(250)];
nSig = numel(SIGMAS);
[pTrk_kde_sig,score_kde_sig] = ...
  arrayfun(@(sig)Prune.maxdensity(pTrkFullStatic,'sigma',sig),SIGMAS,'uni',0);

eTrk_kde_sig = cell(nSig,1);
for iSig=1:nSig
  eTrk_kde_sig{iSig} = errGT(pTrk_kde_sig{iSig},pGT);
  fprintf('iSig %d, sig=%.3f. MeanMdn err: %.4f\n',iSig,SIGMAS(iSig),...
    mean(median(eTrk_kde_sig{iSig})));
end

% sqrt(0.33) is the best, use 1 though about the same and looks more
% reasonable

% Britton mouse: best is sqrt(20), sqrt(50)
%%
iSigBest = 3;
pTrk_kde = pTrk_kde_sig{iSigBest};
score_kde = score_kde_sig{iSigBest};
eTrk_kde = eTrk_kde_sig{iSigBest};

figure;
scatter(score_kde,mean(eTrk_kde,2));
grid on;
title('prune.kde, trkerr vs score','fontweight','bold','interpreter','none');

% Bub: score/trkerr POSITIVELY correlated
%  - Notes on kde appearancecost. It needs to be within-iter comparable but i 
%    think it looks like it doesn't have to be between-iter comparable. Which
%    is good b/c as we saw, with KDE, the maxPr score is not anticorrelated 
%    with error. One theory, when tracking is good, many reps are 
%    similar/close together, with the result that probability gets widely 
%    shared, leading to smaller maxPr scores even for the best one. With bad 
%    tracking, the reps are spread out, and it is more likely that one rep 
%    stands out and wins a greater share of pr.

% Britton: score/trkerr +vely correlated again

%% Static prune: global min calib
[~,~,info] = Prune.globalmin(pTrkFullStatic,'sigma',inf); % sigma should not matter

glbl_d2 = cat(1,info{:}); % [nrep*(nrep-1)/2*nTst x 1] pairwise dist^2 for each replicate
figure;
hist(glbl_d2,0:4:400);
grid on;
xlim([0 450]);

% Bub: looks like scale of d2 is ~5-8

% Britton: scale is ~100

%% Static prune: global min
%SIGMAS = [sqrt(1) sqrt(2) sqrt(5) sqrt(20)];
SIGMAS = [sqrt(5) sqrt(20) sqrt(50) sqrt(100) sqrt(250)];
nSig = numel(SIGMAS);
[pTrk_glbl_sig,score_glbl_sig] = ...
  arrayfun(@(sig)Prune.globalmin(pTrkFullStatic,'sigma',sig),SIGMAS,'uni',0);

eTrk_glbl_sig = cell(nSig,1);
for iSig=1:nSig
  eTrk_glbl_sig{iSig} = errGT(pTrk_glbl_sig{iSig},pGT);
  fprintf('iSig %d, sig=%.3f. MeanMdn err: %.3f\n',iSig,SIGMAS(iSig),...
    mean(median(eTrk_glbl_sig{iSig})));
end

% Bubble: sigma=1 is lowest, use sqrt(2) seems more reasonable

% Britton: sigma=sqrt(100) is the best
%%
iSigBest = 4;
pTrk_glbl = pTrk_glbl_sig{iSigBest};
score_glbl = score_glbl_sig{iSigBest};
eTrk_glbl = eTrk_glbl_sig{iSigBest};

figure;
scatter(score_glbl,mean(eTrk_glbl,2));
grid on;
title('prune.glbl, trkerr vs score','fontweight','bold','interpreter','none');

% Bubble: err and score negatively correlated

% Britton: err and score negatively correlated

%% Trajsmooth with sigma titration
%SIGMAS = [sqrt(0.1) sqrt(0.33) 1 sqrt(2) sqrt(5)];
SIGMAS = [sqrt(5) sqrt(20) sqrt(50) sqrt(100) sqrt(250)];
nSig = numel(SIGMAS);
[pTrk_traj_sig,tblSegments_traj_sig] = ...
  arrayfun(@(sig)Prune.applybesttraj2segs(pTrkFull,pTrkMD,'sigma',sig),SIGMAS,'uni',0);

[tf,loc] = tblismember(tTst,pTrkMD,MFTable.FLDSID);
assert(all(tf));
eTrk_traj_sig = cell(nSig,1);
for iSig=1:nSig
  pTrk = pTrk_traj_sig{iSig};
  pTrk = pTrk(loc,:);
  eTrk_traj_sig{iSig} = errGT(pTrk,pGT);
  fprintf('iSig %d, sig=%.3f. MeanMdn err: %.3f\n',iSig,SIGMAS(iSig),...
    mean(median(eTrk_traj_sig{iSig})));
end

% Bubble: sqrt(0.1) is the best, use 1 though just like with KDE b/c it 
% seems more reasonable and doesn't differ much

% Britton: sqrt(20) is the best

iSigBest = 2;
pTrk_traj = pTrk_traj_sig{iSigBest};
eTrk_traj = eTrk_traj_sig{iSigBest};

%% Alg comparison: End of the day

mednerrpts = [median(eTrk_med)' median(eTrk_kde)' median(eTrk_glbl)' median(eTrk_traj)']
mean(mednerrpts)

arrayfun(@(x)ranksum(eTrk_med(:,x),eTrk_traj(:,x)),1:npts)
friedman([eTrk_med(:) eTrk_traj(:)])
arrayfun(@(x)ranksum(eTrk_glbl(:,x),eTrk_traj(:,x)),1:npts)
friedman([eTrk_glbl(:) eTrk_traj(:)])

% Bubble: Median won but not by a lot, not sure it is significant

% Britton: kde, glbl-min, trajsmooth in a 3-way tie. all better than med

%% Trajsmooth, poslambda titration for best sigma 
%
% Note, it might be better to set a global poslambda rather than just using
% a factor applied to every separate window.
POSLAMBDA_FACS = [0.5 2 4];
nFac = numel(POSLAMBDA_FACS);

[pTrk_traj_lamfac,tblSegments_traj_lamfac] = ...
  arrayfun(@(lamfac)Prune.applybesttraj2segs( ...
    pTrkFull,pTrkMD,'sigma',SIGMAS(iSigBest),'poslambdafac',lamfac),POSLAMBDA_FACS,'uni',0);

eTrk_traj_lamfac = cell(nFac,1);
for iFac=1:nFac
  pTrk = pTrk_traj_lamfac{iFac};
  pTrk = pTrk(loc,:);
  eTrk_traj_lamfac{iFac} = errGT(pTrk,pGT);
  fprintf('iLamFac %d, lamfac=%.3f. MeanMdn err: %.3f\n',iFac,...
    POSLAMBDA_FACS(iFac),mean(median(eTrk_traj_lamfac{iFac})));
end

% Bub: lamfac=0.5 was best, slightly better than lamfac=1 (no lamfac).

% Britton: lamfac=4 was best => poslambda = .005

posLambdaBest = .005;

%% Trajsmooth, dampen titration for best sigma, poslambda

DAMPENS = [0.25 0.75]; % default is 0.5
nFac = numel(DAMPENS); 
[pTrk_traj_damp,tblSegments_traj_damp] = ...
  arrayfun(@(dmp)Prune.applybesttraj2segs(pTrkFull,pTrkMD,...
    'sigma',SIGMAS(iSigBest),'poslambda',posLambdaBest,...
    'dampen',dmp),DAMPENS,'uni',0);

eTrk_traj_damp = cell(nFac,1);
for iFac=1:nFac
  pTrk = pTrk_traj_damp{iFac};
  pTrk = pTrk(loc,:);
  eTrk_traj_damp{iFac} = errGT(pTrk,pGT);
  fprintf('idamp %d, dampen=%.3f. MeanMdn err: %.3f\n',iFac,...
    DAMPENS(iFac),mean(median(eTrk_traj_damp{iFac})));
end

% Bub: no real effect

% Britton: no real effect

%%
%%% Random %%%

%% make a trkfile for best pTrk_kde for import
pTrk_kde_best = pTrk_kde_sig{ISIGBEST}'; % [4xnfrm]
pTrk_kde_best = reshape(pTrk_kde_best,[2 2 2997]);
tfile = TrkFile(pTrk_kde_best);
tfile.save('mov7kdebestsig.trk');

%% Precisely set tracker trackres: original full replicates, plus best traj
% pruning

mov = repmat(7,nLbled,1);
frmLbl = find(tfLbled);
iTgt = ones(nLbled,1);
pTrk = pTrk_traj_best(frmLbl,:);
tblTrkRes = table(mov,frmLbl,iTgt,pTrk,...
  'VariableNames',{'mov' 'frm' 'iTgt' 'pTrk'});

tObj = lObj.tracker;
tObj.setAllTrackResTable(tblTrkRes,1:2);
tObj.trkPFull = pTrkFull(tfLbled,:,:);
tObj.vizLoadXYPrdCurrMovieTarget();
tObj.newLabelerFrame();
notify(lObj,'newTrackingResults');

%% Find pts where kde does poorly but traj does well
difftrk = eTrk_kde_best-eTrk_traj_best;
difftrk = sum(difftrk,2);
[~,idx] = sort(difftrk,'descend');
t = table(7*ones(nLbled,1),frmLbl,ones(nLbled,1),difftrk,...
  'VariableNames',{'mov' 'frm' 'iTgt' 'susp'});
t = t(idx,:);

%%
outlrFcn = tmpOutlierFcn(t);
lObj.suspSetComputeFcn(outlrFcn);
lObj.suspComputeUI();
