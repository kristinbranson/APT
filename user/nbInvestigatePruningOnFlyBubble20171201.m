%% 

% load proj

tLbl = lObj.labelGetMFTableLabeled();

%%
sPrm = lObj.tracker.sPrm;
roiRadius = sPrm.PreProc.TargetCrop.Radius;
tLbl = lObj.labelMFTableAddROI(tLbl,roiRadius);
%% get ims
trxCache = containers.Map();
tblLblConc = tLbl(:,[MFTable.FLDSID {'roi'}]);
tblLblConc = lObj.mftTableConcretizeMov(tblLblConc);
wbObj = WaitBarWithCancel('data read');
[Ilbl,nmasklbl] = CPRData.getFrames(tblLblConc,'wbObj',wbObj,...
  'trxCache',trxCache);

%% generate a trn/testset
movC = categorical(tLbl.mov);
tgtC = categorical(tLbl.iTgt);
grpC = movC.*tgtC;
cvPart = cvpartition(grpC,'kfold',7);
tfTrn = cvPart.training(1);
tfTst = cvPart.test(1);
unique(tfTrn+tfTst)
iTrn = find(tfTrn);
iTst = find(tfTst);
fprintf('%d groups. %d trn %d tst.\n',numel(unique(grpC)),numel(iTrn),numel(iTst));

tblTrn = tLbl(tfTrn,:);
tblTst = tLbl(tfTst,:);
nTrn = numel(iTrn);
nTst = numel(iTst);

%% Train
rng(0);
rc = RegressorCascade(sPrm);
Itrn = Ilbl(tfTrn,:);
bbtrn = CPRData.getBboxes2D(Itrn);
pTrn = tblTrn.pRoi;
othetasTrn = tblTrn.thetaTrx;
[~,~,p0trn,p0trninfo] = rc.trainWithRandInit(Itrn,bbtrn,pTrn,...
  'orientationThetas',othetasTrn);
save rc_trnbase.mat rc p0trn p0trninfo tblTrn pTrn;

%% Track Base discarded just to get p0trk p0trkinfo
rng(0);
Itrk = Ilbl(tfTst,:);
bbtrk = CPRData.getBboxes2D(Itrk);
wbObj = WaitBarWithCancel('tracking');
othetasTrk = tblTst.thetaTrx;
[p_t,pIidx,p0trk,p0trkinfo] = rc.propagateRandInit(Itrk,bbtrk,...
  sPrm.TestInit,'wbObj',wbObj,'orientationThetas',othetasTrk);
delete(wbObj);

save p0trkinfo pIidx p0trk p0trkinfo tblTst;

%% Track, keeping reps
rng(0);
Itrk = Ilbl(tfTst,:);
bbtrk = CPRData.getBboxes2D(Itrk);
wbObj = WaitBarWithCancel('tracking');

szassert(p0trk,[nTst*sPrm.TestInit.Nrep rc.prmModel.D]);
assert(isequal(pIidx,repmat(1:nTst,[1 sPrm.TestInit.Nrep])'));
p_t = rc.propagate(Itrk,bbtrk,p0trk,pIidx,'wbObj',wbObj);
delete(wbObj);

save pTrkFull p_t;

%% Prune prep
load rc_trnbase.mat;
load p0trkinfo
load pTrkFull;
sPrm = lObj.tracker.sPrm;
nTst = height(tblTst);

trkMdl = rc.prmModel;
trkD = trkMdl.D;
Tp1 = rc.nMajor+1;
pTrk = reshape(p_t,[nTst sPrm.TestInit.Nrep trkD Tp1]);
pTrkEnd = pTrk(:,:,:,end); % [nTst x nRep x trkD]

%% Prune: median
[pTrk_med,score_med,info_med] = Prune.median(pTrkEnd);

dTrk = pTrk_med-tblTst.pRoi;
dTrk = reshape(dTrk,[nTst trkMdl.nfids trkMdl.d]);
eTrk_med = sqrt(sum(dTrk.^2,3));
meanerrpt = mean(eTrk_med)
mean(meanerrpt)

figure;
scatter(score_med,mean(eTrk_med,2));
grid on;
title('prune.median, trkerr vs -(repl mad)','fontweight','bold','interpreter','none');

%% Prune: kde calib
[~,~,info] = Prune.maxdensity(pTrkEnd,'sigma',5);

kde_d2 = cat(1,info{:}); % [nrep*(nrep-1)/2*nTst x npt] pairwise dist^2 for each replicate/pt
h = figure;
axs = createsubplots(3,6);
bctrs = 0:1:25;
for i=1:17
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
xlim(axs(1),[0 30]);

% For FlyBub leg (tips), looks like scale of d2 is ~2

% Britton, scale of d2 is ~50-100

%% Prune: kde
SIGMAS = [sqrt(0.1) sqrt(0.33) 1 sqrt(2) sqrt(5)];
nSig = numel(SIGMAS);
[pTrk_kde_sig,score_kde_sig] = ...
  arrayfun(@(sig)Prune.maxdensity(pTrkEnd,'sigma',sig),SIGMAS,'uni',0);

eTrk_kde_sig = cell(nSig,1);
for iSig=1:nSig
  dTrk = pTrk_kde_sig{iSig}-tblTst.pRoi;
  dTrk = reshape(dTrk,[nTst trkMdl.nfids trkMdl.d]);
  eTrk_kde_sig{iSig} = sqrt(sum(dTrk.^2,3));
  fprintf('iSig %d, sig=%.3f. Mean err: %.3f\n',iSig,SIGMAS(iSig),mean(eTrk_kde_sig{iSig}(:)));
end

% sqrt(0.33) is the best, use 1 though about the same and looks more
% reasonable

% Britton mouse: best is sqrt(50)

iSigBest = 3;
pTrk_kde = pTrk_kde_sig{iSigBest};
score_kde = score_kde_sig{iSigBest};
eTrk_kde = eTrk_kde_sig{iSigBest};

figure;
scatter(score_kde,mean(eTrk_kde,2));
grid on;
title('prune.kde, trkerr vs score','fontweight','bold','interpreter','none');

%% Prune: global min calib
[~,~,info] = Prune.globalmin(pTrkEnd,'sigma',inf); % sigma should not matter

glbl_d2 = cat(1,info{:}); % [nrep*(nrep-1)/2*nTst x 1] pairwise dist^2 for each replicate
figure;
hist(glbl_d2,0:1:65);
grid on;
xlim([0 70]);

% looks like scale of d2 is ~5

% Britton: scale is ~100

%% Prune: global min
SIGMAS = [sqrt(1) sqrt(2) sqrt(5) sqrt(20)];
nSig = numel(SIGMAS);
[pTrk_glbl_sig,score_glbl_sig] = ...
  arrayfun(@(sig)Prune.globalmin(pTrkEnd,'sigma',sig),SIGMAS,'uni',0);

eTrk_glbl_sig = cell(nSig,1);
for iSig=1:nSig
  dTrk = pTrk_glbl_sig{iSig}-tblTst.pRoi;
  dTrk = reshape(dTrk,[nTst trkMdl.nfids trkMdl.d]);
  eTrk_glbl_sig{iSig} = sqrt(sum(dTrk.^2,3));
  fprintf('iSig %d, sig=%.3f. Mean err: %.3f\n',iSig,SIGMAS(iSig),mean(eTrk_glbl_sig{iSig}(:)));
end

% sigma=1 is lowest but use sqrt(2)

% Britton: sigma=sqrt(100) is the best

iSigBest = 4;
pTrk_glbl = pTrk_glbl_sig{iSigBest};
score_glbl = score_glbl_sig{iSigBest};
eTrk_glbl = eTrk_glbl_sig{iSigBest};

figure;
scatter(score_glbl,mean(eTrk_glbl,2));
grid on;
title('prune.glbl, trkerr vs score','fontweight','bold','interpreter','none');

%% Alg comparison: End of the day

meanerrpts = [mean(eTrk_med)' mean(eTrk_kde)' mean(eTrk_glbl)']
mean(meanerrpts)

figure
axs = createsubplots(1,2);
for ipt=1:2
  axes(axs(ipt));
  x = [eTrk_med(:,ipt) eTrk_kde(:,ipt) eTrk_glbl(:,ipt)];
  boxplot(x);
  grid on
end

ranksum(eTrk_med(:,1),eTrk_glbl(:,1))
ranksum(eTrk_med(:,2),eTrk_glbl(:,2))


% Bubble: Median won but nothing reaches significance, all 3 methods similar

% Britton: glbl-min won (no significance compared to kde), but both better than median primarily on pt 1

%% Traj Smoothing
PROJFILE = 'reach_all_mice_including_BPN_perturb_TRAINING_AL_trained.lbl';
IMOV = 7;
TRKFILE = 'mov7full.trk';
lbl = load(PROJFILE,'-mat');
trk = load(TRKFILE,'-mat');
%%
lpos = SparseLabelArray.full(lbl.labeledpos{IMOV});
pTrkFull = trk.pTrkFull;
N = size(lpos,3);
K = size(pTrkFull,3);
pLbl = reshape(lpos,[4 N])'; % [Nx4]
pTrkFull = reshape(pTrkFull,[4 K N]);
pTrkFull = permute(pTrkFull,[3 2 1]);

tfLbled = all(~isnan(pLbl),2);
nLbled = nnz(tfLbled);
fprintf('%d labeled frames in mov %d.\n',nLbled,IMOV);
pLbled = pLbl(tfLbled,:); 
clear pLbl;
%%
SIGMAS = [sqrt(0.33) sqrt(1) sqrt(5) sqrt(20) sqrt(100)];
nSig = numel(SIGMAS);
pTrk_traj_sig = ...
  arrayfun(@(sig)Prune.besttraj(pTrkFull,'sigma',sig),SIGMAS,'uni',0);
pTrk_kde_sig = ...
  arrayfun(@(sig)Prune.maxdensity(pTrkFull,'sigma',sig),SIGMAS,'uni',0);
pTrk_med_sig = ...
  arrayfun(@(sig)Prune.median(pTrkFull),SIGMAS,'uni',0);

eTrk_traj_sig = cell(nSig,1);
eTrk_kde_sig = cell(nSig,1);
eTrk_med_sig = cell(nSig,1);
for iSig=1:nSig
  dTrkLbled = pTrk_traj_sig{iSig}(tfLbled,:)-pLbled;
  dTrkLbled = reshape(dTrkLbled,[nLbled 2 2]);
  eTrk_traj_sig{iSig} = sqrt(sum(dTrkLbled.^2,3));
  
  dTrkLbled = pTrk_kde_sig{iSig}(tfLbled,:)-pLbled;
  dTrkLbled = reshape(dTrkLbled,[nLbled 2 2]);
  eTrk_kde_sig{iSig} = sqrt(sum(dTrkLbled.^2,3));
  
  dTrkLbled = pTrk_med_sig{iSig}(tfLbled,:)-pLbled;
  dTrkLbled = reshape(dTrkLbled,[nLbled 2 2]);
  eTrk_med_sig{iSig} = sqrt(sum(dTrkLbled.^2,3));  
  
  fprintf('iSig %d, sig=%.3f. Mean err traj/kde/med: %.3f %.3f %.3f\n',...
    iSig,SIGMAS(iSig),...
    mean(eTrk_traj_sig{iSig}(:)),...
    mean(eTrk_kde_sig{iSig}(:)),...
    mean(eTrk_med_sig{iSig}(:)) );
end

%% check significance
ISIGBEST = 4;
pTrk_traj_best = pTrk_traj_sig{ISIGBEST};
eTrk_traj_best = eTrk_traj_sig{ISIGBEST};
eTrk_kde_best = eTrk_kde_sig{ISIGBEST};
eTrk_med_best = eTrk_med_sig{ISIGBEST};
mean(eTrk_traj_best)
mean(eTrk_kde_best)
mean(eTrk_med_best)
arrayfun(@(ipt)ranksum(eTrk_traj_best(:,ipt),eTrk_kde_best(:,ipt)),1:2)
% traj/kde diff not significant
for ipt=1:2
  kruskalwallis([eTrk_traj_best(:,ipt) eTrk_kde_best(:,ipt)],[]);
end

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
notify(tObj,'newTrackingResults');

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

%% lambdas
LAMBDAS0 = [.013 .0047 .002 .0012 .00052]; % originally generated
%SIGMAS = [sqrt(0.33) sqrt(1) sqrt(5) sqrt(20) sqrt(100)];
nSig = numel(SIGMAS);

lambdas = LAMBDAS0*4;
assert(numel(lambdas)==nSig);

pTrk_traj_sig_lam2 = ...
  arrayfun(@(sig,lam)Prune.besttraj(pTrkFull,'sigma',sig,'poslambda',lam),...
  SIGMAS,lambdas,'uni',0);
% pTrk_kde_sig = ...
%   arrayfun(@(sig)Prune.maxdensity(pTrkFull,'sigma',sig),SIGMAS,'uni',0);
% pTrk_med_sig = ...
%   arrayfun(@(sig)Prune.median(pTrkFull),SIGMAS,'uni',0);

eTrk_traj_sig_lam2 = cell(nSig,1);
% eTrk_kde_sig = cell(nSig,1);
% eTrk_med_sig = cell(nSig,1);
for iSig=1:nSig
  dTrkLbled = pTrk_traj_sig_lam2{iSig}(tfLbled,:)-pLbled;
  dTrkLbled = reshape(dTrkLbled,[nLbled 2 2]);
  eTrk_traj_sig_lam2{iSig} = sqrt(sum(dTrkLbled.^2,3));
    
  fprintf('iSig %d, sig=%.3f. Mean err origlam/newlam: %.3f %.3f\n',...
    iSig,SIGMAS(iSig),...
    mean(eTrk_traj_sig{iSig}(:)),...
    mean(eTrk_traj_sig_lam2{iSig}(:)));
end

% 4x lambdas
% iSig 1, sig=0.574. Mean err origlam/newlam: 8.769 8.452
% iSig 2, sig=1.000. Mean err origlam/newlam: 8.796 8.405
% iSig 3, sig=2.236. Mean err origlam/newlam: 8.563 8.186
% iSig 4, sig=4.472. Mean err origlam/newlam: 8.374 8.096
% iSig 5, sig=10.000. Mean err origlam/newlam: 8.578 8.689

% % 3x lambdas
% iSig 1, sig=0.574. Mean err origlam/newlam: 8.769 8.492
% iSig 2, sig=1.000. Mean err origlam/newlam: 8.796 8.420
% iSig 3, sig=2.236. Mean err origlam/newlam: 8.563 8.161
% iSig 4, sig=4.472. Mean err origlam/newlam: 8.374 8.077
% iSig 5, sig=10.000. Mean err origlam/newlam: 8.578 8.670

% double lambdas
% iSig 1, sig=0.574. Mean err origlam/newlam: 8.769 8.508
% iSig 2, sig=1.000. Mean err origlam/newlam: 8.796 8.484
% iSig 3, sig=2.236. Mean err origlam/newlam: 8.563 8.550
% iSig 4, sig=4.472. Mean err origlam/newlam: 8.374 8.386
% iSig 5, sig=10.000. Mean err origlam/newlam: 8.578 8.588

% half lambdas
% iSig 1, sig=0.574. Mean err origlam/newlam: 8.769 8.508
% iSig 2, sig=1.000. Mean err origlam/newlam: 8.796 8.484
% iSig 3, sig=2.236. Mean err origlam/newlam: 8.563 8.550
% iSig 4, sig=4.472. Mean err origlam/newlam: 8.374 8.386
% iSig 5, sig=10.000. Mean err origlam/newlam: 8.578 8.588


