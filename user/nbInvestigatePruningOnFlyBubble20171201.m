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
iSigBest = 2;
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
IPT = 17;
x = [eTrk_med(:,IPT) eTrk_kde(:,IPT) eTrk_glbl(:,IPT)];
boxplot(x)

% MEDIAN WON
