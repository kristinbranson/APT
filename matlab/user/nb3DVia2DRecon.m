%% Load 2D and 3D tracking res
TRK2D = {
  'rf2dLtype1_pp.trk'
  'rf2dRtype1_pp.trk'
  'rf2dBtype1_pp.trk'
  };

TRK2D = {
  'rf2dL_5exps_initUseFF.trk'
  'rf2dR_5exps_initUseFF.trk'
  'rf2dB_5exps_initUseFF.trk'
  };

TRK3D = 'trackResPruned_goBig_20170117.mat';

trk2d = cellfun(@(x)load(x,'-mat'),TRK2D);
trk3d = load(TRK3D);

%TRNDATA = 'trnData3D_20170106.mat';
TRNDATA = 'trnData3D_20170117.mat';
trnData = load(TRNDATA);

%% Compute consistency of 2D res by computing RE err
assert(strcmp(trnData.tFPtrn.expID{1},'jun22'));
crig = trnData.crigAll(1);

FRMS = 1:2e3;
nFRMS = numel(FRMS);
iPt2dTrk_to_full_cell = {RF.PTS_LSIDE RF.PTS_RSIDE};
XLre = nan(3,18,nFRMS);
dre = nan(18,nFRMS);
for iVwSide = [1 2]
  iPt2dTrk_to_full = iPt2dTrk_to_full_cell{iVwSide};
  for ipt2dTrk = 1:9
    iptFull = iPt2dTrk_to_full(ipt2dTrk);
    
    xy2dTrkSde = squeeze(trk2d(iVwSide).pTrk(ipt2dTrk,:,FRMS))';
    xy2dTrkBot = squeeze(trk2d(3).pTrk(iptFull,:,FRMS))';
    szassert(xy2dTrkSde,[nFRMS 2]);
    szassert(xy2dTrkBot,[nFRMS 2]);
    rc2dTrkSde = xy2dTrkSde(:,[2 1]);
    rc2dTrkBot = xy2dTrkBot(:,[2 1]);
    
    switch iVwSide
      case 1
        [XLre(:,iptFull,:),~,dre(iptFull,:)] = ...
          crig.stereoTriangulateCropped(rc2dTrkSde,rc2dTrkBot,'L','B');
      case 2
        [~,Xtmp,dre(iptFull,:)] = ...
          crig.stereoTriangulateCropped(rc2dTrkSde,rc2dTrkBot,'R','B');
        XLre(:,iptFull,:) = crig.viewXformCPR(Xtmp,3,1);
      otherwise
        assert(false);
    end
  end
end

XLtrk3D = trk3d.pTstTRedSingle(:,:,end)';
szassert(XLtrk3D,[54 nFRMS]);
XLtrk3D = reshape(XLtrk3D,[18 3 nFRMS]);
XLtrk3D = permute(XLtrk3D,[2 1 3]);

% compute difference between 
szassert(XLre,size(XLtrk3D));
d3d = XLre-XLtrk3D;
d3d = sqrt(sum(d3d.^2,1));
szassert(d3d,[1 18 nFRMS]);
d3d = squeeze(d3d)'; % [nFRMSx18]. L2 in 3D (L cam) between 3dtracked and 2dtracked+re points

dre = dre';
szassert(dre,[nFRMS 18]); % stereoRecon error for 2dtracked pts

%%
[mean(dre,1)' mean(d3d,1)']

%% Correlation between recon err and 2d/3d err
hF = figure('windowstyle','docked');
colormap(hF,'jet');
hS = scatter(mean(dre,1)',mean(d3d,1)',30,1:18,'filled');
grid on
xlabel('err stereoRecon','fontweight','bold');
ylabel('dist 2dtrk vs 3dtrk','fontweight','bold');
title('trk dist vs err stereoRecon. Color is pt index.','fontweight','bold');
colorbar

%% timeseries
hF = figure('windowstyle','docked');
ax = axes;
IPT = 4;
plot(FRMS,dre(:,IPT),FRMS,d3d(:,IPT));
grid on;
hF = figure('windowstyle','docked');
scatter(dre(:,IPT),d3d(:,IPT));

%% hists
hF = figure('windowstyle','docked');
ax = createsubplots(1,2);
hist(ax(1),dre(:,IPT));
hist(ax(2),d3d(:,IPT));

%% find frms where i) dre is small and ii) d3d is big
dre_plo = prctile(dre(:,IPT),5);
d3d_phi = prctile(d3d(:,IPT),80);
tf = dre(:,IPT)<=dre_plo & d3d(:,IPT)>=d3d_phi;
nnz(tf)
find(tf)'

%% What is the scale of 3d camL coords? 
szassert(XLtrk3D,[3 18 nFRMS]);
dpts12 = squeeze(XLtrk3D(:,4,:)-XLtrk3D(:,5,:));
szassert(dpts12,[3 nFRMS]);
dpts12 = sqrt(sum(dpts12.^2,1));
mean(dpts12)

%% Movie: just use 2d reconcile.
RF.makeTrkMovie3D(aviCell{2}','3dFrom2d_5expsInitUseFF.avi',...
  'trkRes3D',struct('X',XLre,'crig',crig,'frm',1:2e3));

%% Compute diffs
tr1 = load('trackResPruned_goBig_20170117.mat');
%tr2 = load('trackRes3Dfrom2D_type1pp_20170123.mat');
tr2 = load('trackRes3Dfrom2D_5expsInitUseFF_20170123.mat');
tr3 = load('trackRes3Dfrom2D_5exps_20170123.mat');
nFRMS = 2e3;
XLtrk3D = tr1.pTstTRedSingle(:,:,end)';
szassert(XLtrk3D,[54 nFRMS]);
XLtrk3D = reshape(XLtrk3D,[18 3 nFRMS]);
XLtrk3D = permute(XLtrk3D,[2 1 3]);
%% 
d12 = squeeze(sqrt(sum((XLtrk3D-tr2.XLre).^2,1)));
d23 = squeeze(sqrt(sum((tr2.XLre-tr3.XLre).^2,1)));
d13 = squeeze(sqrt(sum((XLtrk3D-tr3.XLre).^2,1)));
szassert(d12,[18 2e3]);
szassert(d23,[18 2e3]);
szassert(d13,[18 2e3]);
d12srt = sort(d12(:),'descend');
d23srt = sort(d23(:),'descend');
d13srt = sort(d13(:),'descend');
NWORST = 50;
d12worst = d12srt(1:NWORST);
d23worst = d23srt(1:NWORST);
d13worst = d13srt(1:NWORST);
[d12worst_ipt,d12worst_frm] = arrayfun(@(x)find(x==d12),d12worst);
[d23worst_ipt,d23worst_frm] = arrayfun(@(x)find(x==d23),d23worst);
[d13worst_ipt,d13worst_frm] = arrayfun(@(x)find(x==d13),d13worst);
tbl12 = table(d12worst_frm,d12worst_ipt,repmat({'d12'},NWORST,1),d12worst,...
  'VariableNames',{'frm' 'pt' 'type' 'dist3d'});
tbl23 = table(d23worst_frm,d23worst_ipt,repmat({'d23'},NWORST,1),d23worst,...
  'VariableNames',{'frm' 'pt' 'type' 'dist3d' });
tbl13 = table(d13worst_frm,d13worst_ipt,repmat({'d13'},NWORST,1),d13worst,...
  'VariableNames',{'frm' 'pt' 'type' 'dist3d'});
tbl = [tbl12;tbl23;tbl13];
tbl = sortrows(tbl,{'frm','pt'});

%%
% FRM = 424;
% X_1 = squeeze(XLtrk3D(:,4,FRM));
% X_2 = squeeze(tr2.XLre(:,4,FRM));
% X_3 = squeeze(tr3.XLre(:,4,FRM));
% isequal(tr1.crigTrack,tr2.crig,tr3.crig)
% [rL1,cL1] = tr2.crig.projectCPR(X_1,1);
% [rL2,cL2] = tr2.crig.projectCPR(X_2,1);
% [rL3,cL3] = tr2.crig.projectCPR(X_3,1);
% xy = [cL1 rL1;cL2 rL2;cL3 cL3];
% mr = MovieReader;
% mr.open('f:\Dropbox\MultiViewFlyLegTracking\trackingJun22-11-02\bias_video_cam_0_date_2016_06_22_time_11_02_02_v001.avi');
% im = mr.readframe(FRM);
% figure('windowstyle','docked');
% imagesc(im);
% hold on;
% MRKR = {'o' 's' '^'};
% for i=1:3
%   plot(xy(i,1),xy(i,2),'r','Marker',MRKR{i});
% end

%%
tr2 = load('trackRes3Dfrom2D_type12pp_20170123.mat');
RF.makeTrkMovie3D(aviCell{2}','3dFrom2d_type12pp_frms400_500.avi',...
  'trkRes3D',struct('X',tr2.XLre(:,:,400:500),'crig',tr2.crig,'frm',400:500));

%%
tr3 = load('trackRes3Dfrom2D_5exps_20170123.mat');


%% run some optims

