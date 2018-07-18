%% Compile xv results over rounds
nE = 3016;
nH = 3009;
nPch = 39;
xverrE = nan(nE,10,nPch,0);
xverrH = nan(nH,10,nPch,0);
pchsE = cell(0,nPch);
pchsH = cell(0,nPch);
xvresE = cell(nPch,0);
xvresH = cell(nPch,0);

[xverrE(:,:,:,end+1),pchsE(end+1,:),xvresE(:,end+1)] = HPOptim.loadXVres(...
  'pch01','rnd01_easy_fold1',...
  'xv_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm0_20180713_%s_20180713T*.mat',...
  'xvbase','xv_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm0_20180713_20180714T070529.mat');
[xverrH(:,:,:,end+1),pchsH(end+1,:),xvresH(:,end+1)] = HPOptim.loadXVres(...
  'pch01','rnd01_hard_fold1',...
  'xv_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm0_20180713_%s_20180713T*.mat',...
  'xvbase','xv_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm0_20180713_20180714T070315.mat');

[xverrE(:,:,:,end+1),pchsE(end+1,:),xvresE(:,end+1)] = HPOptim.loadXVres(...
  'pch1_20180714','rnd02_easy_fold1',...
  'xv_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm1_20180714_%s_20180714T*.mat',...
  'xvbase','xv_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm1_20180714_20180714T135001.mat');
[xverrH(:,:,:,end+1),pchsH(end+1,:),xvresH(:,end+1)] = HPOptim.loadXVres(...
  'pch1_20180714','rnd02_hard_fold1',...
  'xv_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm1_20180714_%s_20180714T*.mat',...
  'xvbase','xv_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm1_20180714_20180714T135221.mat');

[xverrE(:,:,:,end+1),pchsE(end+1,:),xvresE(:,end+1)] = HPOptim.loadXVres(...
  'pch2_20180715','rnd03_easy_fold1',...
  'xv_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm2_20180715_%s_201807*.mat',...
  'xvbase','xv_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm2_20180715_20180715T212002.mat');
[xverrH(:,:,:,end+1),pchsH(end+1,:),xvresH(:,end+1)] = HPOptim.loadXVres(...
  'pch2_20180715','rnd03_hard_fold1',...
  'xv_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm2_20180715_%s_201807*.mat',...
  'xvbase','xv_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm2_20180715_20180715T212108.mat');

[xverrE(:,:,:,end+1),pchsE(end+1,:),xvresE(:,end+1)] = HPOptim.loadXVres(...
  'pch3_20180716','rnd04_easy_fold1',...
  'xv_sh_trn4523_gt080618_made20180627_cacheddata_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm3_20180716_%s_201807*.mat',...
  'xvbase','xv_sh_trn4523_gt080618_made20180627_cacheddata_hpo_outer3_easy_fold01_tblTrn_hpo_outer3_easy_fold01_inner3_prm3_20180716_20180716T182048.mat');
[xverrH(:,:,:,end+1),pchsH(end+1,:),xvresH(:,end+1)] = HPOptim.loadXVres(...
  'pch3_20180716','rnd04_hard_fold1',...
  'xv_sh_trn4523_gt080618_made20180627_cacheddata_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm3_20180716_%s_201807*.mat',...
  'xvbase','xv_sh_trn4523_gt080618_made20180627_cacheddata_hpo_outer3_hard_fold01_tblTrn_hpo_outer3_hard_fold01_inner3_prm3_20180716_20180716T202115.mat');

%%
nRounds = size(xverrE,4);
isequal(pchsE,pchsH,repmat(pchsE(1,:),nRounds,1))
pchs = pchsE(1,:)';
size(xverrE)
size(xverrH)

%% scores by round
tblres = cell(nRounds,2); % round, easy/hard
scores = nan(nPch,nRounds,2); % pch, round, easy/hard
for iRnd=1:nRounds
  tblres{iRnd,1} = HPOptim.pchScores(xverrE(:,:,:,iRnd),pchs);
  tblres{iRnd,2} = HPOptim.pchScores(xverrH(:,:,:,iRnd),pchs);
  scores(:,iRnd,1) = tblres{iRnd,1}.score;
  scores(:,iRnd,2) = tblres{iRnd,2}.score;
end

%% compare best pchs
IRND = 1;
NBEST = 12;
tblres{IRND,1}(1:NBEST,:)
tblres{IRND,2}(1:NBEST,:)

tE = tblres{IRND,1}(1:NBEST,{'score' 'nptimprovedfull' 'pch'});
tH = tblres{IRND,2}(1:NBEST,{'score' 'nptimprovedfull' 'pch'});
tE.Properties.VariableNames{2} = 'nptimp';
tH.Properties.VariableNames{2} = 'nptimp';
tE
tH

%% Convergence
DOSAVE = true;
SAVEDIR = 'figs';

hFig = [];

JITDX = 0.2;
CLRS = {[0 0 1] [0 0.75 0]}; % easy/hard
MRKRSZ = 30;

hFig(end+1) = figure(11);
hfig = hFig(end);
clf(hfig);
set(hfig,'Name','Convergence','Position',[2561 401 1920 1124]);
ax = axes;
hold(ax,'on');
grid(ax,'on');
for iRnd=1:nRounds
  for iEH=1:2
    x = (nRounds+1)*(iEH-1) + repmat(iRnd,nPch,1);
    x = x + 2*JITDX*(rand(nPch,1)-0.5);
    y = scores(:,iRnd,iEH);
    plot(ax,x,y,'.','markersize',MRKRSZ,'color',CLRS{iEH});
  end
end
xlim(ax,[-0.5 10.5]);
ylim(ax,[-20 10]);
set(ax,'XTick',[1:4 6:9],'XTickLabel',...
  {'Rnd1/easy' 'Rnd2/easy' 'Rnd3/easy' 'Rnd4/easy' 'Rnd1/hard' 'Rnd2/hard' 'Rnd3/hard' 'Rnd4/hard'},...
  'XTickLabelRotation',0,'fontsize',18);
ylabel(ax,'XV prctile improvement score');
set(ax,'fontweight','bold');
title(ax,'Hyperparameter Optimization: "Convergence"?');

if DOSAVE
  for i=1:numel(hFig)
    h = figure(hFig(i));
    fname = h.Name;
    hgsave(h,fullfile(SAVEDIR,[fname '.fig']));
    set(h,'PaperOrientation','landscape','PaperType','arch-d');
    print(h,'-dpdf',fullfile(SAVEDIR,[fname '.pdf']));  
    print(h,'-dpng','-r300',fullfile(SAVEDIR,[fname '.png']));   
    fprintf(1,'Saved %s.\n',fname);
  end
end

%%
%HPOptim.genNewPrmFile('prm0_20180713.mat','prm1_20180714.mat','pch01',...
%  {'TwoLMRad_up';'FernsDepth_up2'});
% HPOptim.genNewPrmFile('prm1_20180714.mat','prm2_20180715.mat',...
%   'pch1_20180714',{'NumMajorIter_up';'RegFactor_dn'})
HPOptim.genNewPrmFile('prm2_20180715.mat','prm3_20180716.mat',...
  'pch2_20180715',{'FernThresholdRad_dn'})

%%
% HPOptim.genAndWritePchs('prm1_20180714.mat','pch1_20180714',{});
% HPOptim.genAndWritePchs('prm2_20180715.mat','pch2_20180715',{});
HPOptim.genAndWritePchs('prm3_20180716.mat','pch3_20180716',{});

%%
DOSAVE = false;
SAVEDIR = 'figs';
PTILES = [60 90];

xverrnormE = xverrE./median(xverrE(:,:,1),1);
xverrnormH = xverrH./median(xverrH(:,:,1),1);
xverrnormEmn = cat(2,mean(xverrnormE(:,1:5,:),2),mean(xverrnormE(:,6:10,:),2));
xverrnormHmn = cat(2,mean(xverrnormH(:,1:5,:),2),mean(xverrnormH(:,6:10,:),2));
assert(isequal(pchsE,pchsH));
pchNames = pchsE;

hFig = [];

hFig(5) = figure(15);
hfig = hFig(5);
set(hfig,'Name','easyzoom vw1','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(...
  xverrnormE(:,1:5,:),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(6) = figure(16);
hfig = hFig(6);
set(hfig,'Name','easyzoom vw2','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(...
  xverrnormE(:,6:10,:),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(7) = figure(17);
hfig = hFig(7);
set(hfig,'Name','hardzoom vw1','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(...
  xverrnormH(:,1:5,:),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(8) = figure(18);
hfig = hFig(8);
set(hfig,'Name','hardzoom vw2','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(...
  xverrnormH(:,6:10,:),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(9) = figure(19);
hfig = hFig(9);
set(hfig,'Name','easyzoom mean','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(xverrnormEmn,'hFig',hfig,...
  'ptNames',{'vw1' 'vw2'},...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(10) = figure(20);
hfig = hFig(10);
set(hfig,'Name','hardzoom mean','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(xverrnormHmn,'hFig',hfig,...
  'ptNames',{'vw1' 'vw2'},...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

if DOSAVE
  for i=1:numel(hFig)
    h = figure(hFig(i));
    fname = h.Name;
    hgsave(h,fullfile(SAVEDIR,[fname '.fig']));
    set(h,'PaperOrientation','landscape','PaperType','arch-d');
    print(h,'-dpdf',fullfile(SAVEDIR,[fname '.pdf']));  
    print(h,'-dpng','-r300',fullfile(SAVEDIR,[fname '.png']));   
    fprintf(1,'Saved %s.\n',fname);
  end
end

%%




PCHDIR = 'pch';
XVRESDIR = 'xvruns20180710';

dd = dir(fullfile(PCHDIR,'*.m'));
pchs = {dd.name}';
npch = numel(pchs);

for i=1:npch
  fprintf(1,'%s\n',pchs{i});
  type(fullfile(PCHDIR,pchs{i}));
end



%%
DOSAVE = false;
SAVEDIR = 'figs';
PTILES = [50 75 90 95 98];

xverrbasemedn = median(xverrE(:,:,1),1);
xverrnorm = xverrE./xverrbasemedn;
pchNames = pchsE;

hFig = [];
xverrplot = xverrnorm;
xverrplotmean = mean(xverrplot,2);

hFig(1) = figure(11);
hfig = hFig(1);
set(hfig,'Name','easyall','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurves(xverrplot(:,:,:,:,1),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(2) = figure(12);
hfig = hFig(2);
set(hfig,'Name','hardall','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurves(xverrplot(:,:,:,:,2),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(3) = figure(13);
hfig = hFig(3);
set(hfig,'Name','easymean','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurves(xverrplotmean(:,:,:,:,1),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(4) = figure(14);
hfig = hFig(4);
set(hfig,'Name','hardmean','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurves(xverrplotmean(:,:,:,:,2),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(5) = figure(15);
hfig = hFig(5);
set(hfig,'Name','easyzoom vw1','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(xverrplot(:,:,1,:,1),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(6) = figure(16);
hfig = hFig(6);
set(hfig,'Name','easyzoom vw2','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(xverrplot(:,:,2,:,1),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(7) = figure(17);
hfig = hFig(7);
set(hfig,'Name','hardzoom vw1','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(xverrplot(:,:,1,:,2),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(8) = figure(18);
hfig = hFig(8);
set(hfig,'Name','hardzoom vw2','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurvesZoomed(xverrplot(:,:,2,:,2),'hFig',hfig,...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(9) = figure(19);
hfig = hFig(9);
set(hfig,'Name','easyzoom mean','Position',[2561 401 1920 1124]);
xvepm = squeeze(xverrplotmean);
[~,ax] = GTPlot.ptileCurvesZoomed(xvepm(:,:,:,1),'hFig',hfig,...
  'ptNames',{'vw1' 'vw2'},...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

hFig(10) = figure(20);
hfig = hFig(10);
set(hfig,'Name','hardzoom mean','Position',[2561 401 1920 1124]);
xvepm = squeeze(xverrplotmean);
[~,ax] = GTPlot.ptileCurvesZoomed(xvepm(:,:,:,2),'hFig',hfig,...
  'ptNames',{'vw1' 'vw2'},...
  'setNames',pchNames,...
  'ptiles',PTILES,...
  'ylimcapbase',true,...
  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});

if DOSAVE
  for i=1:numel(hFig)
    h = figure(hFig(i));
    fname = h.Name;
    hgsave(h,fullfile(SAVEDIR,[fname '.fig']));
    set(h,'PaperOrientation','landscape','PaperType','arch-d');
    print(h,'-dpdf',fullfile(SAVEDIR,[fname '.pdf']));  
    print(h,'-dpng','-r300',fullfile(SAVEDIR,[fname '.png']));   
    fprintf(1,'Saved %s.\n',fname);
  end
end

