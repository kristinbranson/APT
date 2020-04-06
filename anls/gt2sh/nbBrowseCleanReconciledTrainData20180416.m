load trnDataSH_20180417T094303.mat;
load trnDataSH_20180417T094303_IFinalReconciled.mat;

%% Remove outliers, nan rows
tLook = tGT;
hFig = figure;
axs = createsubplots(4,5);
for i=1:20
  axes(axs(i));
  histogram(tLook.pLbl(:,i));
end

%%
idxOut = find(tFinalReconciled.pLbl(:,3)>1e3)
idxNan = find(any(isnan(tFinalReconciled.pLbl),2))
tFinalReconciled([idxOut;idxNan],:) = [];
IFinalReconciled([idxOut;idxNan],:) = [];

idxOut = find(tGT.pLbl(:,2)>400 | tGT.pLbl(:,3)>400)
tGT(idxOut,:) = [];
Igt(idxOut,:) = [];

% NaNs in gt lbls are ok
%idxNan = find(any(isnan(tGT.pLbl),2))

%%
save -v7.3 trnDataSH_Apr18.mat -append tGT Igt;
%save -v7.3 trnDataSH_Apr18.mat tFinalReconciled IFinalReconciled tGT Igt sPrm;

%% 
load trnDataSH_Apr18.mat;
load trnSplits_20180418T173507.mat;

%% normalized shapes
mdl = struct('D',10,'d',2,'nfids',5);
n = height(tFinalReconciled);
nGT = height(tGT);
pLbl = nan(n,10,2); % n,coords,ivw
pLblN = nan(n,10,2); 
pLblGT = nan(nGT,10,2);
pLblGTN = nan(nGT,10,2);
for ivw=1:2  
  bbvw = makeBBoxes(IFinalReconciled(:,ivw));
  bbvwGT = makeBBoxes(Igt(:,ivw));
  
  pLbl(:,:,ivw) = tFinalReconciled.pLbl(:,[1:5 11:15]+5*(ivw-1));
  pLblGT(:,:,ivw) = tGT.pLbl(:,[1:5 11:15]+5*(ivw-1));
  pLblN(:,:,ivw) = shapeGt('projectPose',mdl,pLbl(:,:,ivw),bbvw);
  pLblGTN(:,:,ivw) = shapeGt('projectPose',mdl,pLblGT(:,:,ivw),bbvwGT);  
end

%% Montage: labels vs images
I2viz = IFR_crop2;
xy2viz = xyLbl_FR_crop2;

tfTrn1 = trnSets(:,1);
nTrn = nnz(tfTrn1)
idstrs = strcat(tFinalReconciled.lblCat(tfTrn1),'|',numarr2trimcellstr(find(tfTrn1)));
hFigVw1 = figure('position',[1 41 2560 1484]);
hFigVw2 = figure('position',[2561 401 1920 1124]);

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;

easymontage(I2viz(tfTrn1,1),reshape(xy2viz(tfTrn1,:,:,1),nTrn,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs,'hFig',hFigVw1,...
  'doroi',true);
easymontage(I2viz(tfTrn1,2),reshape(xy2viz(tfTrn1,:,:,2),nTrn,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs,'hFig',hFigVw2,...
  'doroi',true);

%% Montage GT: labels vs images
idstrs = strcat(tGT.lblCat,'|',numarr2trimcellstr((1:height(tGT))'));
hFigVw1 = figure('position',[1 41 2560 1484]);
hFigVw2 = figure('position',[2561 401 1920 1124]);

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;

ROWS = 1:100;

easymontage(Igt(ROWS,1),tGT.pLbl(ROWS,[1:5 11:15]),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs(ROWS),'hFig',hFigVw1,...
  'doroi',true);
easymontage(Igt(ROWS,2),tGT.pLbl(ROWS,[6:10 16:20]),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs(ROWS),'hFig',hFigVw2,...
  'doroi',true);

%% Centroid overlay
% note: ~1000 rows that are 1024x1024
xyLbl = reshape(pLbl,[n 5 2 2]); % n, pt, x/y, ivw
xyLblN = reshape(pLblN,[n 5 2 2]);
xygtLbl = reshape(pLblGT,[nGT 5 2 2]);
xygtLblN = reshape(pLblGTN,[nGT 5 2 2]);
lblCents = squeeze(mean(xyLbl,2)); % [nx2x2]. i, x/y, vw
lblNCents = squeeze(mean(xyLblN,2));
gtLblCents = squeeze(nanmean(xygtLbl,2));
gtLblNCents = squeeze(nanmean(xygtLblN,2));

bbAllVw1 = makeBBoxes([IFinalReconciled(:,1);Igt(:,1)]);
bbAllVw2 = makeBBoxes([IFinalReconciled(:,2);Igt(:,2)]);
assert(isequal(bbAllVw1,bbAllVw2));
tfBigIm = ismember(bbAllVw1,[1 1 1024 1024],'rows');
bigImStr = repmat({''},size(tfBigIm));
bigImStr(tfBigIm) = {'BIG'};
gall = [tFinalReconciled.lblCat;strcat('gt',tGT.lblCat)];
gall = strcat(bigImStr,gall);

hFig = figure('position',[2561 401 1920 1124]);
axs = createsubplots(2,2,.1);
axs = reshape(axs,2,2);
for iVw=1:2
  ax = axs(1,iVw);
  axes(ax);
  gscatter([lblCents(:,1,iVw);gtLblCents(:,1,iVw)],...
           [lblCents(:,2,iVw);gtLblCents(:,2,iVw)],gall);
  hold(ax,'on');
  h = plot(ax,[0 1024],[512 512],'k-');
  h.Annotation.LegendInformation.IconDisplayStyle = 'off';
  h = plot(ax,[768 768],[0 1024],'k-');
  h.Annotation.LegendInformation.IconDisplayStyle = 'off';
  axis(gca,[0 1024 0 1024]);
  grid(gca,'on');
  axis(gca,'ij');
  tstr = sprintf('view%d',iVw); 
  if iVw==1
    tstr = ['Label centroids: ' tstr];
  else
    legend('off');    
  end
  title(tstr,'fontweight','bold','fontsize',16);

  axes(axs(2,iVw));
  gscatter([lblNCents(:,1,iVw);gtLblNCents(:,1,iVw)],...
           [lblNCents(:,2,iVw);gtLblNCents(:,2,iVw)],gall);
  axis(gca,[-1 1 -1 1]);
  grid(gca,'on');
  axis(gca,'ij');
  legend('off');
  title(sprintf('Normalized lbls. view%d',iVw),'fontweight','bold','fontsize',16);
end

% set(axs(3:4),'XTickLabel',[],'YTickLabel',[]);
hFig.Color = [1 1 1];
hFig.PaperOrientation = 'landscape';
hFig.PaperType = 'arch-c';

%% Extreme examples
y = lblNCents(:,2,2);
[~,i] = sort(y);
xtremeRows = i([1 end]);
hFig = figure;
axs = createsubplots(1,2);
for i=1:2
  ax = axs(i);
  axes(ax);
  imagesc(IFinalReconciled{xtremeRows(i),2});
  colormap gray
  axis image
  set(ax,'XTick',[],'YTick',[]);
end
  
hFig.Color = [1 1 1];


%% Scales
% Plot, grouped by cat and framesz.

% xyLbl = reshape(pLbl,[n 5 2 2]); % n, pt, x/y, ivw
% xyLblN = reshape(pLblN,[n 5 2 2]);
% xygtLbl = reshape(pLblGT,[nGT 5 2 2]);
% xygtLblN = reshape(pLblGTN,[nGT 5 2 2]);
% lblCents = squeeze(mean(xyLbl,2)); % [nx2x2]. i, x/y, vw
% lblNCents = squeeze(mean(xyLblN,2));
% gtLblCents = squeeze(mean(xygtLbl,2));
% gtLblNCents = squeeze(mean(xygtLblN,2));

% Vw1, measure up-to-down length
anttipXY = squeeze(mean(xyLbl(:,[3 4],:,1),2)); % n,x/y. av loc of pts 3,4
probXY = squeeze(xyLbl(:,5,:,1)); % n,x/y
vw1Scale = sqrt(sum((anttipXY-probXY).^2,2));
anttipgtXY = squeeze(mean(xygtLbl(:,[3 4],:,1),2)); % n,x/y. av loc of pts 3,4
probgtXY = squeeze(xygtLbl(:,5,:,1)); % n,x/y
vw1ScaleGT = sqrt(sum((anttipgtXY-probgtXY).^2,2));

% Vw2, measure mean centroid-to-lm length.
centXY = reshape(lblCents(:,:,2),[n 1 2]);
vw2Scale = mean(sqrt(sum((xyLbl(:,:,:,2)-centXY).^2,3)),2);
szassert(vw2Scale,[n 1]);
centXYgt = reshape(gtLblCents(:,:,2),[nGT 1 2]);
vw2ScaleGT = mean(sqrt(sum((xygtLbl(:,:,:,2)-centXYgt).^2,3)),2);
szassert(vw2ScaleGT,[nGT 1]);

hFig = figure('position',[2561 401 1920 1124]); 
axs = createsubplots(1,2);

bbAllVw1 = makeBBoxes([IFinalReconciled(:,1);Igt(:,1)]);
bbAllVw2 = makeBBoxes([IFinalReconciled(:,2);Igt(:,2)]);
assert(isequal(bbAllVw1,bbAllVw2));
tfBigIm = ismember(bbAllVw1,[1 1 1024 1024],'rows');
bigImStr = repmat({''},size(tfBigIm));
bigImStr(tfBigIm) = {'BIG'};

gall = [tFinalReconciled.lblCat;strcat('gt',tGT.lblCat)];
gall = strcat(bigImStr,gall);

ax = axs(1);
axes(ax);
boxplot([vw1Scale;vw1ScaleGT],gall);
title('view1','fontweight','bold');
ax = axs(2);
axes(ax);
boxplot([vw2Scale;vw2ScaleGT],gall);
title('view2','fontweight','bold');

%% Crop/recenter data.

for iVw=1:2
  hFig = figure('position',[2561 401 1920 1124]);
  axs = createsubplots(2,2,.1);
  axs = reshape(axs,2,2);
  COORDS = {'x' 'y'};

  for iCoord=1:2
    axes(axs(1,iCoord));
    x = lblCents(:,iCoord,iVw);
    histogram(x,50);
    hold on;
    yl = ylim;
    switch iVw 
      case 1
        if iCoord==1
          plot([115 115],yl,'r-');
        else
          plot([140 140],yl,'r-');
          plot(512-[140 140],yl,'r-');
        end
    end
    tstr = sprintf('vw %d. %s-dir. n=%d. mdn,mad=%.2f,%.2f',iVw,...
      COORDS{iCoord},numel(x),nanmedian(x),mad(x,1));
    title(tstr,'fontweight','bold');
    
    axes(axs(2,iCoord));
    x = gtLblCents(:,iCoord,iVw);
    histogram(x,50);
    hold on;
    yl = ylim;
    switch iVw
      case 1
        if iCoord==1
          plot([115 115],yl,'r-');
        else
          plot([140 140],yl,'r-');
          plot(512-[140 140],yl,'r-');
        end
    end
    tstr = sprintf('GT. vw%d %s-dir. n=%d. mdn,mad=%.2f,%.2f',iVw,...
      COORDS{iCoord},numel(x),nanmedian(x),mad(x,1));
    title(tstr,'fontweight','bold');
  end

  switch iVw
    case 1
      linkaxes(axs(:,1),'x');
      xlim(axs(1,1),[0 310]);
      linkaxes(axs(:,2),'x');
      xlim(axs(1,2),[0 600]);
  end
end

%% View1: Crop to 230x280.
% X. There is a small blip of ims where the fly is crammed to the left, 
%   x<60. Ccrop maximally to left, nothing can be done. Both GT and reg.
% X. The remaining xcentroids have a median of 115. Ims to left of 115, 
%   crop maximally to the left. Ims to right of 115, draw x-values from the
%   ims to left of 115 and set it to that. This way the long tail to the
%   right gets compactified.

XROISZ = 230; % 1:115 are the left half.
XFARLEFTTHRESH = 75;

xcv1 = lblCents(:,1,1); % shape centroid xy's, view 1
xRoiV1Lo = shcropx(xcv1,XROISZ,XFARLEFTTHRESH);

xcv1gt = gtLblCents(:,1,1); % shape centroid xy's, view 1
xRoiV1LoGT = shcropx(xcv1gt,XROISZ,XFARLEFTTHRESH);

%% view1, y. Want to crop total size 280.
% Y1. There are ims too close to both the top AND bottom. For those within
% 140 of the top or bottom, crop maximally to the top/bottom. For those in
% the middle, sample a random jitter with SD~ the X case, but with tails
% symmetrically truncated so the jitter doesn't go into the "endzones".
% This tries to keep the resulting distribution symmetric, except for
% whatever centroids were cropped maximally in each endzone and left in
% place.

SDUSE = 19.5; % from view1, x-dir
YROISZ = 280;

ycv1 = lblCents(:,2,1);
bb = makeBBoxes(IFinalReconciled(:,1));
yRoiV1Lo = shcropy(ycv1,bb(:,4),SDUSE,YROISZ);

ycv1gt = gtLblCents(:,2,1);
bbgt = makeBBoxes(Igt(:,1));
yRoiV1LoGT = shcropy(ycv1gt,bbgt(:,4),SDUSE,YROISZ);

roiV1 = [xRoiV1Lo xRoiV1Lo+XROISZ-1 yRoiV1Lo yRoiV1Lo+YROISZ-1];
roiV1gt = [xRoiV1LoGT xRoiV1LoGT+XROISZ-1 yRoiV1LoGT yRoiV1LoGT+YROISZ-1];

%%
clear IFT_crop;
clear xyLbl_FR_crop;
[IFR_crop(:,1),xyLbl_FR_crop(:,:,:,1)] = ...
                  croproi(IFinalReconciled(:,1),xyLbl(:,:,:,1),roiV1);
[Igt_crop(:,1),xyLbl_GT_crop(:,:,:,1)] = ...
                  croproi(Igt(:,1),xygtLbl(:,:,:,1),roiV1gt);
                
%% View2, X. 
% We have a lot of leeway here and just use shcropy.

SDUSE = 19.5; % from view1, x-dir
XROISZ = 256;

xcv2 = lblCents(:,1,2);
bbv2 = makeBBoxes(IFinalReconciled(:,2));
xRoiV2Lo = shcropy(xcv2,bbv2(:,3),SDUSE,XROISZ);

xcv2gt = gtLblCents(:,1,2);
bbv2gt = makeBBoxes(Igt(:,2));
xRoiV2LoGT = shcropy(xcv2gt,bbv2gt(:,3),SDUSE,XROISZ);

%% View2, Y. 

SDUSE = 19.5; % from view1, x-dir
YROISZ = 256;

ycv2 = lblCents(:,2,2);
bbv2 = makeBBoxes(IFinalReconciled(:,2));
yRoiV2Lo = shcropy(ycv2,bbv2(:,4),SDUSE,YROISZ);

ycv2gt = gtLblCents(:,2,2);
bbv2gt = makeBBoxes(Igt(:,2));
yRoiV2LoGT = shcropy(ycv2gt,bbv2gt(:,4),SDUSE,YROISZ);

%%
roiV2 = [xRoiV2Lo xRoiV2Lo+XROISZ-1 yRoiV2Lo yRoiV2Lo+YROISZ-1];
roiV2gt = [xRoiV2LoGT xRoiV2LoGT+XROISZ-1 yRoiV2LoGT yRoiV2LoGT+YROISZ-1];

[IFR_crop(:,2),xyLbl_FR_crop(:,:,:,2)] = ...
                  croproi(IFinalReconciled(:,2),xyLbl(:,:,:,2),roiV2);
[Igt_crop(:,2),xyLbl_GT_crop(:,:,:,2)] = ...
                  croproi(Igt(:,2),xygtLbl(:,:,:,2),roiV2gt);
                
%%
save trnDataSH_Apr18.mat -append IFR_crop Igt_crop xyLbl_FR_crop xyLbl_GT_crop

%% Centroid overlay, POST CROP
mdl = struct('D',10,'d',2,'nfids',5);

xyLbl_c = xyLbl_FR_crop;
xyLblN_c = nan(n,5,2,2);
xygtLbl_c = xyLbl_GT_crop;
xygtLblN_c = nan(nGT,5,2,2);
bbAll = nan(n+nGT,4,2);
for ivw=1:2    
  bbvw = makeBBoxes(IFR_crop(:,ivw));
  bbvwGT = makeBBoxes(Igt_crop(:,ivw));
  bbAll(:,:,ivw) = [bbvw;bbvwGT];

  pLbl_c = reshape(xyLbl_c(:,:,:,ivw),[n 10]);
  pLblgt_c = reshape(xygtLbl_c(:,:,:,ivw),[nGT 10]);
  pLblN_c = shapeGt('projectPose',mdl,pLbl_c,bbvw);
  pLblgtN_c = shapeGt('projectPose',mdl,pLblgt_c,bbvwGT); 
  xyLblN_c(:,:,:,ivw) = reshape(pLblN_c,n,5,2);
  xygtLblN_c(:,:,:,ivw) = reshape(pLblgtN_c,nGT,5,2);
end

lblCents_c = squeeze(mean(xyLbl_c,2)); % [nx2x2]. i, x/y, vw
lblNCents_c = squeeze(mean(xyLblN_c,2));
gtLblCents_c = squeeze(nanmean(xygtLbl_c,2));
gtLblNCents_c = squeeze(nanmean(xygtLblN_c,2));

gall = [tFinalReconciled.lblCat;strcat('gt',tGT.lblCat)];

hFig = figure('position',[2561 401 1920 1124]);
axs = createsubplots(2,2,.1);
axs = reshape(axs,2,2);
lims = [0 230 0 280;0 256 0 256];
for iVw=1:2
  ax = axs(1,iVw);
  axes(ax);
  gscatter([lblCents_c(:,1,iVw);gtLblCents_c(:,1,iVw)],...
           [lblCents_c(:,2,iVw);gtLblCents_c(:,2,iVw)],gall);
  hold(ax,'on');
%   plot(ax,[0 1024],[512 512],'k-');
%   plot(ax,[768 768],[0 1024],'k-');
  axis(gca,lims(iVw,:));
  grid(gca,'on');
  axis(gca,'ij');
  title(sprintf('vw%d',iVw),'fontweight','bold','fontsize',16);

  axes(axs(2,iVw));
  gscatter([lblNCents_c(:,1,iVw);gtLblNCents_c(:,1,iVw)],...
           [lblNCents_c(:,2,iVw);gtLblNCents_c(:,2,iVw)],gall);
  axis(gca,[-1 1 -1 1]);
  grid(gca,'on');
  axis(gca,'ij');
  title(sprintf('Normalized lbls. vw%d',iVw),'fontweight','bold','fontsize',16);
end

hFig.Color = [1 1 1];
hFig.PaperOrientation = 'landscape';
hFig.PaperType = 'arch-c';


%% Centroid hists, POST CROP

for iVw=1:2
  hFig = figure('position',[2561 401 1920 1124]);
  axs = createsubplots(2,2,.1);
  axs = reshape(axs,2,2);
  COORDS = {'x' 'y'};

  for iCoord=1:2
    axes(axs(1,iCoord));
    x = lblCents_c(:,iCoord,iVw);
    histogram(x,50);
    hold on;
    switch iVw 
      case 1
        if iCoord==1
          xlim([0 230]);
        else
          xlim([0 280]);
        end
      case 2
        xlim([0 256]);
    end
    tstr = sprintf('vw %d. %s-dir. n=%d. mdn,mad=%.2f,%.2f',iVw,...
      COORDS{iCoord},numel(x),nanmedian(x),mad(x,1));
    title(tstr,'fontweight','bold');
    
    axes(axs(2,iCoord));
    x = gtLblCents_c(:,iCoord,iVw);
    histogram(x,50);
    hold on;
    yl = ylim;
    switch iVw
      case 1
        if iCoord==1
          xlim([0 XROISZ]);
        else
          xlim([0 YROISZ]);
        end
      case 2
        xlim([0 256]);
    end
    tstr = sprintf('GT. vw%d %s-dir. n=%d. mdn,mad=%.2f,%.2f',iVw,...
      COORDS{iCoord},numel(x),nanmedian(x),mad(x,1));
    title(tstr,'fontweight','bold');
  end

%   switch iVw
%     case 1
%       linkaxes(axs(:,1),'x');
%       xlim(axs(1,1),[0 310]);
%       linkaxes(axs(:,2),'x');
%       xlim(axs(1,2),[0 600]);
%   end
end

%% Label overlay, POST_CROP

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;
LIMS = [0 230 0 280;0 256 0 256];

hFig = figure('position',[2561 401 1920 1124]); 
axs = createsubplots(2,2);
axs = reshape(axs,2,2);

for iVw=1:2
  ax = axs(1,iVw);
  axes(ax);
  hold(ax,'on');
  ax.Color = [0 0 0];
  for ipt=1:5
    plot(xyLbl_c(:,ipt,1,iVw),xyLbl_c(:,ipt,2,iVw),'.','markerSize',10,...
      'color',COLORS(ipt,:));
  end
  axis(ax,LIMS(iVw,:),'ij');
  title(sprintf('vw%d',iVw),'fontweight','bold');
  
  ax = axs(2,iVw);
  axes(ax);
  hold(ax,'on');
  ax.Color = [0 0 0];
  for ipt=1:5
    plot(xygtLbl_c(:,ipt,1,iVw),xygtLbl_c(:,ipt,2,iVw),'.','markerSize',10,...
      'color',COLORS(ipt,:));
  end
  axis(ax,LIMS(iVw,:),'ij');
  title(sprintf('vw%d, GT',iVw),'fontweight','bold');
  
  linkaxes(axs);
end

%% POSTCROP Montage: labels vs images
tfTrn1 = trnSets(:,1);
nTrn1 = nnz(tfTrn1)
idstrs = strcat(tFinalReconciled.lblCat(tfTrn1),'|',numarr2trimcellstr(find(tfTrn1)));
hFigVw1 = figure('position',[1 41 2560 1484]);
hFigVw2 = figure('position',[2561 401 1920 1124]);

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;

easymontage(IFR_crop(tfTrn1,1),...
  reshape(xyLbl_FR_crop(tfTrn1,:,:,1),nTrn1,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs,'hFig',hFigVw1,...
  'axisimage',true);
easymontage(IFR_crop(tfTrn1,2),...
  reshape(xyLbl_FR_crop(tfTrn1,:,:,2),nTrn1,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs,'hFig',hFigVw2,...
  'axisimage',true);

%% POSECROP Montage GT: labels vs images
idstrs = strcat(tGT.lblCat,'|',numarr2trimcellstr((1:height(tGT))'));
hFigVw1 = figure('position',[1 41 2560 1484]);
hFigVw2 = figure('position',[2561 401 1920 1124]);

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;

ROWS = 1:100;
nrows = numel(ROWS);

easymontage(Igt_crop(ROWS,1),...
  reshape(xyLbl_GT_crop(ROWS,:,:,1),nrows,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs(ROWS),'hFig',hFigVw1,...
  'axisimage',true);
easymontage(Igt_crop(ROWS,2),...
  reshape(xyLbl_GT_crop(ROWS,:,:,2),nrows,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs(ROWS),'hFig',hFigVw2,...
  'axisimage',true);

