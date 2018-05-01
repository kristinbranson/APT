%% 
load trnDataSH_Apr18.mat
%%
ci = load('cropInfo20180426.mat');

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

%% Centroids relatie to clicks
% note: ~1000 rows that are 1024x1024
xyLbl = reshape(pLbl,[n 5 2 2]); % n, pt, x/y, ivw
xyLblN = reshape(pLblN,[n 5 2 2]);
xygtLbl = reshape(pLblGT,[nGT 5 2 2]);
lblCents = squeeze(mean(xyLbl,2)); % [nx2x2]. i, x/y, vw
lblNCents = squeeze(mean(xyLblN,2)); 
gtLblCents = squeeze(nanmean(xygtLbl,2));

tCrop = ci.t;
tFR = tFinalReconciled(:,{'flyID'});
tFR.Properties.VariableNames = {'fly'};
tFR.mov1 = tFinalReconciled.movFile_read(:,1);
tFR.mov2 = tFinalReconciled.movFile_read(:,2);
[tf,loc] = ismember(tFR,tCrop);
assert(all(tf));
cptsvec = reshape(ci.cpts,[height(tCrop) 4]);
tFinalReconciled.cropClickPts = cptsvec(loc,:);

xyCCP = reshape(tFinalReconciled.cropClickPts,n,2,2); % row,{x,y},iVw

delCCPcents = lblCents-xyCCP; % row, {x,y}, iVw
distCCPcents = squeeze(sqrt(sum(delCCPcents.^2,2))); % nx2 (vw)
gall = [tFinalReconciled.lblCat]; % strcat('gt',tGT.lblCat)];
%gall = strcat(bigImStr,gall);

%% Viz clicks-to-centroids dists
hFig = figure(11);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];

axs = createsubplots(2,2,.1);
axs = reshape(axs,2,2);
for ivw=1:2
  ax = axs(1,ivw);
  axes(ax);
  gscatter(delCCPcents(:,1,ivw),delCCPcents(:,2,ivw),gall);
  grid(ax,'on');
  axis(ax,'equal');
  tstr = sprintf('vw%d: vec CCP to cent',ivw);
  title(tstr,'fontweight','bold');  
  
  ax = axs(2,ivw);
  axes(ax);
  hist(distCCPcents(:,ivw),50);
  tstr = sprintf('vw%d: dist CCP to cent',ivw);
  title(tstr,'fontweight','bold');
end

%% One-D centroids relative to clicks
hFig = figure(12);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];

XY = {'x' 'y'};
axs = createsubplots(2,2,.1);
axs = reshape(axs,2,2);
for ivw=1:2
  for icoord=1:2
    ax = axs(icoord,ivw);
    axes(ax);
    
    z = delCCPcents(:,icoord,ivw);
    zsd = std(z);
    histogram(z,50);
    grid(ax,'on');
    tstr = sprintf('vw%d delta-%s. SD=%.3f',ivw,XY{icoord},zsd);
    title(ax,tstr,'fontweight','bold');
  end
end

%% Orignal centroid dist
gall = tFinalReconciled.lblCat; % strcat('gt',tGT.lblCat)];

hFig = figure(13);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];

axs = createsubplots(2,3,.1);
axs = reshape(axs,2,3);
for ivw=1:2
  ax = axs(ivw,1);
  axes(ax);
  gscatter(lblCents(:,1,ivw),lblCents(:,2,ivw),gall);
  grid(ax,'on');
  axis(ax,'ij','auto');
  axis(ax,[0 1024 0 1024]);
  tstr = sprintf('vw%d.',ivw);
  title(ax,tstr,'fontweight','bold');
  
  for icoord=1:2
    ax = axs(ivw,icoord+1);
    axes(ax);
    
    z = lblCents(:,icoord,ivw);
    zsd = std(z);
    histogram(z,50);
    grid(ax,'on');
    tstr = sprintf('vw%d %s. SD=%.3f',ivw,XY{icoord},zsd);
    title(ax,tstr,'fontweight','bold');
  end
end

%% Orignal centroid dist NORMALIZED
gall = tFinalReconciled.lblCat; % strcat('gt',tGT.lblCat)];

hFig = figure(14);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];

axs = createsubplots(2,3,.1);
axs = reshape(axs,2,3);
for ivw=1:2
  ax = axs(ivw,1);
  axes(ax);
  gscatter(lblNCents(:,1,ivw),lblNCents(:,2,ivw),gall);
  grid(ax,'on');
  axis(ax,'ij','auto');
  axis(ax,[-1 1 -1 1]);
  tstr = sprintf('vw%d.',ivw);
  title(ax,tstr,'fontweight','bold');
  
  for icoord=1:2
    ax = axs(ivw,icoord+1);
    axes(ax);
    
    z = lblNCents(:,icoord,ivw);
    zsd = std(z);
    histogram(z,50);
    grid(ax,'on');
    tstr = sprintf('vw%d %s. SD=%.3f',ivw,XY{icoord},zsd);
    title(ax,tstr,'fontweight','bold');
  end
end

%% Viz one example
IFINALROW = 4019;

hFig = figure(15);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];

xyCents = squeeze(lblCents(IFINALROW,:,:));
xycc = reshape(tFinalReconciled.cropClickPts(IFINALROW,:),2,2);

axs = createsubplots(1,2,.1);
%axs = reshape(axs,2,2);
for ivw=1:2
  ax = axs(1,ivw);
  axes(ax);
  
  imagesc(IFinalReconciled{IFINALROW,ivw});
  colormap gray;
  hold on
  axis(ax,'image');
  
  plot(xyCents(1,ivw),xyCents(2,ivw),'gs','markerfacecolor',[0 1 0],'markersize',15);
  plot(xyLbl(IFINALROW,:,1,ivw),xyLbl(IFINALROW,:,2,ivw),'g.','markersize',10);
  plot(xycc(1,ivw),xycc(2,ivw),'r.','markersize',20);
  
  grid(ax,'on');
end

%% Crop
IToCrop = IFinalReconciled;
xyLblToCrop = xyLbl;
tToCrop = tFinalReconciled;
n = height(tToCrop);
szassert(IToCrop,[n 2]);
szassert(xyLblToCrop,[n 5 2 2]);

JITTER_RC = [28 8.5; 29.5 20]; % ivw, {nr,nc}.  based on SDs of 1D distros of delCCPcents
%JITTER_RC = [0 0;0 0]; 
ROI_NRNC = [350 230; 350 350]; % ivw, {nr,nc}
xyCCP = reshape(tToCrop.cropClickPts,n,2,2); % row,{x,y},iVw

IFR_crop3 = cell(size(IToCrop));
xyLbl_FR_crop3 = nan(size(xyLblToCrop));
roi_crop3 = nan(n,4,2); % irow,{xlo,xhi,ylo,yhi},ivw
crop3_xyjitterappld = nan(n,2,2); % irow,{x,y},ivw
for ivw=1:2
  roinr = ROI_NRNC(ivw,1);
  roinc = ROI_NRNC(ivw,2);
  rowjitter = JITTER_RC(ivw,1);
  coljitter = JITTER_RC(ivw,2);
  for i=1:n
    [imnr,imnc] = size(IToCrop{i,ivw});    
    roiCtrCol = xyCCP(i,1,ivw);
    roiCtrRow = xyCCP(i,2,ivw);
    [roi_crop3(i,:,ivw),crop3_xyjitterappld(i,1,ivw),crop3_xyjitterappld(i,2,ivw)] = ...
      cropsmart(imnc,imnr,roinc,roinr,roiCtrCol,roiCtrRow,...
      'rowjitter',rowjitter,'coljitter',coljitter);
  end
  
  [IFR_crop3(:,ivw),xyLbl_FR_crop3(:,:,:,ivw)] = ...
    croproi(IToCrop(:,ivw),xyLblToCrop(:,:,:,ivw),roi_crop3(:,:,ivw));  
end

%%
save trnDataSH_Apr18 -append IFR_crop3 xyLbl_FR_crop3 roi_crop3 crop3_xyjitterappld xyCCP

%% Crop distros

% ROI_NRNC = [320 230; 320 320]; % ivw, {nr,nc}

szassert(xyLbl_FR_crop2,[n 5 2 2]);
lblCropCents = squeeze(mean(xyLbl_FR_crop2,2)); % [nx2x2]. i, x/y, vw

gall = [tFinalReconciled.lblCat]; % strcat('gt',tGT.lblCat)];

hFig = figure(21);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];

axs = createsubplots(2,3,.1);
axs = reshape(axs,2,3);
for ivw=1:2
  ax = axs(ivw,1);
  axes(ax);
  gscatter(lblCropCents(:,1,ivw),lblCropCents(:,2,ivw),gall);
  grid(ax,'on');
  lims = ROI_NRNC(ivw,:);
  axis(ax,[0 lims(2) 0 lims(1)]);
  axis(ax,'ij');
  tstr = sprintf('vw%d cropped centroids',ivw);
  title(tstr,'fontweight','bold');
  
  for icoord=1:2
    ax = axs(ivw,icoord+1);
    axes(ax);
    
    z = lblCropCents(:,icoord,ivw);
    zsd = std(z);
    histogram(z,50);
    grid(ax,'on');
    tstr = sprintf('vw%d %s. SD=%.3f',ivw,XY{icoord},zsd);
    title(ax,tstr,'fontweight','bold');
  end
end


%% Browse crops

roi2viz = roi_crop3;
I2viz = IFR_crop3;
xyLblCrop2viz = xyLbl_FR_crop3;

hFig = figure(31);
hFig.Position = [2561 401 1920 1124];
hFig.Color = [1 1 1];
axs = createsubplots(2,2,.1);
axs = reshape(axs,2,2);

iRows = randperm(n);
for i=iRows(:)'  
  for ivw=1:2
    ax = axs(1,ivw);
    cla(ax);
    axes(ax);

    imagesc(IFinalReconciled{i,ivw});
    colormap gray;
    hold on
    axis(ax,'image');
    title(ax,sprintf('row %d, vw%d',i,ivw),'fontweight','bold');

    xyCents = squeeze(lblCents(i,:,:));
    xycc = reshape(tFinalReconciled.cropClickPts(i,:),2,2);
    plot(xyCents(1,ivw),xyCents(2,ivw),'gs','markerfacecolor',[0 1 0],'markersize',15);
    plot(xyLbl(i,:,1,ivw),xyLbl(i,:,2,ivw),'g.','markersize',10);
    plot(xycc(1,ivw),xycc(2,ivw),'r.','markersize',20);
    roi = roi2viz(i,:,ivw);
    plot(roi(1:2),roi([3 3]),'r-','linewidth',3);
    plot(roi(1:2),roi([4 4]),'r-','linewidth',3);
    plot(roi([1 1]),roi(3:4),'r-','linewidth',3);
    plot(roi([2 2]),roi(3:4),'r-','linewidth',3);
    grid(ax,'on');    
    
    ax = axs(2,ivw);
    cla(ax);
    axes(ax);

    imagesc(I2viz{i,ivw});
    colormap gray;
    hold on
    axis(ax,'image');

    plot(xyLblCrop2viz(i,:,1,ivw),xyLblCrop2viz(i,:,2,ivw),'g.','markersize',10);
    grid(ax,'on');  
  end
  
  input('hk');
end
 