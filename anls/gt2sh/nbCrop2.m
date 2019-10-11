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
  'doroi',false,'axisimage',true);
easymontage(I2viz(tfTrn1,2),reshape(xy2viz(tfTrn1,:,:,2),nTrn,10),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs,'hFig',hFigVw2,...
  'doroi',false,'axisimage',true);

%% Label overlay, POST_CROP

I2viz = IFR_crop3;
xy2viz = xyLbl_FR_crop3;
I2viz2 = IFR_crop2;
xy2viz2 = xyLbl_FR_crop2;

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;
LIMS = [0 230 0 280;0 256 0 256];
LIMS2 = [0 230 0 350;0 350 0 350];

hFig = figure(35);
hFig = figure('position',[2561 401 1920 1124]); 
axs = createsubplots(2,2);
axs = reshape(axs,2,2);

for iVw=1:2
  ax = axs(1,iVw);
  axes(ax);
  hold(ax,'on');
  ax.Color = [0 0 0];
  for ipt=1:5
    plot(xy2viz(:,ipt,1,iVw),xy2viz(:,ipt,2,iVw),'.','markerSize',10,...
      'color',COLORS(ipt,:));
  end
  axis('ij');
  axis(LIMS2(iVw,:));
  title(sprintf('vw%d',iVw),'fontweight','bold');
  
  ax = axs(2,iVw);
  axes(ax);
  hold(ax,'on');
  ax.Color = [0 0 0];
  for ipt=1:5
    plot(xy2viz2(:,ipt,1,iVw),xy2viz2(:,ipt,2,iVw),'.','markerSize',10,...
      'color',COLORS(ipt,:));
  end
  axis(ax,LIMS2(iVw,:),'ij');
  title(sprintf('vw%d, 2',iVw),'fontweight','bold');
  
%   linkaxes(axs);
end


%% xv res
ncrop = 3;
nvw = 2;
xvmats = cell(ncrop,nvw);
for icrop=1:ncrop
for ivw=1:2
  if icrop==1
    cropstr = '';
  else
    cropstr = num2str(icrop);
  end
  matname = sprintf('crop2exp_crop%d__xv__IFR_crop%s__vw%d__xvFRsplit3.mat',...
    icrop,cropstr,ivw);
  xvmats{icrop,ivw} = load(matname,'-mat');
end
end
xvmats0 = cell(1,2);
xvmats0{1} = load('crop2exp__xv__IFinalReconciled__vw1__xvFRsplit3.mat');
xvmats0{2} = load('crop2exp__xv__IFinalReconciled__vw2__xvFRsplit3.mat');
xvmats = [xvmats0;xvmats];
xvmats = cell2mat(xvmats);

n = 4961;
xverr = nan(n,5,2,4);
for icrop=1:4
  for ivw=1:2
    xverr(:,:,ivw,icrop) = cat(1,xvmats(icrop,ivw).errs{:});
  end
end

%% KB percentiles

DOSAVE = true;
PLOTFULL = true;
SAVEDIR = fullfile(pwd,'figsMe');
PTILES = [50 75 90 95 97.5 99 99.5];
XVNTRN = 3307;
npts = 5;

ncrops = 4;
nlandmarks = npts;
nviews = 2;
nptiles = numel(PTILES);
normerr_prctiles = nan(nptiles,nlandmarks,nviews,ncrops);
for l = 1:nlandmarks
  for v = 1:nviews
    for k = 1:ncrops
      normerr_prctiles(:,l,v,k) = prctile(xverr(:,l,v,k),PTILES);
    end
  end
end

hfig = 11;
figure(hfig);
clf
set(hfig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);

colors = jet(nptiles);
hax = createsubplots(nviews,ncrops,[.01 .01;.05 .01]);
hax = reshape(hax,[nviews,ncrops]);

h = nan(1,nptiles);
for viewi = 1:nviews
  if PLOTFULL
    im = Igt{1,viewi};
    xyLbl = pLbl2xyvSH(tGT.pLbl);
    xyLbl = squeeze(xyLbl(1,:,:,viewi)); % nptx2
  else
    im = Igt_crop{1,viewi};
    xyLbl = squeeze(xyLbl_GT_crop(1,:,:,viewi)); % [5x2]
  end
  
  for k = 1:ncrops
    ax = hax(viewi,k);
    imagesc(im,'Parent',ax);
    colormap gray
    axis(ax,'image','off');
    hold(ax,'on');
    plot(ax,xyLbl(:,1),xyLbl(:,2),'m+');
    if viewi==1
      switch k
        case 1
          tstr = sprintf('No crop (NTrn=%d)',XVNTRN);
        case 2
          tstr = sprintf('Cheat crop');
        otherwise
          tstr = sprintf('Fair crop%d',k-1);
      end
      title(ax,tstr,'fontweight','bold','fontsize',22);
    end

    for p = 1:nptiles
      for l = 1:nlandmarks
        rad = normerr_prctiles(p,l,viewi,k);
        h(p) = drawellipse(xyLbl(l,1),xyLbl(l,2),0,rad,rad,...
          'Color',colors(p,:),'Parent',ax,'linewidth',1);
      end
    end
  end
end

set(hfig,'Position',[2561 401 1920 1124]);

legends = cell(1,nptiles);
for p = 1:nptiles
  legends{p} = sprintf('%sth %%ile',num2str(PTILES(p)));
end
hl = legend(h,legends);
set(hl,'Color','k','TextColor','w','EdgeColor','w');
truesize(hfig);

if DOSAVE
  FNAME = 'Crop2exp_Ptiles_bullseye';
  hgsave(hfig,fullfile(SAVEDIR,[FNAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[FNAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[FNAME '.png']));    
end

%% KB: per-landmark frac leq curves

DOSAVE = 1;
minerr = inf;

predcolors = lines(numel(ncrops));

% trkerr: [nx5x2xnntrns]
fracleqerr = cell(nlandmarks,nviews,ncrops);
for l = 1:nlandmarks
  for v = 1:nviews
    for p = 1:ncrops
      sortederr = sort(xverr(:,l,v,p));
      [sortederr,nleqerr] = unique(sortederr);
      fracleqerr{l,v,p} = cat(2,nleqerr./size(xverr,1),sortederr);
      %minerr = min(minerr,fracleqerr{l,v,p}(find(fracleqerr{l,v,p}(:,1)>=minfracplot,1),2));
    end
  end
end

hfig = 14;
figure(hfig);
set(hfig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);
clf;

hax = createsubplots(nviews,nlandmarks,[.05 0;.1 .1]);
hax = reshape(hax,[nviews,nlandmarks]);

% minmaxerr = inf;
% for p = 1:npredfns,
%   minmaxerr = min(minmaxerr,prctile(vectorize(normerr(:,:,p,:)),99.9));
% end

clrs = [0 0 0;1 0 0;0 0 1;0 1 0];
clear h;
for l = 1:nlandmarks
  for v = 1:nviews
    ax = hax(v,l);
    hold(ax,'on');
    grid(ax,'on');

    tfPlot1 = v==1 && l==1;    

    for p=1:ncrops
      if p==1
        lw = 1;
      else
        lw = 1.5;
      end
      h(p) = plot(ax,fracleqerr{l,v,p}(:,2),fracleqerr{l,v,p}(:,1),'-',...
        'linewidth',lw,'color',clrs(p,:));
      tstr = sprintf('vw%d pt%d',v,l);
      if tfPlot1
        tstr = ['ErrCDF vs CropType: ' tstr];       
      end
      title(ax,tstr,'fontweight','bold','fontsize',16);
    end
%     if l == 1 && v == 1,
%       legend(h,prednames,'Location','southeast');
%     end
%     title(hax(v,l),sprintf('%s, %s',lbld.cfg.LabelPointNames{l},lbld.cfg.ViewNames{v}));

    set(ax,'XTick',[1 2 4 8 16 32],'XScale','log');
    if tfPlot1
      title(ax,tstr,'fontweight','bold');
      xlabel(ax,'Error (raw,  px)','fontsize',14);
      ylabel(ax,'Frac. smaller','fontsize',14);
      
      legstr = {'no crop' 'cheat' 'fair2' 'fair3'};
      legend(h,legstr,'location','southeast');
    else
      set(ax,'XTickLabel',[],'YTickLabel',[]);
    end
  end
end

linkaxes(hax(:),'x');
xlim(hax(1),[1 32]);
% set(hax,'XLim',[minerr,minmaxerr],'YLim',[minfracplot,1],'XScale','log');%,'YScale','log');%

% if nlandmarks > 1,
%   xticks = [.01,.025,.05,.10:.10:minmaxerr];
%   xticks(xticks<minerr | xticks > minmaxerr) = [];
%   set(hax,'XTick',xticks);
% end
% yticks = [.01:.01:.05,.1:.1:1];
% yticks(yticks<minfracplot) = [];
% set(hax,'YTick',yticks);
set(hfig,'Units','pixels','Position',[2561 401 1920 1124]);

if DOSAVE
  FNAME = 'Crop2exp_FracLT';
  hgsave(hfig,fullfile('figsMe',[FNAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-c');
  print(hfig,'-dpdf',fullfile('figsMe',[FNAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile('figsMe',[FNAME '.png']));  
end

%% Pctiles

DOSAVE = true;
SAVEDIR = fullfile(pwd,'figsMe');
PTILES = [50 75 90 95 97.5 99 99.5];
XVNTRN = 3307;

hFig = figure(18);
clf
set(hFig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);

axs = createsubplots(nvw,npts+1,[.05 0;.12 .12]);
axs = reshape(axs,nvw,npts+1);
vws = 1:2;
pts = 1:5;
for ivw=vws
  for ipt=[pts inf]
    if ~isinf(ipt)
      % normal branch
      errs = squeeze(xverr(:,ipt,ivw,:)); % nxncrops
      y = prctile(errs,PTILES); % [nptlsxncrops]
      ax = axs(ivw,ipt);
      tstr = sprintf('vw%d pt%d',ivw,ipt);   
    else      
      errs = squeeze(sum(xverr(:,:,ivw,:),2)/npts); % [nxncrops]
      y = prctile(errs,PTILES); % [nptlsxncrops]      
      ax = axs(ivw,npts+1);
      tstr = sprintf('vw%d, mean allpts',ivw);
    end
    axes(ax);
    tfPlot1 = ivw==1 && ipt==1;
    if tfPlot1
      tstr = ['XV err vs CropType: ' tstr];
    end    
    
    args = {'YGrid' 'on' 'XGrid' 'on' 'XLim' [0 5] 'XTick' 1:ncrops ...
      'XTicklabelRotation',45,'XTickLabel',{'none' 'cheat' 'fair2' 'fair3'} ...
      'FontSize' 16};  
    x = 1:4; % croptypes
    h = plot(x,y','.-','markersize',20);
    set(ax,args{:});
    hold(ax,'on');
    ax.ColorOrderIndex = 1;
    
    title(tstr,'fontweight','bold','fontsize',16);
    if tfPlot1
      legstrs = [...
        strcat(numarr2trimcellstr(PTILES'),'%');];
      hLeg = legend(h,legstrs);
      hLeg.FontSize = 10;
      %xlabel('Crop type','fontweight','normal','fontsize',14);

      ystr = sprintf('raw err (px)');
      ylabel(ystr,'fontweight','normal','fontsize',14);
    else
      set(ax,'XTickLabel',[]);
    end
    if ipt==1
    else
      set(ax,'YTickLabel',[]);
    end
  end
end
linkaxes(axs(1,:),'y');
linkaxes(axs(2,:),'y');
ylim(axs(1,1),[0 50]);
ylim(axs(2,1),[0 80]);
%linkaxes(axs(2,:),'y');
% ylim(axs(2,1),[0 20]);

if DOSAVE
%  set(hFig,'InvertHardCopy','off');
  FNAME = 'Crop2exp_Ptiles';
  hgsave(hFig,fullfile('figsMe',[FNAME '.fig']));
  set(hFig,'PaperOrientation','landscape','PaperType','arch-c');
  print(hFig,'-dpdf',fullfile('figsMe',[FNAME '.fig']));
  print(hFig,'-dpng','-r300',fullfile('figsMe',[FNAME '.png']));
  %SaveFigLotsOfWays(hFig,'GTErrVsNTrain',{'fig' 'pdf'});
end

