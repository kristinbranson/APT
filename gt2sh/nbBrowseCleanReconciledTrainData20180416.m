load trnDataSH_20180417T094303.mat;
load trnDataSH_20180417T094303_IFinalReconciled.mat;

%% Montage: labels vs images
tfTrn1 = trnSets(:,1);
nnz(tfTrn1)
idstrs = strcat(tFinalReconciled.lblCat(tfTrn1),'|',numarr2trimcellstr(find(tfTrn1)));
hFigVw1 = figure('position',[1 41 2560 1484]);
hFigVw2 = figure('position',[2561 401 1920 1124]);

YELLOW = [1 1 0];
HOTPINK = [255 105 180]/255;
RED = [1 0 0];
COLORS = [HOTPINK;HOTPINK;YELLOW;YELLOW;RED];
MARKERSIZE = 20;

easymontage(IFinalReconciled(tfTrn1,1),tFinalReconciled.pLbl(tfTrn1,[1:5 11:15]),4,5,...
  'markersize',MARKERSIZE,'color',COLORS,'idstr',idstrs,'hFig',hFigVw1,...
  'doroi',true);
easymontage(IFinalReconciled(tfTrn1,2),tFinalReconciled.pLbl(tfTrn1,[6:10 16:20]),4,5,...
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

%% outliers RECORD ME
max(tFinalReconciled.pLbl,[],1)
idxOut = find(tFinalReconciled.pLbl(:,3)>1e3)
idxNan = find(any(isnan(tFinalReconciled.pLbl),2))
tUse = tFinalReconciled;
tUse([idxOut;idxNan],:) = [];


%% Label overlay. EMP: tFinalReconciled has 3 groups of overall locs
n = height(tUse);
pLbl = nan(n,5,2,2);
for iVw=1:2
  ptmp = tUse.pLbl(:,[1:5 11:15]+(iVw-1)*5);
  pLbl(:,:,:,iVw) = reshape(ptmp,n,5,2);
end
hFig = [figure('position',[1 41 2560 1484]) figure('position',[2561 401 1920 1124])];

for iVw=1:2
  figure(hFig(iVw));
  ax = axes;
  hold(ax,'on');
  ax.Color = [0 0 0];
  for ipt=1:5
    plot(pLbl(:,ipt,1,iVw),pLbl(:,ipt,2,iVw),'.','markerSize',10,'color',COLORS(ipt,:));
  end
  axis(ax,'auto','ij');
  title(sprintf('vw%d',iVw),'fontweight','bold');
end
  
%% Centroid overlay
pLblCents = squeeze(mean(pLbl,2));
szassert(pLblCents,[height(tUse) 2 2]); % i,x/y,iVw
g = tUse.lblCat;
hFig = [figure('position',[1 41 2560 1484]) figure('position',[2561 401 1920 1124])];
for iVw=1:2
  figure(hFig(iVw));
  gscatter(pLblCents(:,1,iVw),pLblCents(:,2,iVw),g);
  axis(gca,'auto');
  axis(gca,'ij');
  title(sprintf('vw%d',iVw),'fontweight','bold');
end
%%
iExLow = find(pLblCents(:,2,2)<100,1)
iExHigh = find(pLblCents(:,2,2)>500,1)

%% Scales

%% to deal with: outliers, NaNs, overall loc, and scale
