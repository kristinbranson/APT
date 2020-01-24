function hfig = PlotSampleFrame(savefile,varargin)

[deltaT,landmarksPlotTrx,colorsOverride,hfig,lw,ms,lwTrx] = myparse(varargin,'deltaT',30,'landmarksPlotTrx',[],'colorsOverride',[],'hfig',[],'LineWidth',2,'MarkerSize',10,'LineWidthTrx',1);

load(savefile); %#ok<LOAD>

if ~isempty(colorsOverride),
  colors = colorsOverride;
end

if isempty(hfig) || ~ishandle(hfig),
  hfig = figure;
else
  figure(hfig);
end
nViews = numel(currIm); %#ok<*USENS>

nPts = max(predIdx.pt);
nTargets = max(predIdx.tgt);

hax = createsubplots(1,nViews);
jx = find(strcmp(imPropNames,'XData'));
jy = find(strcmp(imPropNames,'YData'));
for i = 1:nViews,
  imagesc(imProps{i}{jx},imProps{i}{jy},currIm{i},'Parent',hax(i)); %#ok<*IDISVAR>
  axis(hax(i),'image','off');
%   for j = 1:numel(axPropNames),
%     hax(i).(axPropNames{j}) = axProps{i}{j};
%   end
  hold(hax(i),'on');
  colormap(hax(i),'gray');
end

mint = t-deltaT;
maxt = t;

idxt = predIdx.frm == t;
nPtsPerView = nPts / nViews;
if isempty(landmarksPlotTrx),
  landmarksPlotTrx = 1:nPts;
end

for jTarg = 1:nTargets,
  for i = landmarksPlotTrx,
    idxtargptd = find(predIdx.tgt == jTarg & predIdx.pt==i & predIdx.d == 1);
    [ism,idx] = ismember(mint:maxt,predIdx.frm(idxtargptd));
    x = nan(1,maxt-mint+1);
    x(ism) = xyPredictions(idxtargptd(idx(ism)));
    idxtargptd = find(predIdx.tgt == jTarg & predIdx.pt==i & predIdx.d == 2);
    [ism,idx] = ismember(mint:maxt,predIdx.frm(idxtargptd));
    y = nan(1,maxt-mint+1);
    y(ism) = xyPredictions(idxtargptd(idx(ism)));
    [iColor,iView] = ind2sub([nPtsPerView,nViews],i);
    plot(hax(iView),x,y,'-','Color',colors(iColor,:),'LineWidth',lwTrx);
  end
end


for iView = 1:nViews,
  
  
  for i = 1:nPtsPerView,
    iPt = (iView-1)*nPtsPerView+i;
    j1 = find(idxt & predIdx.pt == iPt & predIdx.d == 1);
    j2 = find(idxt & predIdx.pt == iPt & predIdx.d == 2);
    plot(hax(iView),xyPredictions(j1),xyPredictions(j2),'+','LineWidth',lw,'Color',colors(i,:),'MarkerSize',ms); %#ok<*NODEF>
  end
end