function hfig = PlotSampleFrame(savefile,varargin)

[deltaT,landmarksPlotTrx] = myparse(varargin,'deltat',50,'landmarksPlotTrx',[]);

load(savefile); %#ok<LOAD>

hfig = figure;
nViews = numel(currIm); %#ok<*USENS>

nPts = max(predIdx.pt);
nTargets = max(predIdx.tgt);

hax = createsubplots(1,nViews);
for i = 1:nViews,
  imagesc(currIm{i},'Parent',hax(i)); %#ok<*IDISVAR>
  axis(hax(i),'image');
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
for iView = 1:nViews,
  
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
      plot(x,y,'-','Color',colors(i,:));
    end
  end
  
  for i = 1:nPtsPerView,
    iPt = (iView-1)*nPtsPerView+i;
    j1 = find(idxt & predIdx.pt == iPt & predIdx.d == 1);
    j2 = find(idxt & predIdx.pt == iPt & predIdx.d == 2);
    plot(hax(iView),xyPredictions(j1),xyPredictions(j2),'+','LineWidth',2,'Color',colors(i,:),'MarkerSize',10); %#ok<*NODEF>
  end
end