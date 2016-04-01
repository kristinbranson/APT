function ax1 = axisOverlay(ax0)

ax1 = axes('Parent',ax0.Parent,'Visible','off','HitTest','off','Color','none');
PROPS = {'XLim' 'YLim' 'YDir' 'Position' 'PlotBoxAspectRatio'};
for p = PROPS,p=p{1}; %#ok<FXSET>
  ax1.(p) = ax0.(p);
end