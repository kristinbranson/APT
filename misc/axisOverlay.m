function ax1 = axisOverlay(ax0,ax1)

if exist('ax1','var')==0
  ax1 = axes('Parent',ax0.Parent,'Visible','off','HitTest','off','Color','none');
end
PROPS = {'XLim' 'YLim' 'YDir' 'Position' 'PlotBoxAspectRatio'};
for p = PROPS,p=p{1}; %#ok<FXSET>
  ax1.(p) = ax0.(p);
end
linkaxes([ax0 ax1]);
hLink = linkprop([ax0 ax1],{'PlotBoxAspectRatio' 'XDir' 'YDir'});
ax1.UserData = hLink;