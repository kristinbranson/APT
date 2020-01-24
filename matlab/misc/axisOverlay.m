function ax1 = axisOverlay(ax0,ax1)

if exist('ax1','var')==0
  ax1 = axes('Parent',ax0.Parent,'Visible','off','PickableParts','none','Color','none');
end
PROPS = {'XLim' 'YLim' 'YDir' 'Position' 'PlotBoxAspectRatio'};
for p = PROPS,p=p{1}; %#ok<FXSET>
  if ~isequal(ax1.(p),ax0.(p))
    % AL20160928. Protect against setting props twice. When ax0 and ax1 are
    % already linked via linkaxes, setting PROPS can set XLimMode/YLimMode
    % to 'manual', which may be undesirable.
    ax1.(p) = ax0.(p);
  end
end
% AL20160928: preserve X/Ylimmode, linkaxes messes with it. It's possible
% this is necessary in some use cases but so far we don't need that.
xlm0 = ax0.XLimMode;
ylm0 = ax0.YLimMode;
xlm1 = ax1.XLimMode;
ylm1 = ax1.YLimMode;
linkaxes([ax0 ax1]);
ax0.XLimMode = xlm0;
ax0.YLimMode = ylm0;
ax1.XLimMode = xlm1;
ax1.YLimMode = ylm1;
hLink = linkprop([ax0 ax1],{'PlotBoxAspectRatio' 'XDir' 'YDir'});
ax1.UserData = hLink;