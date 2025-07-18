function makeFigureMenubarAndToolbarAPTAppropriateBang(fig)
% Delete buttons from the figure toolbar that are not wanted in APT, etc

% Want the zoom/pan buttons in the figure toolbar
addToolbarExplorationButtons(fig) ;  
  % N.B.: this sets fig.MenuBar to 'figure'

% Make sure the menubar and toolbar are showing
fig.MenuBar = fif(strcmp(fig.Tag, 'main_figure'), 'figure', 'none') ;
fig.ToolBar = 'figure' ;
fig.DockControls = false ;

% Delete all the default menubar items
pulldownMenus = findall(fig,'type','uimenu') ;
deleteValidGraphicsHandles(pulldownMenus) ;

% Delete all the toolbar buttons we don't want.
hs = findall(fig,'type','uitoolbar');
KEEP = {'Exploration.Rotate' 'Exploration.Pan' 'Exploration.ZoomOut' ...
        'Exploration.ZoomIn'};
hh = findall(hs,'-not','type','uitoolbar','-property','Tag');
for hs=hh(:)'
  if ~ishandle(hs),
    continue
  end
  if ~any(strcmp(hs.Tag,KEEP))
    delete(hs);
  end
end

% Fix for >=2025a issues
if isMATLABReleaseOlderThan('R2025a')
  % do nothing
else
  enableLegacyExplorationModes(fig) ;
end

% Configure a callback to keep rotations 2D
r = rotate3d(fig) ;
r.ActionPostCallback = @rectifyMainAxesRotation ;

end  % function



function rectifyMainAxesRotation(fig, ~)
ax = findall(fig, 'Tag', 'axes_curr') ;
if ~isscalar(ax)
  % Something is wrong---just exit
  return
end
[az, ~] = view(ax) ;
ax.View = [az 90] ;  % Force the view to be from directly above
end  % function

