function result = mainFigureFakeMenuTags()
% Return the tags of menus that are "fake" --- i.e. invisible menus that exist only
% to provide keyboard shortcuts via the Matlab menu accelerator mechanism.
result = {
  'menu_view_zoom_toggle'
  'menu_view_pan_toggle'
  'menu_view_hide_trajectories'
  'menu_view_hide_predictions'
  } ;
end % function
