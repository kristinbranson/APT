function test_prediction_lines_in_axes()
% Test that prediction line objects in axes_curr have the expected tags
% and non-NaN coordinate data after loading a tracked project.
linux_project_file_path = ...
  '/groups/branson/bransonlab/apt/unittest/with-trx-project-with-short-movie-tracked.lbl' ;
if ispc()
  project_file_path = strrep(linux_project_file_path, '/groups/branson/bransonlab', 'Z:') ;
  replace_path = { '/groups/branson/bransonlab', 'Z:' } ;
else
  project_file_path = linux_project_file_path ;
  replace_path = [] ;
end

[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleaner = onCleanup(@()(delete(controller))) ;
cleaner2 = onCleanup(@()(delete(labeler))) ;

% Show predictions for all targets
controller.controlActuated('menu_view_showhide_preds_all_targets') ;

% Check that axes_curr has 17 prediction line objects with the expected tags
nLandmarks = 17 ;
nTargets = 10 ;
for iLandmark = 1:nLandmarks
  tag = sprintf('dt_mdn_joint_fpn_pred_%d', iLandmark) ;
  hLine = findobj(controller.axes_curr, 'Tag', tag) ;
  if isempty(hLine)
    error('Could not find line object with tag %s', tag) ;
  end
  if numel(hLine) ~= 1
    error('Expected 1 line object with tag %s, but found %d', tag, numel(hLine)) ;
  end
  xData = get(hLine, 'XData') ;
  yData = get(hLine, 'YData') ;
  if numel(xData) ~= nTargets
    error('Tag %s: expected XData to have %d elements, but got %d', ...
          tag, nTargets, numel(xData)) ;
  end
  if numel(yData) ~= nTargets
    error('Tag %s: expected YData to have %d elements, but got %d', ...
          tag, nTargets, numel(yData)) ;
  end
  if any(isnan(xData))
    error('Tag %s: XData contains NaNs', tag) ;
  end
  if any(isnan(yData))
    error('Tag %s: YData contains NaNs', tag) ;
  end
end

end  % function
