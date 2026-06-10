function test_gt_mode_with_zero_gt_movies()
% Test that Evaluate > Groundtruthing Mode can be entered on a project with saved tracks.

[~, unittest_dir_path, replace_path] = get_test_project_paths() ;
project_file_path = fullfile(unittest_dir_path, 'htflies-10-with-saved-tracks.lbl') ;
[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

% Actuate Evaluate > Groundtruthing Mode
controller.menu_evaluate_gtmode_actuated_([], []) ;
drawnow() ;

assert(labeler.gtIsGTMode, ...
       'Expected to be in GT mode after actuation.') ;

% With no GT movies, the axes should have no visible label-like graphics
ax = controller.axes_curr ;
children = get(ax, 'Children') ;
childCount = numel(children) ;
isBadFromChildIndex = false(1, childCount) ;
for i = 1:childCount
  h = children(i) ;
  if ~strcmp(get(h, 'Visible'), 'on')
    continue ;
  end
  if isa(h, 'matlab.graphics.primitive.Text')
    position = get(h, 'Position') ;
    str = get(h, 'String') ;
    if all(isfinite(position)) && ~isempty(str)
      isBadFromChildIndex(i) = true ;
    end
  end
  if isa(h, 'matlab.graphics.chart.primitive.Line') || isa(h, 'matlab.graphics.primitive.Line')
    xd = get(h, 'XData') ;
    yd = get(h, 'YData') ;
    if any(~isnan(xd) & ~isnan(yd))
      isBadFromChildIndex(i) = true ;
    end
  end
end

if any(isBadFromChildIndex)
  badChildren = children(isBadFromChildIndex) ;
  badTags = arrayfun(@(h) sprintf('%s:%s', class(h), get(h, 'Tag')), badChildren, 'UniformOutput', false) ;
  error('Found %d visible text/line objects on axes_curr after entering GT mode with no GT movies:\n  %s', ...
        numel(badTags), strjoin(badTags, '\n  ')) ;
end

fprintf('test_gt_mode_with_zero_gt_movies passed.\n') ;

end  % function
