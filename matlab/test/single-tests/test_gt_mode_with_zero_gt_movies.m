function test_gt_mode_with_saved_tracks()
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
for i = 1:numel(children)
  h = children(i) ;
  if ~strcmp(get(h, 'Visible'), 'on')
    continue ;
  end
  if isa(h, 'matlab.graphics.primitive.Text')
    error('Found visible text object on axes_curr after entering GT mode with no GT movies: "%s".', ...
          get(h, 'String')) ;
  end
  if isa(h, 'matlab.graphics.chart.primitive.Line') || isa(h, 'matlab.graphics.primitive.Line')
    xd = get(h, 'XData') ;
    yd = get(h, 'YData') ;
    if any(~isnan(xd) & ~isnan(yd))
      error('Found visible line with non-NaN data on axes_curr after entering GT mode with no GT movies (Tag: "%s").', ...
            get(h, 'Tag')) ;
    end
  end
end

fprintf('test_gt_mode_with_saved_tracks passed.\n') ;

end  % function
