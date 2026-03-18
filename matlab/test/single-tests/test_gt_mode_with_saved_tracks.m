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

fprintf('test_gt_mode_with_saved_tracks passed.\n') ;

end  % function
