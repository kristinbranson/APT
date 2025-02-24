function test_sam_two_view_training()
  % Test training for alice project
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, '2011_mouse_cam13_updated_movie_paths_20241111_modded.lbl') ;
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training('backend', fif(ispc(), 'docker', 'bsub')) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'mdn_joint_fpn')
    error('Training was not done with GRONe aka mdn_joint_fpn') ;
  end
end  % function
