function test_alice_project()
  % Test training for alice project
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019_updated_20250219.lbl') ;
  training_params = struct('dlc_override_dlsteps', {true}) ;  % scalar struct
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training('algo_name','deeplabcut', ...
                       'backend', 'docker', ...
                       'training_params', training_params) ;
end  % function
