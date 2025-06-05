function test_alice_bubble_training()
  % Test training for alice project
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019_updated_20250306.lbl') ;
  training_params = struct('dlc_override_dlsteps', {true}) ;  % scalar struct
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training('algo_spec','deeplabcut', ...
                       'backend', docker_unless_janelia_cluster_then_conda(), ...
                       'training_params', training_params) ;
end  % function
