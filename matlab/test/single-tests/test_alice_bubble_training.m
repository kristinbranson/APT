function test_alice_bubble_training()
  % Test training for alice project
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019_updated_20250306.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  training_params = struct('dlc_override_dlsteps', {true}) ;  % scalar struct
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ;
  % algo_spec = 'deeplabcut' ;
  algo_spec = DLNetType.deeplabcut ;  
  tester.test_training('algo_spec',algo_spec, ...
                       'backend', backend, ...
                       'backend_params', backend_params, ...
                       'training_params', training_params) ;
end  % function
