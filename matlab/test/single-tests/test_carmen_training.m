function test_carmen_training()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'pez7_al_updated_20241015.lbl') ;
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend = 'docker' ;
  backend_params = synthesize_backend_params(backend) ; 
  tester.test_training('backend', backend, ...
                       'backend_params', backend_params) ;
end  % function
