function test_carmen_training()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/pez7_al_updated_20241015.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ; 
  tester.test_training('backend', backend, ...
                       'backend_params', backend_params) ;
end  % function
