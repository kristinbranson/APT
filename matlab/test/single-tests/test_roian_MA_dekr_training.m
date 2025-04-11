function test_roian_MA_dekr_training()
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2024-11-19-with-rois-added-and-fewer-smaller-movies-mmpose1.lbl') ;  
  backend = docker_unless_janelia_cluster_then_conda() ;  % project is only set up with correct env/image for docker backend
  %backend_params = synthesize_backend_params(backend) ;
  backend_params = cell(1,0) ;  % Use the docker params in the project, b/c dekr use an mmpose 1 environment
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training('algo_name','multi_dekr', ...
                       'backend',backend, ...
                       'backend_params', backend_params) ;  
end  % function
