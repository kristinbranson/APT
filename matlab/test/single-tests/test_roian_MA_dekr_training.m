function test_roian_MA_dekr_training()
  lsb_exec_cluster = getenv('LSB_EXEC_CLUSTER') ;
  if strcmpi(lsb_exec_cluster, 'Janelia')
    warning('Because the backend environment for the %s is sort of a one-off, this test always passes on the Janelia LSF cluster.', mfilename());
    % Revisit this once the h100 environment is used for everything.  
    % -- ALT, 2025-06-24
    return
  end    
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies-mmpose1.lbl') ;  
  backend = 'docker' ;  % project is only set up with correct env/image for docker backend
  backend_params = synthesize_backend_params(backend) ;
  %backend_params = cell(1,0) ;  % Use the docker params in the project, b/c dekr use an mmpose 1 environment
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  % algo_spec = 'multi_dekr' ;
  algo_spec = DLNetType.multi_dekr ;
  tester.test_training('algo_spec',algo_spec, ...
                       'backend',backend, ...
                       'backend_params', backend_params, ...
                       'niters', 600) ;  
end  % function
