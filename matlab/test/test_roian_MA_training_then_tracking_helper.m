function test_roian_MA_training_then_tracking_helper(algo_name)
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2024-11-19-with-rois-added-and-fewer-smaller-movies.lbl') ;  
  backend_params = janelia_bsub_backend_params() ;
  backend = fif(ispc(), 'docker', 'bsub') ;      
  tester = LabelerProjectTester(project_file_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training_then_tracking('algo_name',algo_name, ...
                                     'backend',backend, ...
                                     'backend_params', backend_params) ;
end  % function
