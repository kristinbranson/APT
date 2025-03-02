function test_roian_MA_dekr_training()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2024-11-19-with-rois-added-and-fewer-smaller-movies-mmpose1.lbl') ;  
  backend_params = janelia_bsub_backend_params() ;
  backend = 'docker' ;  % project is only set up with correct env/image for docker backend
  tester = LabelerProjectTester(project_file_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training('algo_name','multi_dekr', ...
                       'backend',backend, ...
                       'backend_params', backend_params) ;  
end  % function
