function test_carmen_training()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'pez7_al_updated_20241015.lbl') ;  
  backend_params = janelia_bsub_backend_params() ;
  backend = fif(ispc(), 'docker', 'bsub') ;      
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;  % force deletion on exit to ensure cleanup
  tester.test_training('backend',backend, ...
                       'backend_params', backend_params);
end  % function
