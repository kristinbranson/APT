function test_carmen_tracking()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'pez7_al_updated_20241015_lightly_trained.lbl') ;  
  backend_params = janelia_bsub_backend_params() ;
  backend = fif(ispc(), 'docker', 'bsub') ;      
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;  % force deletion on exit to ensure cleanup
  tester.test_tracking('backend',backend, ...
                       'backend_params', backend_params);
  if ~all(size(tester.labeler.tracker.trkP.pTrk{1})==[10 2 101])
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
  end
  if ~all(isfinite(tester.labeler.tracker.trkP.pTrk{1}), 'all') 
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
  end
end  % function
