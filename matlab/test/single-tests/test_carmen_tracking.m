function test_carmen_tracking()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'pez7_al_updated_20241015_lightly_trained.lbl') ;
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  if ~isempty(tester.labeler.tracker.trkP)
    error('tester.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;  
  end
  backend_params = janelia_bsub_backend_params() ;
  tester.test_tracking('backend', fif(ispc(), 'docker', 'bsub'), ...
                       'backend_params', backend_params) ;
  if ~isequal(size(tester.labeler.tracker.trkP.pTrk{1}), [10 2 101])
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
  end
  if ~all(isfinite(tester.labeler.tracker.trkP.pTrk{1}), 'all')
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
  end  
end  % function
