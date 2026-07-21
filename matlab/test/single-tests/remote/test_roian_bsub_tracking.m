function test_roian_bsub_tracking()
  if ispc() 
    warning('bsub backend is not supported on Windows, so %s always passes on Windows', mfilename());
    return
  end  
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  if ~isempty(tester.labeler.tracker.trkP)
    error('tester.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;  
  end
  backend = 'bsub' ;
  backend_params = synthesize_backend_params(backend) ;
  tester.test_tracking('algo_name', 'magrone', ...
                       'backend', backend, ...
                       'backend_params', backend_params) ;
  if ~isequal(size(tester.labeler.tracker.trkP.pTrk{1}), [4 2 101])
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
  end
  if ~all(isfinite(tester.labeler.tracker.trkP.pTrk{1}), 'all')
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
  end  
end  % function
