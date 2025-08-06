function test_AR_GRONe_SA_tracking_conda()
  % Test tracking on a project using some of AR's flybubble videos.
  % That project is a multianimal project.
  if ispc() ,
    warning('conda backend is not supported on Windows, so %s always passes on Windows', mfilename());
    return
  end
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3_lightly_trained.lbl') ;  
  backend = 'conda' ;
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;  
  if ~isempty(tester.labeler.tracker.trkP)
    error('testObj.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;
  end
  tester.test_tracking('backend', backend, 'backend_params', backend_params);
  if ~isequal(size(tester.labeler.tracker.trkP.pTrk{3}), [17 2 201])
    error('After tracking, testObj.labeler.tracker.trkP.pTrk{3} is the wrong size') ;
  end
  if ~all(isfinite(tester.labeler.tracker.trkP.pTrk{3}), 'all')
    error('After tracking, testObj.labeler.tracker.trkP.pTrk{3} has non-finite elements') ;
  end
end  % function
