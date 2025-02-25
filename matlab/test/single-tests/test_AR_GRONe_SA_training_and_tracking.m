function test_AR_GRONe_SA_training_and_tracking()
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_2.lbl') ;  
  backend_params = janelia_bsub_backend_params() ;
  backend = fif(ispc(), 'docker', 'bsub') ;      
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;  
  tester.test_training('backend', backend, 'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'mdn_joint_fpn')
    error('Training was not done with GRONe aka mdn_joint_fpn') ;
  end
  if ~isempty(tester.labeler.tracker.trkP)
    error('testObj.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;
  end
  tester.test_tracking();
  if ~isequal(size(tester.labeler.tracker.trkP.pTrk{3}), [17 2 201])
    error('After tracking, testObj.labeler.tracker.trkP.pTrk{3} is the wrong size') ;
  end
  if ~all(isfinite(tester.labeler.tracker.trkP.pTrk{3}), 'all')
    error('After tracking, testObj.labeler.tracker.trkP.pTrk{3} has non-finite elements') ;
  end
end  % function
