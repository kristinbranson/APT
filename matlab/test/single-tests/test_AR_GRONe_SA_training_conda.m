function test_AR_GRONe_SA_training_conda()
  % Test training on a project using some of AR's flybubble videos.
  % That project is a multianimal project.
  if ispc() ,
    warning('conda backend is not supported on Windows, so %s always passes on Windows', mfilename());
    return
  end
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3.lbl') ;  
  backend = 'conda' ;
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;  
  tester.test_training('backend', backend, 'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'mdn_joint_fpn')
    error('Training was not done with GRONe aka mdn_joint_fpn') ;
  end
  if ~isempty(tester.labeler.tracker.trkP)
    error('testObj.labeler.tracker.trkP is nonempty---it should be empty just after training') ;
  end
end  % function
