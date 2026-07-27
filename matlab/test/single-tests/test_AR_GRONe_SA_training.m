function test_AR_GRONe_SA_training()
  % Test training and tracking on a project using some of AR's flybubble videos.
  % That project is a multianimal project.
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
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
