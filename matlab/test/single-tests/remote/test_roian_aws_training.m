function test_roian_aws_training()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend = 'aws' ;
  backend_params = synthesize_backend_params(backend) ; 
  tester.test_training('algo_spec',DLNetType.multi_mdn_joint_torch, ...
                       'backend', backend, ...
                       'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'magrone')
    error('Training was not done with multianimal GRONe aka magrone') ;
  end
end  % function
