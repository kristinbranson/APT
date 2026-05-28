function test_stephen_training()
  % Test training for stephen project
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/sh_test_lbl_20200310_modded_resaved_tweaked_20240122.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ; 
  tester.test_training('backend', backend, ...
                       'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'mdn_joint_fpn')
    error('Training was not done with GRONe aka mdn_joint_fpn') ;
  end
end  % function
