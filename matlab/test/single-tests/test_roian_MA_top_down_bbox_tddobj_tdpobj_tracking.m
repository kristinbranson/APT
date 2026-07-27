function test_roian_MA_top_down_bbox_tddobj_tdpobj_tracking()
  % algo_spec = 'ma_top_down_bbox_tddobj_tdpobj' ;
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-2-stages.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_tracking('backend',backend, ...
                       'backend_params', backend_params) ;  
end  % function
