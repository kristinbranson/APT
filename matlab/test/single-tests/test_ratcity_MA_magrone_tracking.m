function test_ratcity_MA_magrone_tracking(varargin)
  [backend] = myparse(varargin,'backend','');
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/ratCity_round12_movie_size.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  if strcmp(backend,'')
    backend = docker_unless_janelia_cluster_then_conda() ;
  end
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  % tester.test_tracking('backend', backend, ...
                       % 'backend_params', backend_params) ;
  % test ID linking: train ID model for 200 iterations, track only 500 frames
  tester.test_id_tracking('backend', backend, ...
                          'backend_params', backend_params, ...
                          'startframe', 500, 'endframe',1000, ...
                          'id_niters', 200) ;
end  % function
