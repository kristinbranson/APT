function test_ratcity_MA_magrone_tracking()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/ratCity_round12_movie_size.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
    % The graph-cut ID linking path needs the gco package, which is present in the
    % docker image (as of MK 20260506) but not in the conda environment.  See
    % synthesize_backend_params().
  backend = 'docker' ;
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  % test ID linking: train ID model for 200 iterations, track only 500 frames
  tester.test_id_tracking('backend', backend, ...
                          'backend_params', backend_params, ...
                          'startframe', 500, 'endframe',1000, ...
                          'id_niters', 200) ;
end  % function
