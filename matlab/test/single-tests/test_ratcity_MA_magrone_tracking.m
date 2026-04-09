function test_ratcity_MA_magrone_tracking()
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'ratCity_round12_movie_size.lbl') ;
  backend = docker_unless_janelia_cluster_then_conda() ;
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
