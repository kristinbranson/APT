function test_ratcity_MA_magrone_nocrop_training()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/ratCity_round12_movie_size.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  % backend = docker_unless_janelia_cluster_then_conda() ;
  backend = 'bsub' ;  % Need a lot of VRAM
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  % without cropping
  training_params = struct('multi_crop_ims',false,'batch_size',2);
  tester.test_training('algo_spec', DLNetType.multi_mdn_joint_torch, ...
                       'backend', backend, ...
                       'backend_params', backend_params,...
                       'training_params',training_params) ;
end  % function
