function test_chimpact_MA_top_down_detr_training()
  % Test training of the ChimpACT multi-animal top-down 2-stage tracker
  % (stage 1: detect_mmdetect detector, stage 2: mdn_joint_fpn pose).
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/chimpAct_noTestlabels_detr400k_nocrop.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_training('algo_spec',[ DLNetType.detect_mmdetect DLNetType.mdn_joint_fpn ], ...
                       'backend',backend, ...
                       'backend_params', backend_params) ;
end  % function
