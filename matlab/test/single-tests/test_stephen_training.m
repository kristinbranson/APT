function test_stephen_training()
  % Test training for stephen project
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'sh_test_lbl_20200310_modded_resaved_tweaked_20240122.lbl') ;
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend_params = janelia_bsub_backend_params() ; 
  tester.test_training('backend', fif(ispc(), 'docker', 'bsub'), ...
                       'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'mdn_joint_fpn')
    error('Training was not done with GRONe aka mdn_joint_fpn') ;
  end
end  % function
