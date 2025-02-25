function test_roian_aws_training()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2024-11-19-with-rois-added-and-fewer-smaller-movies.lbl') ;
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend_params = janelia_taylora_aws_backend_params() ; 
  tester.test_training('algo_name','magrone', ...
                       'backend', 'aws', ...
                       'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'magrone')
    error('Training was not done with multianimal GRONe aka magrone') ;
  end
end  % function
