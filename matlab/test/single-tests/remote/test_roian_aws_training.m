function test_roian_aws_training()
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, ...
                               'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl') ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  backend = 'aws' ;
  backend_params = synthesize_backend_params(backend) ; 
  tester.test_training('algo_spec','magrone', ...
                       'backend', backend, ...
                       'backend_params', backend_params) ;
  if ~isequal(tester.labeler.tracker.algorithmName, 'magrone')
    error('Training was not done with multianimal GRONe aka magrone') ;
  end
end  % function
