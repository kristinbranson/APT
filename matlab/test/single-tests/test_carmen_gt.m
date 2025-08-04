function test_carmen_gt()
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'pez7_al_updated_20241015_lightly_trained.lbl') ;  
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_gtcompute('backend',backend, ...
                        'backend_params', backend_params) ;  
  tbl = tester.labeler.gtTblRes ;
  actual_size = size(tbl) ;
  correct_size = [1539 13] ;
  if ~isequal(actual_size, correct_size)
      error('After GT tracking, tester.labeler.gtTblRes is of shape %s, but should be of shape %s', ...
            format_size(actual_size), ...
            format_size(correct_size)) ;
  end
  err = tbl.meanL2err ;
  if ~(median(err(:), 'omitnan') < 10)
    error('Median value of tester.labeler.gtTblRes.meanL2err(:) is too large') ;
  end
end  % function
