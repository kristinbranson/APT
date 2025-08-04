function test_roian_MA_gt()
  if ispc() 
    warning('The project videos are too large to easily copy, so %s always passes on Windows', mfilename());
    return
  end    
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2024-11-19-with-gt-added.lbl') ;  
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_gtcompute('backend',backend, ...
                        'backend_params', backend_params) ;  
  tbl = tester.labeler.gtTblRes ;
  if ~isequal(size(tbl), [11 13])
      error('After GT tracking, testObj.labeler.gtTblRes is the wrong size') ;
  end
  err = tbl.meanL2err ;
  if ~(median(err(:), 'omitnan') < 50)
    error('Median value of tester.labeler.gtTblRes.meanL2err(:) is too large') ;
  end
end  % function
