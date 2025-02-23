function test_carmen_gt()
  [~, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'pez7_al_updated_20241015_lightly_trained.lbl') ;  
  backend_params = janelia_bsub_backend_params() ;
  backend = fif(ispc(), 'docker', 'bsub') ;      
  tester = LabelerProjectTester(project_file_path) ;
  oc = onCleanup(@()(delete(tester))) ;  % force deletion on exit to ensure cleanup
  tester.test_gtcompute('backend',backend, ...
                        'backend_params', backend_params);
  tbl = tester.labeler.gtTblRes ;
  if ~isequal(size(tbl), [1539 11])
    error('After GT tracking, testObj.labeler.gtTblRes is the wrong size') ;      
  err = tbl.meanL2err ;
  if ~( median(err, 'omitnan') < 10 )
    error('Median value of testObj.labeler.gtTblRes.meanL2err is too large') ;
  end
end  % function
