function test_sam_two_view_tracking()
  % Test training for alice project
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, '2011_mouse_cam13_updated_movie_paths_20241111_modded_lightly_trained.lbl') ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  if ~isempty(tester.labeler.tracker.trkP)
    error('tester.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;  
  end
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ;
  tester.test_tracking('backend', backend, ...
                       'backend_params', backend_params) ;
  if ~isequal(size(tester.labeler.tracker.trkP.pTrk{1}), [14 2 201])
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
  end
  if ~all(isfinite(tester.labeler.tracker.trkP.pTrk{1}), 'all')
    error('After tracking, tester.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
  end  
end  % function
