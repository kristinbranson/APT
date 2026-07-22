function test_chimpact_MA_top_down_detr_gt()
  % Test ground-truth accuracy computation for the ChimpACT multi-animal
  % top-down 2-stage tracker (stage 1: detect_mmdetect, stage 2: mdn_joint_fpn),
  % using the trained tracker and GT-labeled frames present in the project.
  if ispc()
    warning('The project videos are too large to easily copy, so %s always passes on Windows', mfilename());
    return
  end
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'chimpAct_noTestlabels_detr400k_nocrop.lbl') ;
  backend = docker_unless_janelia_cluster_then_conda() ;
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  tester.test_gtcompute('backend',backend, ...
                        'backend_params', backend_params) ;
  tbl = tester.labeler.gtTblRes ;
  if isempty(tbl)
    error('After GT tracking, tester.labeler.gtTblRes is empty') ;
  end
  err = tbl.meanL2err ;
  if ~(median(err(:), 'omitnan') < 50)
    error('Median value of tester.labeler.gtTblRes.meanL2err(:) is too large') ;
  end
end  % function
