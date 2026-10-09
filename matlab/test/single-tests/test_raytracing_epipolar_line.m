function test_raytracing_epipolar_line()
  % The ray-tracing calibration rig (CalRigNPairwiseCalibratedRayTracing)
  % computes an epipolar line by running
  % raytracing_calib/programs/return_projected_ray_two_cameras_prism.py in
  % the APT_raytracing conda env through pyrunfile().  This pins the lines
  % the rig returns for one flyprism calibration (the model file lives with
  % the other test fixtures), so the Python bridge -- pyenv(), pyrunfile()
  % and its name=value arguments -- is exercised end to end.  Needs the
  % APT_raytracing conda env, as the rig itself does.

  linux_model_path = '/groups/branson/bransonlab/PTR_fly_annotations_3D/exp55/best_model_weights_only.pth' ;
  model_path = localize_test_project_path(linux_model_path) ;

  calibration = struct() ;
  calibration.nviews = 4 ;
  calibration.calibrations = {} ;
  calibration.model_path = model_path ;
  calibration.dividing_col = [1255 1255] ;
  calibration.image_width = [1920 1920] ;
  rig = CalRigNPairwiseCalibratedRayTracing(calibration) ;

  % View 1 (primary virtual) labelled, line drawn in view 2 (primary real).
  [xEPL, yEPL] = rig.computeEpiPolarLine(1, [500 600], 2, []) ;
  assert(isequal(size(xEPL), [1 2]) && isequal(size(yEPL), [1 2]), ...
         'The epipolar line is not two points: x is %s, y is %s', mat2str(size(xEPL)), mat2str(size(yEPL))) ;
  assert(all(abs(xEPL - 498.794711) < 1e-3) && all(abs(yEPL - 5010.806941) < 1e-3), ...
         'Epipolar line 1->2 was x = %s, y = %s', mat2str(xEPL, 9), mat2str(yEPL, 9)) ;

  % View 2 (primary real) labelled, line drawn in view 3 (secondary virtual).
  [xEPL, yEPL] = rig.computeEpiPolarLine(2, [700 400], 3, []) ;
  assert(all(abs(xEPL - 1.009155) < 1e-3) && all(abs(yEPL - 1254.985352) < 1e-3), ...
         'Epipolar line 2->3 was x = %s, y = %s', mat2str(xEPL, 9), mat2str(yEPL, 9)) ;
end  % function
