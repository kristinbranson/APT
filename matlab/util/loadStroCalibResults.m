function [nameL,nameR,info,intrinsic,extrinsic] = loadStroCalibResults(file,tfAssertRecompute)
  % Load a Bouguet-toolbox stereo calibration results file
  % (Calib_Results_stereo.mat) into name/info/intrinsic/extrinsic structs.
  %
  % file: path to the .mat file
  % tfAssertRecompute: if true (default), assert that the stereo calibration
  %   did not recompute the single-camera intrinsics
  %
  % intrinsic.l, intrinsic.r: structs with fields alpha_c, cc, fc, kc
  % extrinsic: om, T, R for coords-in-Right-frame wrt coords-in-Left-frame

  if ~exist('tfAssertRecompute','var')
    tfAssertRecompute = true;
  end

  cr = load(file);

  nameL = cr.calib_name_left;
  nameR = cr.calib_name_right;

  if tfAssertRecompute
    assert(cr.recompute_intrinsic_left==0);
    assert(cr.recompute_intrinsic_right==0);
  end
  info = struct();
  info.dX = cr.dX;
  info.nx = cr.nx;
  info.ny = cr.ny;
  info.normT = 2*cr.dX;

  fprintf(1,'[cc_left cc_left_error]:\n');
  disp([round(cr.cc_left) cr.cc_left_error]);
  fprintf(1,'[cc_right cc_right_error]:\n');
  disp([round(cr.cc_right) cr.cc_right_error]);
  intrinsic = struct();
  PRMS = {'alpha_c' 'cc' 'fc' 'kc'};
  for p=PRMS,p=p{1}; %#ok<FXSET>
    intrinsic.l.(p) = cr.([p '_left']);
    intrinsic.r.(p) = cr.([p '_right']);
  end

  extrinsic = struct();
  extrinsic.om = cr.om; % om, T, R are for coords-in-Right-frame wrt coords-in-Left-frame
  extrinsic.T = cr.T;
  extrinsic.R = rodrigues(cr.om); % X_Right = R*X_Left + T
end  % function
