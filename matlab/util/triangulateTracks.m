function trksnew = triangulateTracks(trks, crig)
  % Reconcile multiview tracking results in 3D by triangulating each
  % landmark through the calibration rig.  trks is a cell array of raw
  % trkfile structs, one per view, as DeepTracker.loadRawTrackFile() returns
  % them; crig is the project's CalRig (or a cell holding it).  Returns a
  % cell array of structs, one per view, each with the single-view positions
  % kept as pTrkSingleView, the reprojected positions as pTrk, and the 3D
  % positions (npt x 3 x nfrm) as pTrk3d.  Can throw.
  %
  % Formerly PostProcess.triangulate(); the heatmap-era PostProcess class
  % was removed on 2026-10-11, this being the one piece of it APT used.

  nvw = length(trks) ;
  if iscell(crig)
    crig = crig{1} ;
  end

  for i = 2:length(trks)
    if ~isequal(trks{i-1}.pTrkFrm, trks{i}.pTrkFrm) || ~isequal(trks{i-1}.pTrkiTgt, trks{i}.pTrkiTgt)
      error('Cannot perform 3D postprocessing; trkfiles differ in frames/targets tracked.') ;
    end
  end

  if isfield(trks{1}, 'pTrk3D')
    error('Trkfile for view 1 already contains ''pTrk3D'' field. This trkfile has already been 3d-postprocessed.') ;
  end

  ptrks = cell(nvw, 1) ;
  for i = 1:nvw
    ptrks{i} = trks{i}.pTrk{1} ;
  end

  [npt, ~, nfrm, ntgt] = size(ptrks{1}) ;
  for i = 2:nvw
    assert(isequal(size(ptrks{1}), size(ptrks{i})), ...
           'Trkfiles contain position arrays with inconsistent sizes.') ;
  end

  fprintf(1, 'Performing 3d reconciliation: triangulate...\n') ;
  wbObj = WaitBarWithCancelCmdline('3d reconciliation') ;
  oc = onCleanup(@()delete(wbObj)) ;

  assert(ntgt == 1, 'Expected single-target data in trkfiles.') ;
  ptrk_reshaped = cell(nvw, 1) ;
  for i = 1:nvw
    ptrk_reshaped{i} = reshape(permute(ptrks{i}, [2 3 1]), 2, nfrm*npt) ;  % coord, frm*pt
  end
  all_ptrks = cat(3, ptrk_reshaped{:}) ;

  X = nan(3, nfrm*npt) ;
  ptrkrp = nan(size(all_ptrks)) ;
  for i = 1:nfrm*npt
    [X(:, i), ptrkrp(:, i, :)] = crig.triangulate(all_ptrks(:, i, :)) ;
  end
  X = permute(reshape(X, [3 nfrm npt]), [3 1 2]) ;  % npt x 3 x nfrm
  ptrkrp = reshape(ptrkrp, [2 nfrm npt 1 nvw]) ;
  ptrkrp = permute(ptrkrp, [3 1 2 4 5]) ;  % npt x 2 x nfrm x 1 x nvw

  trksnew = cell(nvw, 1) ;
  for i = 1:nvw
    trksnew{i} = struct( ...
      'pTrkSingleView', {trks{i}.pTrk}, ...
      'pTrk', {{ptrkrp(:, :, :, :, i)}}, ...
      'pTrk3d', {{X}} ...
      ) ;
  end
end  % function
