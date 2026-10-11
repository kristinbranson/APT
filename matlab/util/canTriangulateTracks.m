function [isPossible, message] = canTriangulateTracks(labeler, tblMFT)
  % Whether the movies in tblMFT (a table with a .mov column of movie
  % indices) can be tracked and then 3D-reconciled: a multiview project
  % whose 3D reconciliation is on needs calibration data for every movie
  % about to be tracked.  Returns true when the project is single-view or
  % reconciliation is off.  message says why not, else is empty.
  %
  % Formerly PostProcess.canPostProcess(); the heatmap-era PostProcess
  % class was removed on 2026-10-11.

  isPossible = true ;
  message = '' ;

  if labeler.nview == 1
    return
  end

  sPrmAll = labeler.trackGetTrainingParams() ;
  pptype = APTParameters.getPostProcessReconcile3dType(sPrmAll) ;
  if strcmpi(pptype, 'None')
    return
  end

  movieIndices = unique(tblMFT.mov) ;
  hasCalibration = true(size(movieIndices)) ;
  for i = 1:numel(movieIndices)
    vcd = labeler.getViewCalibrationDataMovIdx(movieIndices(i)) ;
    if isempty(vcd)
      hasCalibration(i) = false ;
    end
  end

  if any(~hasCalibration)
    isPossible = false ;
    message = sprintf(['Postprocessing includes triangulation, but movies with the following indices are missing ' ...
                       'calibration files: %s. Please add calibration files for these movies.'], ...
                      mat2str(movieIndices(~hasCalibration))) ;
  end
end  % function
