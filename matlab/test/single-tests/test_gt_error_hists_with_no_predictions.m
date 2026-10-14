function test_gt_error_hists_with_no_predictions()
  % Tests that the ground-truth error histograms can be plotted when the tracker
  % predicted nothing for the ground-truth frames.  In that case every L2 error is
  % missing, so LabelerController.createGTResultFigures_() hands PlotErrorHists() an
  % error array that is either empty (all rows dropped as all-NaN) or all-NaN.
  % PlotErrorHists() should draw empty histograms rather than erroring while deriving
  % bin edges from data that has no finite values.

  keypointCount = 16 ;
  viewCount = 1 ;

  % No ground-truth rows survive filtering, because none of them has any prediction.
  % This is what createGTResultFigures_() produces once it drops the all-NaN rows.
  emptyErrors = nan(0, keypointCount, viewCount) ;
  checkPlotErrorHistsRuns_(emptyErrors, 'errors for zero ground-truth rows') ;

  % Rows are present, but every error within them is missing.
  missingErrors = nan(10, keypointCount, viewCount) ;
  checkPlotErrorHistsRuns_(missingErrors, 'all-missing errors') ;

  % Every error takes the same value, so the bin edges derived from the data would
  % collapse to a single repeated point.
  constantErrors = zeros(10, keypointCount, viewCount) ;
  checkPlotErrorHistsRuns_(constantErrors, 'all-zero errors') ;
end  % function

function checkPlotErrorHistsRuns_(errors, description)
  % Calls PlotErrorHists() on errors, and errors out if it fails.
  fig = figure('Visible', 'off') ;
  oc = onCleanup(@()(delete(fig))) ;  %#ok<NASGU>
  try
    PlotErrorHists(errors, ...
                   'hparent', fig, ...
                   'prc_vals', [50 75 90 95 98], ...
                   'nbins', 50, ...
                   'maxprctile', 98, ...
                   'kpnames', arrayfun(@(i)(sprintf('pt%d', i)), (1:size(errors,2))', 'UniformOutput', false)) ;
  catch me
    error('PlotErrorHists() failed on %s: %s', description, me.message) ;
  end
end  % function
