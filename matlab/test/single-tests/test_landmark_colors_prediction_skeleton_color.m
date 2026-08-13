function test_landmark_colors_prediction_skeleton_color()
  % Headless test for the independent prediction skeleton color in the Landmark
  % Cosmetics dialog.
  %
  % Labels and predictions store separate skeleton colors in the model
  % (labelPointsPlotInfo vs predPointsPlotInfo), but the dialog used to expose a
  % single swatch that was applied to both.  This test constructs the controller
  % with a lightweight mock Labeler and asserts that both swatches exist, that
  % they pick up their respective colors from the model, and that editing one
  % sends a skeletonSpecs struct in which the two landmark set types carry
  % different colors while still sharing a line width.

  labelColor = [1 0 0] ;
  predColor = [0 0 1] ;
  lastSkeletonSpecs = [] ;

  labeler = makeMockLabeler_(labelColor, predColor) ;
  labeler.setLandmarkAndSkeletonCosmetics = ...
    @(colorSpecs, markerSpecs, skeletonSpecs, trajSpecs)(recordApply(skeletonSpecs)) ;
  parent = struct('mainFigurePixelPosition', [100 100 800 600]) ;
  controller = LandmarkColorsController(parent, labeler) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  % Both swatches should exist and be parented to the dialog.
  widgetProps = {'pbSkeletonColor_', 'pbSkeletonColorPred_'} ;
  for i = 1 : numel(widgetProps)
    prop = widgetProps{i} ;
    h = controller.(prop) ;
    assert(~isempty(h) && isgraphics(h), 'Skeleton widget "%s" is not a valid graphics handle', prop) ;
    assert(ancestor(h, 'figure') == controller.hFig, ...
           'Skeleton widget "%s" is not parented to the dialog figure', prop) ;
  end

  % Each swatch should reflect its own landmark set type's color, not a shared one.
  assert(isequal(controller.pbSkeletonColor_.BackgroundColor, labelColor), ...
         'Label skeleton swatch did not reflect the label skeleton color') ;
  assert(isequal(controller.pbSkeletonColorPred_.BackgroundColor, predColor), ...
         'Prediction skeleton swatch did not reflect the prediction skeleton color') ;

  % Changing the prediction color must not disturb the label color.
  newPredColor = [0 1 0] ;
  controller.pbSkeletonColorPred_.BackgroundColor = newPredColor ;
  controller.applyActuated_() ;

  % saveState_() forwards only the landmark set types whose properties actually
  % changed, so editing the prediction color alone must send exactly one entry.
  % With the old shared swatch both would have changed, so this is the assertion
  % that pins down the independence.
  assert(~isempty(lastSkeletonSpecs), 'Applying did not reach setLandmarkAndSkeletonCosmetics()') ;
  assert(isscalar(lastSkeletonSpecs), ...
         'Editing the prediction color should change the prediction skeleton only, got %d entries', ...
         numel(lastSkeletonSpecs)) ;
  assert(lastSkeletonSpecs.landmarkSetType == LandmarkSetType.Prediction, ...
         'The changed skeleton entry is not the Prediction landmark set type') ;
  assert(isequal(lastSkeletonSpecs.SkeletonProps.Color, newPredColor), ...
         'Prediction skeleton color did not reach skeletonSpecs') ;

  % The full widget state should still describe both types: the label color
  % untouched, the prediction color updated, and the line width shared.
  allSpecs = controller.skelControlsGet_() ;
  assert(numel(allSpecs) == numel(enumeration('LandmarkSetType')), ...
         'skelControlsGet_ should return one entry per landmark set type') ;
  labelEntry = allSpecs(1) ;
  predEntry = allSpecs(2) ;
  assert(labelEntry.landmarkSetType == LandmarkSetType.Label, ...
         'First skeletonSpecs entry is not the Label landmark set type') ;
  assert(predEntry.landmarkSetType == LandmarkSetType.Prediction, ...
         'Second skeletonSpecs entry is not the Prediction landmark set type') ;
  assert(isequal(labelEntry.SkeletonProps.Color, labelColor), ...
         'Label skeleton color changed when only the prediction color was edited') ;
  assert(isequal(predEntry.SkeletonProps.Color, newPredColor), ...
         'Prediction skeleton color is not reflected in the widget state') ;
  assert(isequal(labelEntry.SkeletonProps.LineWidth, predEntry.SkeletonProps.LineWidth), ...
         'Skeleton line width should be shared between labels and predictions') ;

  function recordApply(skeletonSpecs)
    % Capture what the controller hands to the Labeler.
    lastSkeletonSpecs = skeletonSpecs ;
  end  % nested function
end  % function

function labeler = makeMockLabeler_(labelColor, predColor)
  % Minimal struct graph supplying what LandmarkColorsController reads, with
  % different skeleton colors for the label and prediction landmark set types.
  npts = 4 ;
  labelPpi = makeMockPointsPlotInfo_(npts, labelColor) ;
  predPpi = makeMockPointsPlotInfo_(npts, predColor) ;
  trx = struct('TrajColor', [1 1 0], 'TrajLineWidth', 1, 'TrxIDLblFontSize', 15) ;
  labeler = struct('nPhysPoints', npts, ...
                   'labelPointsPlotInfo', labelPpi, ...
                   'predPointsPlotInfo', predPpi, ...
                   'hasTrx', false, ...
                   'maIsMA', false, ...
                   'projPrefs', struct('Trx', trx)) ;
end  % function

function ppi = makeMockPointsPlotInfo_(npts, skeletonColor)
  % Fields consumed: ColorMapName, Colors (for LandmarkColorSpec); MarkerProps,
  % TextProps, TextOffset, SkeletonProps (for the marker/skeleton controls).
  markerProps = struct('Marker', '.', 'MarkerSize', 20, 'LineWidth', 2) ;
  textProps = struct('Visible', 'on', 'FontSize', 12, 'FontAngle', 'normal') ;
  skeletonProps = struct('Color', skeletonColor, 'LineWidth', 2) ;
  ppi = struct('ColorMapName', 'jet', ...
               'Colors', jet(npts), ...
               'MarkerProps', markerProps, ...
               'TextProps', textProps, ...
               'TextOffset', 10, ...
               'SkeletonProps', skeletonProps) ;
end  % function
