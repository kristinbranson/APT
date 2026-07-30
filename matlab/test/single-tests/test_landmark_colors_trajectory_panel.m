function test_landmark_colors_trajectory_panel()
  % Headless test for the Trajectory pane of the Landmark Cosmetics dialog.
  %
  % LandmarkColorsController grows a "Trajectory" panel (color, line width, ID
  % label font size) for projects that draw trajectories (trx or MA).  This
  % test constructs the controller with a lightweight mock Labeler, asserts the
  % pane and its widgets appear for an MA project, that editing them sends a
  % trajSpecs struct (with the expected fields/values) to
  % setLandmarkAndSkeletonCosmetics(), and that the pane is absent for a project
  % with no trajectories.

  % ---------------------------------------------------------------------------
  % MA project: the Trajectory pane should be present.
  % ---------------------------------------------------------------------------
  lastTrajSpecs = [] ;

  labeler = makeMockLabeler_(true) ;  % maIsMA = true
  labeler.setLandmarkAndSkeletonCosmetics = ...
    @(colorSpecs, markerSpecs, skeletonSpecs, trajSpecs)(recordApply(trajSpecs)) ;
  parent = struct('mainFigurePixelPosition', [100 100 800 600]) ;
  controller = LandmarkColorsController(parent, labeler) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  assert(controller.tfTrajControlsShown_, ...
         'Trajectory pane was not shown for an MA project') ;

  widgetProps = {'pnlTraj_', 'pbTrajColor_', 'sldTrajLineWidth_', 'txTrajLineWidth_', 'editTrajFontSize_'} ;
  for i = 1 : numel(widgetProps)
    prop = widgetProps{i} ;
    h = controller.(prop) ;
    assert(~isempty(h) && isgraphics(h), 'Trajectory widget "%s" is not a valid graphics handle', prop) ;
    assert(ancestor(h, 'figure') == controller.hFig, 'Trajectory widget "%s" is not parented to the dialog figure', prop) ;
  end

  % The controls should reflect the project's initial trajectory prefs.
  assert(isequal(controller.pbTrajColor_.BackgroundColor, [1 1 0]), ...
         'Trajectory color button did not reflect the initial TrajColor') ;
  assert(abs(2^controller.sldTrajLineWidth_.Value - 1) < 1e-6, ...
         'Trajectory line-width slider did not reflect the initial TrajLineWidth') ;
  assert(str2double(controller.editTrajFontSize_.String) == 15, ...
         'Trajectory font-size edit did not reflect the initial TrxIDLblFontSize') ;

  % Edit each control, then apply.  applyActuated_ only forwards trajSpecs when
  % it differs from the open-time baseline, so changing the controls should
  % produce a non-empty trajSpecs carrying the new values.
  controller.pbTrajColor_.BackgroundColor = [0 1 0] ;
  controller.sldTrajLineWidth_.Value = log2(4) ;   % -> line width 4.0
  controller.editTrajFontSize_.String = '20' ;
  controller.applyActuated_() ;

  assert(~isempty(lastTrajSpecs), ...
         'Editing the trajectory controls did not forward a trajSpecs struct') ;
  assert(isequal(lastTrajSpecs.TrajColor, [0 1 0]), ...
         'trajSpecs.TrajColor did not match the edited color') ;
  assert(abs(lastTrajSpecs.TrajLineWidth - 4) < 1e-6, ...
         'trajSpecs.TrajLineWidth (%.3f) did not match the edited width', lastTrajSpecs.TrajLineWidth) ;
  assert(lastTrajSpecs.TrxIDLblFontSize == 20, ...
         'trajSpecs.TrxIDLblFontSize (%g) did not match the edited font size', lastTrajSpecs.TrxIDLblFontSize) ;

  % The line-width readout text should track the slider.
  assert(strcmp(strtrim(controller.txTrajLineWidth_.String), '4.0'), ...
         'Trajectory line-width readout "%s" did not track the slider', controller.txTrajLineWidth_.String) ;

  % ---------------------------------------------------------------------------
  % Non-trajectory project: the Trajectory pane should be absent.
  % ---------------------------------------------------------------------------
  labelerNoTraj = makeMockLabeler_(false) ;  % maIsMA = false, hasTrx = false
  labelerNoTraj.setLandmarkAndSkeletonCosmetics = ...
    @(colorSpecs, markerSpecs, skeletonSpecs, trajSpecs)(recordApply(trajSpecs)) ;
  controllerNoTraj = LandmarkColorsController(parent, labelerNoTraj) ;
  cleaner2 = onCleanup(@()(delete(controllerNoTraj))) ;

  assert(~controllerNoTraj.tfTrajControlsShown_, ...
         'Trajectory pane was shown for a project with no trajectories') ;
  assert(isempty(controllerNoTraj.pnlTraj_), ...
         'Trajectory panel handle should be empty for a project with no trajectories') ;

  function recordApply(trajSpecs)
    lastTrajSpecs = trajSpecs ;
  end  % nested function
end  % function

function labeler = makeMockLabeler_(isMA)
  % Minimal struct graph supplying what LandmarkColorsController reads: the
  % keypoint colors/markers/skeleton plot info (as in the colors smoke test),
  % plus hasTrx/maIsMA and the projPrefs.Trx trajectory prefs that drive the
  % Trajectory pane.
  npts = 4 ;
  ppi = makeMockPointsPlotInfo_(npts) ;
  trx = struct('TrajColor', [1 1 0], 'TrajLineWidth', 1, 'TrxIDLblFontSize', 15) ;
  labeler = struct('nPhysPoints', npts, ...
                   'labelPointsPlotInfo', ppi, ...
                   'predPointsPlotInfo', ppi, ...
                   'hasTrx', false, ...
                   'maIsMA', isMA, ...
                   'projPrefs', struct('Trx', trx)) ;
end  % function

function ppi = makeMockPointsPlotInfo_(npts)
  % Fields consumed: ColorMapName, Colors (for LandmarkColorSpec); MarkerProps,
  % TextProps, TextOffset, SkeletonProps (for the marker/skeleton controls).
  markerProps = struct('Marker', '.', 'MarkerSize', 20, 'LineWidth', 2) ;
  textProps = struct('Visible', 'on', 'FontSize', 12, 'FontAngle', 'normal') ;
  skeletonProps = struct('Color', [1 1 1], 'LineWidth', 2) ;
  ppi = struct('ColorMapName', 'jet', ...
               'Colors', jet(npts), ...
               'MarkerProps', markerProps, ...
               'TextProps', textProps, ...
               'TextOffset', 10, ...
               'SkeletonProps', skeletonProps) ;
end  % function
