function test_landmark_colors_gui_smoke()
  % Headless smoke test for the de-GUIDE'd Landmark Cosmetics dialog.
  %
  % LandmarkColors used to be a GUIDE app (LandmarkColors.m + .fig) that stored
  % its state in guidata and blocked on uiwait.  It is now the subcontroller
  % class LandmarkColorsController, which builds its figure programmatically,
  % keeps state/widget handles as instance properties, and is modal but
  % non-blocking (edits applied live via the apply callback).  This test
  % constructs the controller with a lightweight mock Labeler, asserts the
  % figure opens with the expected controls by Tag, and exercises the
  % apply-to-model path once.

  applyCallCount = 0 ;

  labeler = makeMockLabeler_() ;
  labeler.setLandmarkAndSkeletonCosmetics = @(varargin)(recordApply()) ;
  parent = struct('mainFigurePixelPosition', [100 100 800 600]) ;
  controller = LandmarkColorsController(parent, labeler) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  assert(~isempty(controller.hFig) && isgraphics(controller.hFig, 'figure'), ...
         'LandmarkColorsController did not create a valid figure') ;
  assert(strcmp(controller.hFig.Tag, 'figure_landmarkcolors'), ...
         'Dialog figure has unexpected Tag "%s"', controller.hFig.Tag) ;

  % The layout is fixed-pixel with no reflow logic, so the dialog must not be
  % resizable (a resize would strand widgets at their original coordinates).
  assert(strcmp(controller.hFig.Resize, 'off'), ...
         'Dialog figure should not be resizable (Resize="%s")', controller.hFig.Resize) ;

  % Every widget the controller drives should be a live graphics handle parented
  % (directly or indirectly) to the dialog figure.  We check the instance
  % properties rather than findobj-by-Tag because imagesc() legitimately resets
  % the colormap axes' Tag (NextPlot 'replace'); the controller uses the stored
  % handle, not the Tag.
  widgetProps = {'pbDone_', 'uibuttongroup1_', 'radiobutton_colormap_', 'radiobutton_manual_', ...
                 'uipanel_manual_', 'uipanel_colormap_', 'axes_colormap_', 'popupmenu_colormap_', ...
                 'slider_brightness_', 'edit_brightness_', 'pumShowing_', 'cbApplyAll_', ...
                 'pnlMarkersLabels_', 'tblProps_', 'uipanel6_', 'sldSkeletonLineWidth_', 'pbSkeletonColor_'} ;
  for i = 1 : numel(widgetProps)
    prop = widgetProps{i} ;
    h = controller.(prop) ;
    assert(~isempty(h) && isgraphics(h), 'Widget "%s" is not a valid graphics handle', prop) ;
    assert(ancestor(h, 'figure') == controller.hFig, 'Widget "%s" is not parented to the dialog figure', prop) ;
  end

  % A per-landmark manual-color button should have been created for each point.
  assert(numel(controller.hbuttons_) == labeler.nPhysPoints, ...
         'Expected %d manual-color buttons, found %d', labeler.nPhysPoints, numel(controller.hbuttons_)) ;

  % Exercise the apply-to-model path: it should read the table/skeleton state
  % and invoke the apply callback without error.
  controller.applyActuated_() ;
  assert(applyCallCount == 1, 'Apply callback was not invoked exactly once (count=%d)', applyCallCount) ;

  function recordApply()
    applyCallCount = applyCallCount + 1 ;
  end  % nested function
end  % function

function labeler = makeMockLabeler_()
  % Build a minimal struct graph supplying what LandmarkColorsController reads:
  % a nPhysPoints count and label/pred points-plot-info structs shaped like the
  % real labelPointsPlotInfo/predPointsPlotInfo.
  npts = 4 ;
  ppi = makeMockPointsPlotInfo_(npts) ;
  labeler = struct('nPhysPoints', npts, ...
                   'labelPointsPlotInfo', ppi, ...
                   'predPointsPlotInfo', ppi, ...
                   'hasTrx', false, ...
                   'maIsMA', false) ;
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
