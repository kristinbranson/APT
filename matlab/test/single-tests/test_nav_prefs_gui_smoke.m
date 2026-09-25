function test_nav_prefs_gui_smoke()
  % Headless smoke test for the de-GUIDE'd Navigation Preferences dialog.
  %
  % NavPrefsNew used to be a GUIDE app (NavPrefsNew.m + NavPrefsNewMat.m) that
  % stored its state in guidata / UserData and blocked on uiwait.  It is now
  % the subcontroller class NavPrefsController, which builds its figure
  % programmatically, keeps widget handles as instance properties, and is
  % modal but non-blocking.  This test constructs the controller with a
  % lightweight mock Labeler, asserts the figure opens with the expected
  % controls populated from the model, exercises the edit-field validation
  % and the seek-mode enable logic, checks that the model write reaches the
  % Labeler, and finally that Apply closes the dialog.

  threshCalls = [] ;

  labeler = makeMockLabeler_() ;
  labeler.setMovieShiftArrowNavModeThresh = @(v)(recordThresh(v)) ;
  parent = struct('mainFigurePixelPosition', [100 100 800 600]) ;
  controller = NavPrefsController(parent, labeler) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  assert(~isempty(controller.hFig) && isgraphics(controller.hFig, 'figure'), ...
         'NavPrefsController did not create a valid figure') ;
  assert(strcmp(controller.hFig.Tag, 'figure_navprefs'), ...
         'Dialog figure has unexpected Tag "%s"', controller.hFig.Tag) ;
  assert(strcmp(controller.hFig.Resize, 'off'), ...
         'Dialog figure should not be resizable (Resize="%s")', controller.hFig.Resize) ;

  % Every widget the controller drives should be a live graphics handle
  % parented to the dialog figure.
  widgetProps = {'etFrameSkip_', 'etPlaybackSpeed_', 'etPlaybackLoopDiameter_', ...
                 'pumShiftArrow_', 'txTimelineProp_', 'pumShiftArrowTimelineThreshCmp_', ...
                 'etShiftArrowTimelineThresh_', 'pbApply_', 'pbCancel_'} ;
  for i = 1 : numel(widgetProps)
    prop = widgetProps{i} ;
    h = controller.(prop) ;
    assert(~isempty(h) && isgraphics(h), 'Widget "%s" is not a valid graphics handle', prop) ;
    assert(ancestor(h, 'figure') == controller.hFig, 'Widget "%s" is not parented to the dialog figure', prop) ;
  end

  % The controls are populated from the model.
  assert(strcmp(controller.etFrameSkip_.String, '10'), 'Frame skip not populated from model') ;
  assert(strcmp(controller.etPlaybackSpeed_.String, '30'), 'Playback speed not populated from model') ;
  assert(strcmp(controller.etPlaybackLoopDiameter_.String, '50'), 'Loop radius not populated from model') ;
  assert(controller.selectedShiftArrowMode_() == ShiftArrowMovieNavMode.NEXTLABELED, ...
         'Seek mode popup not populated from model') ;
  assert(strcmp(controller.etShiftArrowTimelineThresh_.String, '0.25'), 'Threshold not populated from model') ;
  cmpPum = controller.pumShiftArrowTimelineThreshCmp_ ;
  assert(strcmp(cmpPum.String{cmpPum.Value}, '>='), 'Comparison popup not populated from model') ;
  assert(strcmp(controller.txTimelineProp_.String, 'dx_body_mean_abs'), ...
         'Timeline property name not populated from model') ;

  % The threshold row is disabled for a non-threshold seek mode, and enabled
  % once NEXTTIMELINETHRESH is picked.
  assert(strcmp(controller.etShiftArrowTimelineThresh_.Enable, 'off'), ...
         'Threshold row should be disabled for the NEXTLABELED seek mode') ;
  assert(strcmp(controller.pumShiftArrowTimelineThreshCmp_.Enable, 'off')) ;
  assert(strcmp(controller.txTimelineProp_.Enable, 'off')) ;
  modes = controller.shiftArrowModes_ ;
  controller.pumShiftArrow_.Value = find(modes == ShiftArrowMovieNavMode.NEXTTIMELINETHRESH) ;
  controller.pumShiftArrowActuated_() ;
  assert(strcmp(controller.etShiftArrowTimelineThresh_.Enable, 'on'), ...
         'Threshold row should be enabled for the NEXTTIMELINETHRESH seek mode') ;
  assert(strcmp(controller.pumShiftArrowTimelineThreshCmp_.Enable, 'on')) ;
  assert(strcmp(controller.txTimelineProp_.Enable, 'on')) ;

  % An invalid entry reverts to the model value.
  controller.etFrameSkip_.String = 'abc' ;
  controller.etFrameSkipActuated_() ;
  assert(strcmp(controller.etFrameSkip_.String, '10'), 'Non-numeric frame skip should revert') ;
  controller.etPlaybackSpeed_.String = '-5' ;
  controller.etPlaybackSpeedActuated_() ;
  assert(strcmp(controller.etPlaybackSpeed_.String, '30'), 'Non-positive playback speed should revert') ;
  controller.etPlaybackLoopDiameter_.String = '0' ;
  controller.etPlaybackLoopDiameterActuated_() ;
  assert(strcmp(controller.etPlaybackLoopDiameter_.String, '50'), 'Non-positive loop radius should revert') ;
  controller.etShiftArrowTimelineThresh_.String = 'x' ;
  controller.etShiftArrowTimelineThreshActuated_() ;
  assert(strcmp(controller.etShiftArrowTimelineThresh_.String, '0.25'), 'Non-numeric threshold should revert') ;

  % The model write carries every edit to the Labeler.  (The mock is a struct
  % held in controller.labeler_, so the writes are read back from there.)
  controller.etFrameSkip_.String = '25' ;
  controller.etPlaybackSpeed_.String = '12' ;
  controller.etPlaybackLoopDiameter_.String = '7' ;
  controller.etShiftArrowTimelineThresh_.String = '0.5' ;
  controller.pumShiftArrowTimelineThreshCmp_.Value = 2 ;  % '<'
  controller.writePreferencesToModel_() ;
  written = controller.labeler_ ;
  assert(written.movieFrameStepBig == 25, 'movieFrameStepBig not written') ;
  assert(written.moviePlayFPS == 12, 'moviePlayFPS not written') ;
  assert(written.moviePlaySegRadius == 7, 'moviePlaySegRadius not written') ;
  assert(written.movieShiftArrowNavMode == ShiftArrowMovieNavMode.NEXTTIMELINETHRESH, ...
         'movieShiftArrowNavMode not written') ;
  assert(strcmp(written.movieShiftArrowNavModeThreshCmp, '<'), 'movieShiftArrowNavModeThreshCmp not written') ;
  assert(isequal(threshCalls, 0.5), 'setMovieShiftArrowNavModeThresh not called exactly once with 0.5') ;

  % Apply closes the dialog.
  hFig = controller.hFig ;
  controller.applyActuated_() ;
  assert(~isvalid(controller), 'Apply should delete the controller') ;
  assert(~isgraphics(hFig), 'Apply should close the dialog figure') ;

  function recordThresh(v)
    threshCalls(end+1) = v ;
  end  % nested function
end  % function

function labeler = makeMockLabeler_()
  % Minimal struct graph supplying what NavPrefsController reads: the six
  % navigation-preference properties and an infoTimelineModel whose
  % getCurPropSmart() names the current timeline statistic.
  infoTimelineModel = struct('getCurPropSmart', @()(deal('Labels', struct('name', 'dx_body_mean_abs')))) ;
  labeler = struct('movieFrameStepBig', 10, ...
                   'moviePlayFPS', 30, ...
                   'moviePlaySegRadius', 50, ...
                   'movieShiftArrowNavMode', ShiftArrowMovieNavMode.NEXTLABELED, ...
                   'movieShiftArrowNavModeThresh', 0.25, ...
                   'movieShiftArrowNavModeThreshCmp', '>=', ...
                   'infoTimelineModel', infoTimelineModel) ;
end  % function
