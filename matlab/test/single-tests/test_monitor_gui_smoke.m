function test_monitor_gui_smoke()
  % Headless smoke test for the de-GUIDE'd training/tracking monitor GUIs.
  %
  % TrainMonitorController and TrackMonitorViz used to build their figures by loading
  % a GUIDE .fig via a gui_mainfcn dispatcher (TrainMonitorGUI.m /
  % TrackMonitorGUI.m).  They now build their figures programmatically in a
  % createGui_() method and keep the widget handles as instance properties
  % (no more guidata reach-throughs).  This test constructs each Viz with a
  % lightweight mock Labeler -- enough to exercise the real constructor and the
  % figure build -- and asserts the figure opens and exposes the expected
  % controls by Tag.  It runs headless under xvfb on any supported release.

  test_one_monitor_(@TrainMonitorController, ...
                    'figure_TrainMonitor', ...
                    {'axes_loss', 'axes_dist', 'text_clusterinfo', 'text_clusterstatus', ...
                     'popupmenu_actions', 'pushbutton_action', 'pushbutton_startstop'}) ;

  test_one_monitor_(@TrackMonitorViz, ...
                    'figure_TrackMonitor', ...
                    {'axes_wait', 'edit_trackerinfo', 'text_clusterinfo', 'text_clusterstatus', ...
                     'popupmenu_actions', 'pushbutton_action', 'pushbutton_startstop'}) ;

  fprintf('test_monitor_gui_smoke: PASSED\n') ;
end  % function

function test_one_monitor_(vizConstructor, figureTag, expectedTags)
  % Construct one monitor Viz with a mock Labeler and check its figure/controls.
  labeler = makeMockLabeler_() ;
  parent = [] ;  % a LabelerController; only captured in the CloseRequestFcn closure, never called here
  viz = vizConstructor(parent, labeler) ;
  cleaner = onCleanup(@()(delete(viz))) ;

  assert(~isempty(viz.hfig) && isgraphics(viz.hfig, 'figure'), ...
         'Monitor Viz did not create a valid figure') ;
  assert(strcmp(viz.hfig.Tag, figureTag), ...
         'Monitor figure has Tag "%s", expected "%s"', viz.hfig.Tag, figureTag) ;

  for i = 1 : numel(expectedTags)
    tag = expectedTags{i} ;
    h = findobj(viz.hfig, 'Tag', tag) ;
    assert(~isempty(h), 'Expected control with Tag "%s" not found in %s', tag, figureTag) ;
  end
end  % function

function labeler = makeMockLabeler_()
  % Build a minimal struct graph that supplies exactly what the monitor Viz
  % constructors read.  Uses real enums where the constructors compare against
  % them.
  dmc = struct('getStages', 1, 'getViews', 1, 'getSplits', 0, 'n', 1) ;
  poller = struct('trackStyle_', apt.TrackStyle.movie, 'resultSize', [1 1 1]) ;
  tracker = struct('trnLastDMC', dmc, ...
                   'bgTrainPoller', struct(), ...
                   'bgTrackPoller', poller, ...
                   'nFramesToTrack', 100, ...
                   'getTrackerInfoString', 'Mock tracker info') ;
  backend = struct('type', DLBackEnd.Conda) ;
  labeler = struct('tracker', tracker, ...
                   'backend', backend, ...
                   'nview', 1, ...
                   'bgTrnIsRunning', false, ...
                   'lastTrainEndCause', EndCause.complete, ...
                   'bgTrkIsRunning', false, ...
                   'lastTrackEndCause', EndCause.complete) ;
end  % function
