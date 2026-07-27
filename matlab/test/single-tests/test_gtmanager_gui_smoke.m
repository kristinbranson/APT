function test_gtmanager_gui_smoke()
  % Headless smoke test for the de-guidata'd "Groundtruth Navigator" window.
  %
  % GTManager.m (a bare uifigure function with state in guidata and Labeler
  % listeners) is now the subcontroller class GTManagerController.  This test
  % constructs it against a bare (project-less) Labeler -- a real Labeler is
  % needed because the controller adds listeners on Labeler events -- with a
  % mock parent whose main figure is empty, so updateAll_ takes its early-out
  % and does no heavy model work.  It asserts the window opens with the
  % expected widgets and that the Labeler listeners were installed.

  labeler = Labeler('isgui', false) ;
  labelerCleaner = onCleanup(@()(delete(labeler))) ;  %#ok<NASGU>
  parent = struct('mainFigure_', gobjects(1,0)) ;  % empty main figure -> updateAll_ early-outs
  controller = GTManagerController(parent, labeler) ;
  cleaner = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>

  assert(~isempty(controller.hFig) && isgraphics(controller.hFig, 'figure'), ...
         'GTManagerController did not create a valid figure') ;
  assert(strcmp(controller.hFig.Name, 'Groundtruth Navigator'), ...
         'Window has unexpected Name "%s"', controller.hFig.Name) ;

  widgetProps = {'tblGTMovie_', 'tblFrame_', 'pbNextUnlabeled_', 'pbGoSelected_', ...
                 'pbComputeGT_', 'pbUpdate_'} ;
  for i = 1 : numel(widgetProps)
    prop = widgetProps{i} ;
    h = controller.(prop) ;
    assert(~isempty(h) && isgraphics(h), 'Widget "%s" is not a valid graphics handle', prop) ;
  end

  assert(numel(controller.listener_) == 6, ...
         'Expected 6 Labeler listeners, found %d', numel(controller.listener_)) ;

  fprintf('test_gtmanager_gui_smoke: PASSED\n') ;
end  % function
