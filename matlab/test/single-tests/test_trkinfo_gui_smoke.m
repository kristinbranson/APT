function test_trkinfo_gui_smoke()
  % Headless smoke test for the de-guidata'd "Track Info" window.
  %
  % TrkInfoUI.m (a bare uifigure function with state in guidata) is now the
  % subcontroller class TrkInfoController.  This test constructs it with a
  % lightweight mock Labeler that has no tracker (so updateMovie_ takes its
  % no-tracking-results path) and asserts the window opens with the expected
  % widgets.

  labeler = makeMockLabeler_() ;
  parent = struct('tvTrkPred_', []) ;  % only used by navigation callbacks, not fired here
  controller = TrkInfoController(parent, labeler) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  assert(~isempty(controller.hFig) && isgraphics(controller.hFig, 'figure'), ...
         'TrkInfoController did not create a valid figure') ;
  assert(strcmp(controller.hFig.Name, 'Track Info'), ...
         'Window has unexpected Name "%s"', controller.hFig.Name) ;

  widgetProps = {'mov_tbl_', 'tbl_', 'sf_btn_', 'ef_btn_', 'prev_btn_', 'next_btn_'} ;
  for i = 1 : numel(widgetProps)
    prop = widgetProps{i} ;
    h = controller.(prop) ;
    assert(~isempty(h) && isgraphics(h), 'Widget "%s" is not a valid graphics handle', prop) ;
  end

  % With no tracker, updateMovie_ should leave the summary table empty.
  assert(~controller.has_data_, 'Expected has_data_ false with no tracker') ;

  fprintf('test_trkinfo_gui_smoke: PASSED\n') ;
end  % function

function labeler = makeMockLabeler_()
  % Minimal struct graph supplying what TrkInfoController reads at construction:
  % maIsMA, currMovie, movieFilesAllFullGTaware, and an (empty) tracker.
  labeler = struct('maIsMA', true, ...
                   'currMovie', 1, ...
                   'movieFilesAllFullGTaware', {{'movie1.avi' ; 'movie2.avi'}}, ...
                   'tracker', []) ;
end  % function
