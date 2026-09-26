function test_keyboard_frame_navigation()
  % The keyboard shortcuts must be wired: LabelerController registers cbkKPF
  % as the KeyPressFcn of the main figure and of every control that has the
  % property, and a right-arrow key press through that handler advances the
  % current frame (left-arrow steps it back).
  %
  % Regression test for the Python port, where the graphics layer declared
  % KeyPressFcn on nothing and dispatched no key events: the registration
  % loop (findall '-property' 'KeyPressFcn') found nothing, so none of the
  % shortcuts worked.  The key event is synthesized as a struct and handed to
  % the registered handler, which is what the window system does on a key.

  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;
  drawnow() ;

  mainFigure = controller.mainFigure_ ;
  figureHandler = get(mainFigure, 'KeyPressFcn') ;
  assert(isa(figureHandler, 'function_handle'), ...
         'The main figure has no KeyPressFcn registered') ;

  % Every control that has the property got the same handler, so a shortcut
  % works whichever control has focus.  (The frame edit box is left out on
  % purpose, so typing a frame number is not intercepted.)
  ownerHandles = findall(mainFigure, '-property', 'KeyPressFcn') ;
  registeredCount = 0 ;
  for i = 1 : numel(ownerHandles)
    if isa(get(ownerHandles(i), 'KeyPressFcn'), 'function_handle')
      registeredCount = registeredCount + 1 ;
    end
  end
  assert(registeredCount > 1, ...
         'Only %d object(s) carry the key handler; the controls should too', registeredCount) ;

  frameBefore = labeler.currFrame ;
  keyEvent = struct('Key', 'rightarrow', 'Character', '') ;
  keyEvent.Modifier = {} ;
  figureHandler(mainFigure, keyEvent) ;
  assert(labeler.currFrame == frameBefore + 1, ...
         'Right arrow did not advance the frame: %d -> %d', frameBefore, labeler.currFrame) ;
  keyEvent.Key = 'leftarrow' ;
  figureHandler(mainFigure, keyEvent) ;
  assert(labeler.currFrame == frameBefore, ...
         'Left arrow did not step the frame back: now %d', labeler.currFrame) ;
end  % function
