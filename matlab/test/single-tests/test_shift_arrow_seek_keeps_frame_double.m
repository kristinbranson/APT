function test_shift_arrow_seek_keeps_frame_double()
  % A shift-arrow press seeks to the next labeled frame, which the Labels
  % struct stores as uint32.  The frame index the Labeler ends up holding
  % must be a double all the same: as a uint32, (currFrame-1)/(nframes-1)
  % is integer division and rounds to 0, so the frame slider stays at the
  % left end, and currFrame - r saturates at 0, so the timeline's left
  % limit sticks at 0.  Found by comparing the Python port with MATLAB side
  % by side: the port never left double.

  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;
  drawnow() ;

  mainFigure = controller.mainFigure_ ;
  figureHandler = get(mainFigure, 'KeyPressFcn') ;
  keyEvent = struct('Key', 'rightarrow', 'Character', '') ;
  keyEvent.Modifier = {'shift'} ;
  frameBefore = labeler.currFrame ;
  figureHandler(mainFigure, keyEvent) ;
  drawnow() ;
  assert(labeler.currFrame ~= frameBefore, ...
         'Shift-right arrow did not seek to another frame') ;
  assert(isa(labeler.currFrame, 'double'), ...
         'currFrame is a %s after a shift-arrow seek; it must stay double', class(labeler.currFrame)) ;

  expectedSliderValue = (labeler.currFrame - 1) / (labeler.nframes - 1) ;
  sliderValue = get(controller.slider_frame, 'Value') ;
  assert(abs(sliderValue - expectedSliderValue) < 1e-9, ...
         'The frame slider did not follow the seek: Value %g, expected %g', sliderValue, expectedSliderValue) ;
end  % function
