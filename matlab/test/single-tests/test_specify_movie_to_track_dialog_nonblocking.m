function test_specify_movie_to_track_dialog_nonblocking()
% Test that the movie-details dialog (SpecifyMovieToTrackGUI) is a
% non-blocking modal subcontroller.
%
% Constructing the dialog should return once it is up (no uiwait, no run()
% method), the dialog figure should be modal, and its delegate (here the
% LabelerController) should hold it as a subcontroller
% (specifyMovieToTrackController_).  Cancelling / closing the dialog should
% call deleteSpecifyMovieToTrackController() on the delegate, tearing the
% subcontroller down and deleting the figure.

linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleaner = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>

% Build a minimal movie-data struct for the current movie.  The dialog
% fills in the remaining fields itself.
mIdx = labeler.currMovIdx ;
movdata = struct() ;
movdata.movfiles = labeler.getMovieFilesAllFullMovIdx(mIdx) ;

% Constructing the dialog is expected to return promptly (no uiwait
% blocking).  If it blocked, the test would hang here rather than proceed.
% The LabelerController is the delegate and holds the dialog, exactly as it
% does when the dialog is opened from the Track > Current Movie menu.
controller.specifyMovieToTrackController_ = ...
  SpecifyMovieToTrackGUI(labeler, controller, movdata) ;
subcontroller = controller.specifyMovieToTrackController_ ;
assert(isa(subcontroller, 'SpecifyMovieToTrackGUI') && isvalid(subcontroller), ...
       'The movie-details dialog was not stored as a subcontroller') ;

% The dialog figure should exist and be modal.
hFig = findall(0, 'Type', 'figure', 'Tag', 'figure_SpecifyMovieToTrack') ;
assert(isscalar(hFig), 'The movie-details dialog figure never appeared') ;
assert(strcmp(get(hFig, 'WindowStyle'), 'modal'), ...
       'The movie-details dialog is not modal') ;

% Cancelling the dialog should tear down the subcontroller and delete the
% figure.  Drive it exactly as the Cancel button callback would.
cancelButton = findall(hFig, 'Tag', 'controlbutton_cancel') ;
assert(isscalar(cancelButton), 'Could not find the Cancel button') ;
feval(cancelButton.Callback, cancelButton, []) ;
drawnow ;

assert(isempty(controller.specifyMovieToTrackController_), ...
       'Cancelling the dialog did not clear the subcontroller') ;
assert(isempty(findall(0, 'Type', 'figure', 'Tag', 'figure_SpecifyMovieToTrack')), ...
       'Cancelling the dialog did not delete the figure') ;

end  % function
