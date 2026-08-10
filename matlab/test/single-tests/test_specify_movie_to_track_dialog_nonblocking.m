function test_specify_movie_to_track_dialog_nonblocking()
% Test that the movie-details dialog (SpecifyMovieToTrackController) is a
% non-blocking modal subcontroller.
%
% Constructing the dialog should return once it is up (no uiwait, no run()
% method), the dialog figure should be modal, and its delegate should hold
% it as a subcontroller (specifyMovieToTrackController_).  The dialog should
% be centered on the delegate controller's figure, using
% mainFigurePixelPosition() for a LabelerController delegate and
% figurePixelPosition() for a TrackBatchGUIController delegate.  Cancelling /
% closing the dialog should call deleteSpecifyMovieToTrackController() on the
% delegate, tearing the subcontroller down and deleting the figure.
%
% Both delegate types are exercised below.

linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleaner = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>

% Build a minimal movie-data struct for the current movie.  The dialog fills
% in the remaining fields itself.
mIdx = labeler.currMovIdx ;
movdata = struct() ;
movdata.movfiles = labeler.getMovieFilesAllFullMovIdx(mIdx) ;

% ---------------------------------------------------------------------------
% Delegate 1: LabelerController (as opened from Track > Current Movie).
% ---------------------------------------------------------------------------

% Constructing the dialog is expected to return promptly (no uiwait
% blocking).  If it blocked, the test would hang here rather than proceed.
% The LabelerController is the delegate and holds the dialog.
controller.specifyMovieToTrackController_ = ...
  SpecifyMovieToTrackController(labeler, controller, movdata) ;
subcontroller = controller.specifyMovieToTrackController_ ;
assert(isa(subcontroller, 'SpecifyMovieToTrackController') && isvalid(subcontroller), ...
       'The movie-details dialog was not stored as a subcontroller') ;

% The dialog figure should exist, be modal, and be centered on the main
% figure.
hFig = findall(0, 'Type', 'figure', 'Tag', 'figure_SpecifyMovieToTrack') ;
assert(isscalar(hFig), 'The movie-details dialog figure never appeared') ;
assert(strcmp(get(hFig, 'WindowStyle'), 'modal'), ...
       'The movie-details dialog is not modal') ;
assertFigureCenteredOn(hFig, controller.mainFigurePixelPosition(), ...
       'The dialog is not centered on the main figure') ;

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

% ---------------------------------------------------------------------------
% Delegate 2: TrackBatchGUIController (as opened from the batch dialog's
% Add / details buttons).
% ---------------------------------------------------------------------------

% Open the batch dialog; it becomes the delegate for the nested dialog.
controller.menu_track_batch_track_actuated_([], []) ;
trackBatchController = controller.trackBatchGUIController_ ;
assert(isa(trackBatchController, 'TrackBatchGUIController') && isvalid(trackBatchController), ...
       'The batch-tracking dialog was not created') ;

trackBatchController.specifyMovieToTrackController_ = ...
  SpecifyMovieToTrackController(labeler, trackBatchController, movdata) ;
subcontroller = trackBatchController.specifyMovieToTrackController_ ;
assert(isa(subcontroller, 'SpecifyMovieToTrackController') && isvalid(subcontroller), ...
       'The movie-details dialog was not stored as a subcontroller of the batch dialog') ;

hFig = findall(0, 'Type', 'figure', 'Tag', 'figure_SpecifyMovieToTrack') ;
assert(isscalar(hFig), 'The movie-details dialog figure never appeared (batch delegate)') ;
assert(strcmp(get(hFig, 'WindowStyle'), 'modal'), ...
       'The movie-details dialog is not modal (batch delegate)') ;
assertFigureCenteredOn(hFig, trackBatchController.figurePixelPosition(), ...
       'The dialog is not centered on the batch dialog figure') ;

% Tearing down via the delegate should clear the subcontroller and delete
% the figure.
trackBatchController.deleteSpecifyMovieToTrackController() ;
drawnow ;
assert(isempty(trackBatchController.specifyMovieToTrackController_), ...
       'Teardown did not clear the subcontroller (batch delegate)') ;
assert(isempty(findall(0, 'Type', 'figure', 'Tag', 'figure_SpecifyMovieToTrack')), ...
       'Teardown did not delete the figure (batch delegate)') ;

end  % function

function assertFigureCenteredOn(hFig, delegatePosition, message)
% Assert that hFig's center coincides (to within a pixel or so) with the
% center of the [x y w h] pixel rectangle delegatePosition.
oldUnits = hFig.Units ;
hFig.Units = 'pixels' ;
figurePosition = hFig.Position ;
hFig.Units = oldUnits ;
delegateCenter = delegatePosition(1:2) + delegatePosition(3:4)/2 ;
figureCenter = figurePosition(1:2) + figurePosition(3:4)/2 ;
assert(max(abs(delegateCenter - figureCenter)) < 2, message) ;
end  % function
