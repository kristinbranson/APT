function test_track_batch_dialog_nonblocking()
% Test that the batch-tracking dialog is a non-blocking modal subcontroller.
%
% The menu actuation should return once the dialog is up (no uiwait), the
% dialog figure should be modal, and the LabelerController should hold the
% dialog as a subcontroller (trackBatchGUIController_).  Dismissing the
% dialog (Cancel / close) should tear the subcontroller down and delete
% the figure.  Opening it a second time should replace, not accumulate,
% the dialog.

linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;  %#ok<ASGLU>
cleaner = onCleanup(@()(delete(controller))) ;

% The menu actuation is expected to return promptly (no uiwait blocking).
% If it blocked, the test would hang here rather than proceed.
controller.menu_track_batch_track_actuated_([], []) ;

% The dialog should now be up and held as a subcontroller.
subcontroller = controller.trackBatchGUIController_ ;
assert(isa(subcontroller, 'TrackBatchGUIController') && isvalid(subcontroller), ...
       'The batch-tracking dialog was not stored as a subcontroller') ;

% The dialog figure should exist and be modal.
hFig = findall(0, 'Type', 'figure', 'Tag', 'figure_SelectTrackBatch') ;
assert(isscalar(hFig), 'The batch-tracking dialog figure never appeared') ;
assert(strcmp(get(hFig, 'WindowStyle'), 'modal'), ...
       'The batch-tracking dialog is not modal') ;

% Opening the dialog again should replace it, not accumulate a second one.
controller.menu_track_batch_track_actuated_([], []) ;
hFigAgain = findall(0, 'Type', 'figure', 'Tag', 'figure_SelectTrackBatch') ;
assert(isscalar(hFigAgain), ...
       'Reopening the batch-tracking dialog accumulated multiple figures') ;
assert(~isvalid(subcontroller), ...
       'Reopening the dialog did not delete the previous subcontroller') ;

% Cancelling the dialog should tear down the subcontroller and delete the
% figure.  Drive it exactly as the Cancel button callback would.
cancelButton = findall(hFigAgain, 'Tag', 'controlbutton_cancel') ;
assert(isscalar(cancelButton), 'Could not find the Cancel button') ;
feval(cancelButton.ButtonPushedFcn, cancelButton, []) ;
drawnow ;

assert(isempty(controller.trackBatchGUIController_), ...
       'Cancelling the dialog did not clear the subcontroller') ;
assert(isempty(findall(0, 'Type', 'figure', 'Tag', 'figure_SelectTrackBatch')), ...
       'Cancelling the dialog did not delete the figure') ;

end  % function
