function test_autotune_no_duplicate_align_entry()
% Test that the "Align using tail->head body axis" parameter
% (ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta) appears exactly once
% in the Auto-tune tab of the training-parameters dialog for a bottom-up
% multi-animal project.  It used to be rendered twice: once as a
% hardcoded special-case suggestion (value true) and again in the
% generic loop over auto-computed parameters (value false).

linuxProjectFilePath = ...
  '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;

% Make the current tracker a bottom-up multi-animal tracker (a single MA
% pose net, no detection stage), which is the case that used to render
% AlignUsingTrxTheta twice.
maposenets = Labeler.getAllTrackerTypes() ;
labeler.trackMakeNewTrackerGivenNetTypes(maposenets(1)) ;
assert(labeler.maIsMA && ~labeler.trackerIsTwoStage, ...
       'Test precondition failed: expected a bottom-up MA project') ;

% The dialog is modal, but the menu actuation returns once it is up, so
% the test can drive it directly.
controller.menu_track_setparametersfile_actuated_([], []) ;
hFig = findall(0, 'Type', 'figure', 'Name', 'Training Parameters') ;
assert(isscalar(hFig), 'The training-parameters dialog never appeared') ;
dialogCleanupObj = onCleanup(@()(deleteIfValid(hFig))) ;  %#ok<NASGU>

% Locate the Auto-tune tab and count the AlignUsingTrxTheta value
% controls in it.  Controls in the Auto-tune tab are tagged with the
% bare parameter field name.
autoTab = findall(hFig, 'Title', 'Auto-tune') ;
assert(isscalar(autoTab), 'Could not find the Auto-tune tab') ;
alignControls = findall(autoTab, 'Tag', 'AlignUsingTrxTheta') ;
assert(numel(alignControls) == 1, ...
       'AlignUsingTrxTheta appears %d times in the Auto-tune tab, expected exactly once', ...
       numel(alignControls)) ;

% Dismiss the dialog
cancelButton = findall(hFig, 'Tag', 'pb_cancel') ;
assert(isscalar(cancelButton), 'No Cancel button in the dialog') ;
feval(cancelButton.ButtonPushedFcn, cancelButton, []) ;
drawnow ;

end  % function


function deleteIfValid(hFig)
% Delete a figure, tolerating an already-deleted one.
if isvalid(hFig)
  delete(hFig) ;
end
end  % function
