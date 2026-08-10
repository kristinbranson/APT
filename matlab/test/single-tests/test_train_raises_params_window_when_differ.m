function test_train_raises_params_window_when_differ()
% Test that starting training, when auto-set is on and the auto-computed
% parameters differ from the current ones by more than 10%, first raises a
% heads-up dialog (rather than training immediately): cancelling it aborts
% training, while continuing opens the Training Parameters window, and
% cancelling that window aborts training too.

linuxProjectFilePath = ...
  '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;

% Use a local backend so the tracker is fit to be trained (the fixture's
% saved backend is a long-gone AWS instance).  Training is never actually
% spawned by this test, which cancels out of the parameters window.
backend = docker_unless_janelia_cluster_then_conda() ;
labeler.set_backend_property('type', backend) ;

% Make the current tracker a bottom-up multi-animal tracker.
maposenets = Labeler.getAllTrackerTypes() ;
labeler.trackMakeNewTrackerGivenNetTypes(maposenets(1)) ;
assert(labeler.maIsMA && ~labeler.trackerIsTwoStage, ...
       'Test precondition failed: expected a bottom-up MA project') ;

% Auto-set on, and force the current parameters to differ from the
% auto-computed ones by more than 10% (a tiny animal-box radius will be far
% from the auto-computed one).  Clear the unsaved-changes flag so the train
% path does not raise a save prompt.
labeler.trackAutoSetParams = true ;
sPrm = labeler.trackGetTrainingParams() ;
sPrm.ROOT.MultiAnimal.TargetCrop.ManualRadius = 1 ;
labeler.trackSetTrainingParams(sPrm) ;
labeler.doesNeedSave_ = false ;

% Sanity-check that the auto-params really do differ, so the window is
% expected to be raised.
assert(doAutoParamsDifferFromCurrent(labeler), ...
       'Test precondition failed: auto-params do not differ from current') ;

% Start training as the Train button would.  With auto-set on and the params
% differing, this should raise the (non-blocking) heads-up dialog and return
% without training or opening the Training Parameters window yet.
controller.pbTrain_actuated_([], []) ;

headsUpFig = findall(0, 'Type', 'figure', 'Name', 'Auto-Tune Parameters Differ') ;
assert(isscalar(headsUpFig), 'The auto-params-differ heads-up dialog was not raised') ;
assert(isempty(findall(0, 'Type', 'figure', 'Name', 'Training Parameters')), ...
       'The Training Parameters window opened before the user acknowledged the heads-up') ;
assert(~labeler.bgTrnIsRunning, ...
       'Training started even though the heads-up dialog is still up') ;

% Cancelling the heads-up must abort training and open no window.
cancelHeadsUp = findall(headsUpFig, 'Tag', 'pb_cancel') ;
assert(isscalar(cancelHeadsUp), 'No Cancel button in the heads-up dialog') ;
feval(cancelHeadsUp.ButtonPushedFcn, cancelHeadsUp, []) ;
drawnow ;
assert(isempty(findall(0, 'Type', 'figure', 'Name', 'Auto-Tune Parameters Differ')), ...
       'The heads-up dialog was not dismissed on Cancel') ;
assert(isempty(findall(0, 'Type', 'figure', 'Name', 'Training Parameters')), ...
       'The Training Parameters window opened after the heads-up was cancelled') ;
assert(~labeler.bgTrnIsRunning, 'Training started after the heads-up was cancelled') ;

% Start training again and this time continue through the heads-up, which
% should open the Training Parameters window.
controller.pbTrain_actuated_([], []) ;
headsUpFig = findall(0, 'Type', 'figure', 'Name', 'Auto-Tune Parameters Differ') ;
assert(isscalar(headsUpFig), 'The heads-up dialog was not raised on the second attempt') ;
continueHeadsUp = findall(headsUpFig, 'Tag', 'pb_continue') ;
assert(isscalar(continueHeadsUp), 'No Continue button in the heads-up dialog') ;
feval(continueHeadsUp.ButtonPushedFcn, continueHeadsUp, []) ;
drawnow ;

hFig = findall(0, 'Type', 'figure', 'Name', 'Training Parameters') ;
assert(isscalar(hFig), 'The Training Parameters window was not opened after Continue') ;
dialogCleanupObj = onCleanup(@()(deleteIfValid(hFig))) ;  %#ok<NASGU>
assert(~labeler.bgTrnIsRunning, ...
       'Training started even though the parameters window is still up') ;

% Cancel the window: this must abort training (no continuation is called).
cancelButton = findall(hFig, 'Tag', 'pb_cancel') ;
assert(isscalar(cancelButton), 'No Cancel button in the parameters window') ;
feval(cancelButton.ButtonPushedFcn, cancelButton, []) ;
drawnow ;

assert(isempty(controller.parameterSetupModalController_), ...
       'The parameters window controller was not cleared after Cancel') ;
assert(~labeler.bgTrnIsRunning, ...
       'Training started after the parameters window was cancelled') ;

end  % function


function deleteIfValid(hFig)
% Delete a figure, tolerating an already-deleted one.
if isvalid(hFig)
  delete(hFig) ;
end
end  % function
