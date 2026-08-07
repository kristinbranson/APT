function test_train_raises_params_window_when_differ()
% Test that starting training raises the Training Parameters window (rather
% than training immediately) when auto-set is on and the auto-computed
% parameters differ from the current ones by more than 10%, and that
% cancelling that window aborts training.

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

% Start training as the Train button would.  With auto-set on and the
% params differing, this should raise the (non-blocking) Training Parameters
% window and return without training.
controller.pbTrain_actuated_([], []) ;

hFig = findall(0, 'Type', 'figure', 'Name', 'Training Parameters') ;
assert(isscalar(hFig), 'The Training Parameters window was not raised at training start') ;
dialogCleanupObj = onCleanup(@()(deleteIfValid(hFig))) ;  %#ok<NASGU>

% Training must not have started yet: the window is up awaiting the user.
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
