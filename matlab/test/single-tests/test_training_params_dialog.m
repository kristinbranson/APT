function test_training_params_dialog()
% Test that the training-parameters dialog (Track > Configure tracking
% parameters) opens from the menu with populated parameter controls,
% that some commonly-used training parameters appear in it, that
% raising the user level in the dropdown never decreases the number of
% parameter controls, and that editing a parameter and pressing Apply
% propagates the new value to the Labeler.

linuxProjectFilePath = ...
  '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;

% Parameters to spot-check: training parameters (AffectsTraining ==
% true) that should be visible in the dialog at the highest level.
spotCheckFieldPaths = { 'ROOT.DeepTrack.GradientDescent.dl_steps', ...
                        'ROOT.DeepTrack.GradientDescent.batch_size', ...
                        'ROOT.DeepTrack.DataAugmentation.rrange' } ;

% The parameter to edit through the dialog, and the value to set it to.
editFqn = 'ROOT.DeepTrack.GradientDescent.dl_steps' ;
sPrmBefore = labeler.trackGetTrainingParams() ;
oldEditValue = sPrmBefore.ROOT.DeepTrack.GradientDescent.dl_steps ;
newEditValue = oldEditValue + 1234 ;

% State written by the timer callback while the dialog is up
didDriveDialog = false ;
dialogError = [] ;
controlCountFromSweepIndex = [] ;

% The dialog blocks in uiwait(), so a timer drives it: once the dialog
% appears, the callback sweeps the level dropdown, inspects the
% parameter controls, edits one, and presses Apply, which unblocks the
% menu actuation below.
timerObj = timer('StartDelay', 2, ...
                 'Period', 1, ...
                 'ExecutionMode', 'fixedSpacing', ...
                 'TasksToExecute', 60, ...
                 'TimerFcn', @driveDialogBang) ;
timerCleanupObj = onCleanup(@()(stopAndDeleteTimer(timerObj))) ;  %#ok<NASGU>
start(timerObj) ;

controller.menu_track_setparametersfile_actuated_([], []) ;  % blocks until dialog dismissed

assert(didDriveDialog, ...
       'The training-parameters dialog never appeared') ;
if ~isempty(dialogError)
  rethrow(dialogError) ;
end

% The dialog should have had a healthy number of parameter controls even
% at the most basic level
fprintf('%s: shown control counts by level: %s\n', ...
        mfilename(), mat2str(controlCountFromSweepIndex)) ;
assert(controlCountFromSweepIndex(1) > 10, ...
       'Expected more than 10 parameter controls at the most basic level, found %d', ...
       controlCountFromSweepIndex(1)) ;

% Raising the level should never decrease the number of controls, and
% the highest level should show strictly more than the lowest
assert(all(diff(controlCountFromSweepIndex) >= 0), ...
       'Raising the level decreased the parameter control count: %s', ...
       mat2str(controlCountFromSweepIndex)) ;
assert(controlCountFromSweepIndex(end) > controlCountFromSweepIndex(1), ...
       'The highest level should show more parameter controls than the lowest') ;

% The edit made in the dialog, applied with the Apply button, should
% have propagated to the model
sPrmAfter = labeler.trackGetTrainingParams() ;
editValueFromLabeler = sPrmAfter.ROOT.DeepTrack.GradientDescent.dl_steps ;
assert(isequal(double(editValueFromLabeler), double(newEditValue)), ...
       'dl_steps is %g after Apply, expected %g', ...
       double(editValueFromLabeler), double(newEditValue)) ;

fprintf('test_training_params_dialog passed.\n') ;

  function driveDialogBang(~, ~)
    % Timer callback: drive the open training-parameters dialog.
    hFig = findall(0, 'Type', 'figure', 'Name', 'Training parameters') ;
    if isempty(hFig)
      return
    end
    stop(timerObj) ;
    didDriveDialog = true ;
    didApply = false ;
    try
      % Sweep the level dropdown from the lowest level to the highest,
      % recording the parameter-control count at each level
      levelDropdown = findall(hFig, 'Tag', 'popupmenu_level') ;
      assert(isscalar(levelDropdown), 'No level dropdown in the dialog') ;
      levelNames = levelDropdown.Items ;
      levels = PropertyLevelsEnum(levelNames) ;
      [~, sweepOrder] = sort(double(levels)) ;
      sweepCount = numel(sweepOrder) ;
      controlCountFromSweepIndex = nan(1, sweepCount) ;
      for sweepIndex = 1 : sweepCount
        levelDropdown.Value = levelNames{sweepOrder(sweepIndex)} ;
        feval(levelDropdown.ValueChangedFcn, levelDropdown, []) ;
        drawnow ;
        controlCountFromSweepIndex(sweepIndex) = countParameterControls(hFig) ;
      end

      % At the highest level, spot-check that commonly-used training
      % parameters appear as controls in the dialog
      for i = 1 : numel(spotCheckFieldPaths)
        fieldPath = spotCheckFieldPaths{i} ;
        control = findParameterControlByFqn(hFig, fieldPath) ;
        assert(~isempty(control), ...
               'No parameter control for %s', fieldPath) ;
      end

      % Edit one parameter through its control, as a user edit would:
      % setting the value and firing the change callback writes the new
      % value into the dialog's parameter tree, which Apply reads from
      editControl = findParameterControlByFqn(hFig, editFqn) ;
      assert(~isempty(editControl), 'No parameter control for %s', editFqn) ;
      editControl.Value = newEditValue ;
      feval(editControl.ValueChangedFcn, editControl, []) ;
      drawnow ;

      % Apply: closes the dialog and hands the edited parameters to the
      % blocked menu actuation, which writes them to the Labeler
      applyButton = findall(hFig, 'Tag', 'pb_apply') ;
      assert(isscalar(applyButton), 'No Apply button in the dialog') ;
      feval(applyButton.ButtonPushedFcn, applyButton, []) ;
      didApply = true ;
    catch err
      dialogError = err ;
    end
    % On an error path the dialog may still be up: dismiss it so the
    % blocked menu actuation can return.
    if ~didApply && isvalid(hFig)
      try
        cancelButton = findall(hFig, 'Tag', 'pb_cancel') ;
        feval(cancelButton.ButtonPushedFcn, cancelButton, []) ;
      catch
        delete(hFig) ;
      end
    end
  end  % function

end  % function


function count = countParameterControls(hFig)
% Count the per-parameter value controls in the dialog.  Value controls
% carry a ValueChangedFcn and a tag derived from their tree path; the
% level dropdown and the Viz toggle buttons are excluded.
controls = findall(hFig, '-property', 'ValueChangedFcn') ;
isParamControl = arrayfun(@(c)(~isempty(c.Tag) && ...
                               ~startsWith(c.Tag, 'tb_viz_') && ...
                               ~strcmp(c.Tag, 'popupmenu_level')), ...
                          controls) ;
count = sum(isParamControl) ;
end  % function


function control = findParameterControlByFqn(hFig, fqn)
% Find the value control for a parameter, given its fully-qualified name
% (e.g. 'ROOT.DeepTrack.GradientDescent.dl_steps').  Control tags are
% underscore-joined tree paths, so match on the underscore-joined FQN
% suffix (excluding the Viz toggle buttons, which share the suffix).
% Returns [] if not found; if the parameter is shown on several tabs,
% returns one of its (synchronized) controls.
suffix = strrep(strrep(fqn, 'ROOT.', ''), '.', '_') ;
pattern = ['(^|_)', regexptranslate('escape', suffix), '$'] ;
candidates = findall(hFig, '-regexp', 'Tag', pattern) ;
isVizButton = arrayfun(@(c)(startsWith(c.Tag, 'tb_viz_')), candidates) ;
candidates = candidates(~isVizButton) ;
if isempty(candidates)
  control = [] ;
else
  control = candidates(1) ;
end
end  % function


function stopAndDeleteTimer(timerObj)
% Stop and delete a timer, tolerating an already-deleted one.
if isvalid(timerObj)
  stop(timerObj) ;
  delete(timerObj) ;
end
end  % function
