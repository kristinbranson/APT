function test_training_params_dialog()
% Test that the training-parameters dialog (Track > Configure tracking
% parameters) opens from the menu with a populated table, that some
% commonly-used training parameters appear in it, and that raising the
% user level in the dropdown never decreases the number of table rows.

linuxProjectFilePath = ...
  '/groups/branson/bransonlab/apt/unittest/alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019_updated_20250306.lbl' ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;
% Skip auto-computation of parameters: it is slow and can raise dialogs
labeler.trackAutoSetParams = false ;

% Parameters to spot-check.  These are training parameters both on main
% and on the training-parameters2 branch (AffectsTraining == true there).
% Display names are resolved from the tree at runtime, since they differ
% between the two versions.
spotCheckFieldPaths = { 'ROOT.DeepTrack.GradientDescent.dl_steps', ...
                        'ROOT.DeepTrack.GradientDescent.batch_size', ...
                        'ROOT.DeepTrack.DataAugmentation.rrange' } ;

% State written by the timer callback while the dialog is up
didDriveDialog = false ;
dialogError = [] ;
rowCountFromSweepIndex = [] ;

% The dialog blocks in uiwait(), so a timer drives it: once the dialog
% appears, the callback sweeps the level dropdown, inspects the table,
% and presses Cancel, which unblocks the menu actuation below.
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

% The dialog should have had a healthy number of rows even at the most
% basic level
assert(rowCountFromSweepIndex(1) > 10, ...
       'Expected more than 10 rows at the most basic level, found %d', ...
       rowCountFromSweepIndex(1)) ;

% Raising the level should never decrease the number of rows, and the
% highest level should show strictly more than the lowest
assert(all(diff(rowCountFromSweepIndex) >= 0), ...
       'Raising the level decreased the table row count: %s', ...
       mat2str(rowCountFromSweepIndex)) ;
assert(rowCountFromSweepIndex(end) > rowCountFromSweepIndex(1), ...
       'The highest level should show more rows than the lowest') ;

fprintf('test_training_params_dialog passed.\n') ;

  function driveDialogBang(~, ~)
    % Timer callback: drive the open training-parameters dialog.
    hFig = findall(0, 'Type', 'figure', 'Name', 'Training parameters') ;
    if isempty(hFig)
      return
    end
    stop(timerObj) ;
    didDriveDialog = true ;
    try
      handles = guidata(hFig) ;

      % Sweep the level dropdown from the lowest level to the highest,
      % recording the table row count at each level
      levelNames = cellstr(get(handles.popupmenu_level, 'String')) ;
      levels = PropertyLevelsEnum(levelNames) ;
      [~, sweepOrder] = sort(double(levels)) ;
      sweepCount = numel(sweepOrder) ;
      rowCountFromSweepIndex = nan(1, sweepCount) ;
      for sweepIndex = 1 : sweepCount
        set(handles.popupmenu_level, 'Value', sweepOrder(sweepIndex)) ;
        ParameterSetup('popupmenu_level_Callback', ...
                       handles.popupmenu_level, [], guidata(hFig)) ;
        drawnow ;
        % Count all non-hidden properties in the (rebuilt) table model:
        % the on-screen row count depends on category expansion state
        propsList = getappdata(hFig, 'propsList') ;
        rowCountFromSweepIndex(sweepIndex) = ...
          numel(collectShownPropertyNames(propsList)) ;
      end

      % At the highest level, spot-check that commonly-used training
      % parameters appear as rows in the table, under the display name
      % the tree gives them
      shownRowNames = collectShownPropertyNames(getappdata(hFig, 'propsList')) ;
      for i = 1 : numel(spotCheckFieldPaths)
        fieldPath = spotCheckFieldPaths{i} ;
        node = handles.tree.findnode(fieldPath) ;
        assert(~isempty(node), 'Parameter %s is missing from the tree', fieldPath) ;
        assert(node.Data.Visible, ...
               'Parameter %s is not visible at the highest level', fieldPath) ;
        expectedRowName = node.Data.DispNameUse ;
        assert(ismember(expectedRowName, shownRowNames), ...
               'No table row named "%s" (for %s)', expectedRowName, fieldPath) ;
      end
    catch err
      dialogError = err ;
    end
    % Dismiss the dialog so the blocked menu actuation can return
    if isvalid(hFig)
      try
        handles = guidata(hFig) ;
        ParameterSetup('pbCancel_Callback', handles.pbCancel, [], guidata(hFig)) ;
      catch
        delete(hFig) ;
      end
    end
  end  % function

end  % function


function names = collectShownPropertyNames(javaPropertyList)
% Recursively collect the display names of all non-hidden properties in a
% java.util.List of com.jidesoft.grid.DefaultProperty objects.
names = cell(0, 1) ;
if isempty(javaPropertyList)
  return
end
for i = 0 : (javaPropertyList.size() - 1)
  prop = javaPropertyList.get(i) ;
  if prop.isHidden()
    continue
  end
  names{end+1, 1} = char(prop.getName()) ;  %#ok<AGROW>
  children = prop.getChildren() ;
  if ~isempty(children) && children.size() > 0
    childNames = collectShownPropertyNames(children) ;
    names = [names ; childNames] ;  %#ok<AGROW>
  end
end
end  % function


function stopAndDeleteTimer(timerObj)
% Stop and delete a timer, tolerating an already-deleted one.
if isvalid(timerObj)
  stop(timerObj) ;
  delete(timerObj) ;
end
end  % function
