function test_tracking_params_dialog()
% Test that the tracking-parameters dialog (Track > Configure tracking
% parameters for this run) opens from the menu with a populated table,
% that some commonly-used tracking parameters appear in it, that
% raising the user level in the dropdown never decreases the number of
% table rows, and that editing a parameter in the table and pressing
% Apply propagates the new value to the Labeler.

linuxProjectFilePath = ...
  '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;

% Parameters to spot-check.  These are tracking parameters both on main
% (they live in the ROOT.Track subtree) and on the training-parameters2
% branch (AffectsTraining == false there).  Display names are resolved
% from the tree at runtime.
spotCheckFieldPaths = { 'ROOT.Track.NFramesSmall', ...
                        'ROOT.Track.NFramesLarge', ...
                        'ROOT.Track.NFramesNeighborhood' } ;

% The parameter to edit through the table, and the value to set it to.
% The FQN is relative to ROOT, matching the tracking-parameter tree that
% the dialog shows (which is also how propertiesGUI2 addresses its
% mirror tree).
editFqn = 'Track.NFramesNeighborhood' ;
sPrmBefore = labeler.trackGetTrainingParams() ;
oldEditValue = sPrmBefore.ROOT.Track.NFramesNeighborhood ;
newEditValue = oldEditValue + 37 ;

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

controller.menu_track_settrackparams_actuated_([], []) ;  % blocks until dialog dismissed

assert(didDriveDialog, ...
       'The tracking-parameters dialog never appeared') ;
if ~isempty(dialogError)
  rethrow(dialogError) ;
end

% The dialog should show the whole tracking tree at the highest level.
% For an MA project that includes the MultiAnimal.Track subtree as well
% as the Track category with its three NFrames* leaves.
% (PostProcess's reconcile3dType requires a multiview project.)
fprintf('%s: shown property counts by level: %s\n', ...
        mfilename(), mat2str(rowCountFromSweepIndex)) ;
assert(rowCountFromSweepIndex(end) > 10, ...
       'Expected more than 10 rows at the highest level, found %d', ...
       rowCountFromSweepIndex(end)) ;

% Raising the level should never decrease the number of rows, and the
% highest level should show strictly more than the lowest
assert(all(diff(rowCountFromSweepIndex) >= 0), ...
       'Raising the level decreased the table row count: %s', ...
       mat2str(rowCountFromSweepIndex)) ;
assert(rowCountFromSweepIndex(end) > rowCountFromSweepIndex(1), ...
       'The highest level should show more rows than the lowest') ;

% The edit made in the dialog, applied with the Apply button, should
% have propagated to the model
sPrmAfter = labeler.trackGetTrainingParams() ;
editValueFromLabeler = sPrmAfter.ROOT.Track.NFramesNeighborhood ;
assert(isequal(double(editValueFromLabeler), double(newEditValue)), ...
       'NFramesNeighborhood is %g after Apply, expected %g', ...
       double(editValueFromLabeler), double(newEditValue)) ;

fprintf('test_tracking_params_dialog passed.\n') ;

  function driveDialogBang(~, ~)
    % Timer callback: drive the open tracking-parameters dialog.
    hFig = findall(0, 'Type', 'figure', 'Name', 'Tracking parameters') ;
    if isempty(hFig)
      return
    end
    stop(timerObj) ;
    didDriveDialog = true ;
    didApply = false ;
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

      % At the highest level, spot-check that commonly-used tracking
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

      % Edit one parameter through the Java table model, as a user edit
      % would: setting the value fires the property-change callback,
      % which updates the mirror tree that Apply reads from
      editNamePath = displayNamePathForFqn(handles.tree, editFqn) ;
      editProp = findJavaPropertyByNamePath(getappdata(hFig, 'propsList'), ...
                                            editNamePath) ;
      assert(~isempty(editProp), 'No table property with FQN %s', editFqn) ;
      editProp.setValue(int32(newEditValue)) ;
      didMirrorUpdate = false ;
      for pollIndex = 1 : 50
        drawnow ;
        mirror = getappdata(hFig, 'mirror') ;
        if isequal(double(mirror.getValue(editFqn)), double(newEditValue))
          didMirrorUpdate = true ;
          break
        end
        pause(0.1) ;
      end
      assert(didMirrorUpdate, ...
             'The mirror tree did not pick up the edit to %s', editFqn) ;

      % Apply: closes the dialog and hands the edited parameters to the
      % blocked menu actuation, which writes them to the Labeler
      ParameterSetup('pbApply_Callback', handles.pbApply, [], guidata(hFig)) ;
      didApply = true ;
    catch err
      dialogError = err ;
    end
    % On an error path the dialog may still be up: dismiss it so the
    % blocked menu actuation can return.  (After Apply the figure is
    % still valid here -- its deletion is deferred to the OutputFcn --
    % so a second dismissal would wrongly delete it mid-uiwait.)
    if ~didApply && isvalid(hFig)
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


function namePath = displayNamePathForFqn(tree, fqn)
% Convert a ROOT-relative FQN into the corresponding path of display
% names, which is how propertiesGUI2 names its Java property objects.
fieldNames = strsplit(fqn, '.') ;
namePath = cell(1, numel(fieldNames)) ;
partialFqn = 'ROOT' ;
for i = 1 : numel(fieldNames)
  partialFqn = [partialFqn '.' fieldNames{i}] ;  %#ok<AGROW>
  node = tree.findnode(partialFqn) ;
  assert(~isempty(node), 'No tree node %s', partialFqn) ;
  namePath{i} = node.Data.DispNameUse ;
end
end  % function


function prop = findJavaPropertyByNamePath(javaPropertyList, namePath)
% Recursively find the Java property whose display-name path matches
% namePath (a cellstr).  Returns [] if not found.
prop = [] ;
if isempty(javaPropertyList) || isempty(namePath)
  return
end
for i = 0 : (javaPropertyList.size() - 1)
  candidate = javaPropertyList.get(i) ;
  if strcmp(char(candidate.getName()), namePath{1})
    if isscalar(namePath)
      prop = candidate ;
      return
    end
    children = candidate.getChildren() ;
    if ~isempty(children) && children.size() > 0
      prop = findJavaPropertyByNamePath(children, namePath(2:end)) ;
      if ~isempty(prop)
        return
      end
    end
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
