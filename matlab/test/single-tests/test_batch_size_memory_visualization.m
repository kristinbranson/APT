function test_batch_size_memory_visualization()
% Test that selecting the "batch size" parameter in the training-parameters
% dialog brings up the memory visualization: the visualization pane and its
% axes become visible, the axes contains a plotted line, and the axis labels
% are as expected.

linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;
% Skip auto-computation of parameters: it is slow and can raise dialogs
labeler.trackAutoSetParams = false ;

% The batch-size parameter carries the memory visualization
% (ParameterVisualizationMemory).  The FQN is relative to ROOT, matching how
% propertiesGUI2 names its Java property objects.
batchSizeFqn = 'DeepTrack.GradientDescent.batch_size' ;

% State written by the timer callback while the dialog is up
didDriveDialog = false ;
dialogError = [] ;
vizResult = [] ;

% The dialog blocks in uiwait(), so a timer drives it: once the dialog
% appears, the callback selects the batch-size property (as a mouse click
% would), captures the visualization state, and presses Cancel, which
% unblocks the menu actuation below.
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
assert(~isempty(vizResult), 'The visualization state was never captured') ;

% Selecting the batch-size parameter should have exposed the visualization
% pane: the dialog figure is widened to reveal the (always-'on') pnlViz.
assert(strcmp(vizResult.pnlVizVisible, 'on'), ...
       'The visualization panel is not visible') ;
assert(vizResult.figWidthAfter > vizResult.figWidthBefore, ...
       ['Selecting batch size did not widen the dialog to expose the ' ...
        'visualization pane (width %g -> %g)'], ...
       vizResult.figWidthBefore, vizResult.figWidthAfter) ;

% The visualization axes should be visible and contain a plotted line
assert(strcmp(vizResult.axVisible, 'on'), ...
       'The visualization axes is not visible') ;
assert(vizResult.lineCount >= 1, ...
       'The visualization axes contains no line object') ;

% The memory visualization plots memory required against batch size
assert(strcmp(vizResult.xLabel, 'Batch size'), ...
       'x-axis label is "%s", expected "Batch size"', vizResult.xLabel) ;
assert(strcmp(vizResult.yLabel, 'Memory required (GB)'), ...
       'y-axis label is "%s", expected "Memory required (GB)"', vizResult.yLabel) ;

fprintf('test_batch_size_memory_visualization passed.\n') ;

  function driveDialogBang(~, ~)
    % Timer callback: select batch size in the open dialog and capture the
    % resulting visualization state.
    hFig = findall(0, 'Type', 'figure', 'Name', 'Training parameters') ;
    if isempty(hFig)
      return
    end
    stop(timerObj) ;
    didDriveDialog = true ;
    try
      handles = guidata(hFig) ;
      pvh = getappdata(hFig, 'parameterVizHandler') ;
      assert(~isempty(pvh), 'No parameterVizHandler in the dialog appdata') ;

      % Figure width before selecting: at init the pane is toggled hidden
      hFig.Units = 'pixels' ;
      figWidthBefore = hFig.Position(3) ;

      % Find the batch-size Java property and select it, as the table's
      % mouse-pressed callback would.
      batchNamePath = displayNamePathForFqn(handles.tree, batchSizeFqn) ;
      batchProp = findJavaPropertyByNamePath(getappdata(hFig, 'propsList'), ...
                                             batchNamePath) ;
      assert(~isempty(batchProp), ...
             'No table property with FQN %s', batchSizeFqn) ;
      pvh.propSelected(batchProp, now) ;
      drawnow ;

      % Capture (as plain values) the state the main body will assert on
      axViz = handles.axViz ;
      hFig.Units = 'pixels' ;
      result = struct() ;
      result.figWidthBefore = figWidthBefore ;
      result.figWidthAfter = hFig.Position(3) ;
      result.pnlVizVisible = handles.pnlViz.Visible ;
      result.axVisible = axViz.Visible ;
      result.lineCount = numel(findobj(axViz, 'Type', 'line')) ;
      result.xLabel = get(get(axViz, 'XLabel'), 'String') ;
      result.yLabel = get(get(axViz, 'YLabel'), 'String') ;
      vizResult = result ;
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
