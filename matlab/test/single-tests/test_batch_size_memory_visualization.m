function test_batch_size_memory_visualization()
% Test that toggling the visualization button next to the "batch size"
% parameter in the training-parameters dialog brings up the memory
% visualization: the visualization pane gains an axes containing a
% plotted line, with the expected axis labels.

linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
labeler.isInBatchMode = true ;

% The batch-size parameter carries the memory visualization
% (ParameterVisualizationMemory), toggled by the Viz button next to its
% value control.
batchSizeFqn = 'ROOT.DeepTrack.GradientDescent.batch_size' ;

% State written by the timer callback while the dialog is up
didDriveDialog = false ;
dialogError = [] ;
vizResult = [] ;

% The dialog blocks in uiwait(), so a timer drives it: once the dialog
% appears, the callback toggles the batch-size Viz button (as a mouse
% click would), captures the visualization state, and presses Cancel,
% which unblocks the menu actuation below.
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

% Toggling the batch-size Viz button should have populated the
% visualization pane with an axes containing a plotted line
assert(vizResult.axesCount >= 1, ...
       'The visualization pane contains no axes') ;
assert(vizResult.lineCount >= 1, ...
       'The visualization axes contains no line object') ;

% The memory visualization plots memory required against batch size
assert(strcmp(vizResult.xLabel, 'Batch size'), ...
       'x-axis label is "%s", expected "Batch size"', vizResult.xLabel) ;
assert(strcmp(vizResult.yLabel, 'Memory required (GB)'), ...
       'y-axis label is "%s", expected "Memory required (GB)"', vizResult.yLabel) ;

fprintf('test_batch_size_memory_visualization passed.\n') ;

  function driveDialogBang(~, ~)
    % Timer callback: toggle the batch-size Viz button in the open dialog
    % and capture the resulting visualization state.
    hFig = findall(0, 'Type', 'figure', 'Name', 'Training parameters') ;
    if isempty(hFig)
      return
    end
    stop(timerObj) ;
    didDriveDialog = true ;
    try
      % The visualization pane starts out empty
      vizPanel = findall(hFig, 'Tag', 'panel_right') ;
      assert(isscalar(vizPanel), 'No visualization panel in the dialog') ;
      lineCountBefore = numel(findall(vizPanel, 'Type', 'line')) ;
      assert(lineCountBefore == 0, ...
             'The visualization pane already contains a plot') ;

      % Find the batch-size Viz toggle button and toggle it on, as a
      % mouse click would
      suffix = strrep(strrep(batchSizeFqn, 'ROOT.', ''), '.', '_') ;
      pattern = ['^tb_viz_.*', regexptranslate('escape', suffix), '$'] ;
      vizButton = findall(hFig, '-regexp', 'Tag', pattern) ;
      assert(~isempty(vizButton), ...
             'No Viz button for %s', batchSizeFqn) ;
      vizButton = vizButton(1) ;
      vizButton.Value = 1 ;
      feval(vizButton.ValueChangedFcn, vizButton, []) ;
      drawnow ;

      % Capture (as plain values) the state the main body will assert on
      vizAxes = findall(vizPanel, 'Type', 'axes') ;
      result = struct() ;
      result.axesCount = numel(vizAxes) ;
      if isempty(vizAxes)
        result.lineCount = 0 ;
        result.xLabel = '' ;
        result.yLabel = '' ;
      else
        result.lineCount = numel(findall(vizAxes(1), 'Type', 'line')) ;
        result.xLabel = get(get(vizAxes(1), 'XLabel'), 'String') ;
        result.yLabel = get(get(vizAxes(1), 'YLabel'), 'String') ;
      end
      vizResult = result ;
    catch err
      dialogError = err ;
    end
    % Dismiss the dialog so the blocked menu actuation can return
    if isvalid(hFig)
      try
        cancelButton = findall(hFig, 'Tag', 'pb_cancel') ;
        feval(cancelButton.ButtonPushedFcn, cancelButton, []) ;
      catch
        delete(hFig) ;
      end
    end
  end  % function

end  % function


function stopAndDeleteTimer(timerObj)
% Stop and delete a timer, tolerating an already-deleted one.
if isvalid(timerObj)
  stop(timerObj) ;
  delete(timerObj) ;
end
end  % function
