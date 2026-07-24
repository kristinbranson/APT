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

% The dialog is modal, but the menu actuation returns once it is up, so
% the test can drive it directly.
controller.menu_track_setparametersfile_actuated_([], []) ;
hFig = findall(0, 'Type', 'figure', 'Name', 'Training parameters') ;
assert(isscalar(hFig), 'The training-parameters dialog never appeared') ;
dialogCleanupObj = onCleanup(@()(deleteIfValid(hFig))) ;  %#ok<NASGU>

% The visualization pane starts out empty
vizPanel = findall(hFig, 'Tag', 'panel_right') ;
assert(isscalar(vizPanel), 'No visualization panel in the dialog') ;
lineCountBefore = numel(findall(vizPanel, 'Type', 'line')) ;
assert(lineCountBefore == 0, ...
       'The visualization pane already contains a plot') ;

% Find the batch-size Viz toggle button and toggle it on, as a mouse
% click would
suffix = strrep(strrep(batchSizeFqn, 'ROOT.', ''), '.', '_') ;
pattern = ['^tb_viz_.*', regexptranslate('escape', suffix), '$'] ;
vizButton = findall(hFig, '-regexp', 'Tag', pattern) ;
assert(~isempty(vizButton), ...
       'No Viz button for %s', batchSizeFqn) ;
vizButton = vizButton(1) ;
vizButton.Value = 1 ;
feval(vizButton.ValueChangedFcn, vizButton, []) ;
drawnow ;

% Toggling the batch-size Viz button should have populated the
% visualization pane with an axes containing a plotted line
vizAxes = findall(vizPanel, 'Type', 'axes') ;
assert(numel(vizAxes) >= 1, ...
       'The visualization pane contains no axes') ;
lineCount = numel(findall(vizAxes(1), 'Type', 'line')) ;
assert(lineCount >= 1, ...
       'The visualization axes contains no line object') ;

% The memory visualization plots memory required against batch size
xLabel = get(get(vizAxes(1), 'XLabel'), 'String') ;
yLabel = get(get(vizAxes(1), 'YLabel'), 'String') ;
assert(strcmp(xLabel, 'Batch size'), ...
       'x-axis label is "%s", expected "Batch size"', xLabel) ;
assert(strcmp(yLabel, 'Memory required (GB)'), ...
       'y-axis label is "%s", expected "Memory required (GB)"', yLabel) ;

% Dismiss the dialog
cancelButton = findall(hFig, 'Tag', 'pb_cancel') ;
assert(isscalar(cancelButton), 'No Cancel button in the dialog') ;
feval(cancelButton.ButtonPushedFcn, cancelButton, []) ;
drawnow ;

fprintf('test_batch_size_memory_visualization passed.\n') ;

end  % function


function deleteIfValid(hFig)
% Delete a figure, tolerating an already-deleted one.
if isvalid(hFig)
  delete(hFig) ;
end
end  % function
