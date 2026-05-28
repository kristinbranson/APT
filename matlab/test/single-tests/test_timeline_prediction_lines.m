function test_timeline_prediction_lines()
% Test that InfoTimeline prediction lines are present after loading a
% tracked project, and absent after clearing tracking results.
linux_project_file_path = ...
  '/groups/branson/bransonlab/apt/unittest/with-trx-project-with-short-movie-tracked-relocated.lbl' ;
[project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleaner = onCleanup(@()(delete(controller))) ;
cleaner2 = onCleanup(@()(delete(labeler))) ;

% Check that the timeline has prediction lines with valid data
nLandmarks = 17 ;
nFrames = 1000 ;
ax = controller.axes_timeline_manual ;
for iLandmark = 1:nLandmarks
  tag = sprintf('InfoTimeline_Pt%d', iLandmark) ;
  hLine = findobj(ax, 'Tag', tag) ;
  if isempty(hLine)
    error('Could not find line object with tag %s', tag) ;
  end
  if numel(hLine) ~= 1
    error('Expected 1 line object with tag %s, but found %d', tag, numel(hLine)) ;
  end
  xData = get(hLine, 'XData') ;
  yData = get(hLine, 'YData') ;
  if numel(xData) ~= nFrames
    error('Tag %s: expected XData to have %d elements, but got %d', ...
          tag, nFrames, numel(xData)) ;
  end
  if numel(yData) ~= nFrames
    error('Tag %s: expected YData to have %d elements, but got %d', ...
          tag, nFrames, numel(yData)) ;
  end
  if any(isnan(xData))
    error('Tag %s: XData contains NaNs', tag) ;
  end
  if any(isnan(yData))
    error('Tag %s: YData contains NaNs', tag) ;
  end
end

% Clear all tracking results
labeler.clearTrackingResults() ;

% Verify that the timeline prediction lines are gone, all-NaN, or invisible
for iLandmark = 1:nLandmarks
  tag = sprintf('InfoTimeline_Pt%d', iLandmark) ;
  hLine = findobj(ax, 'Tag', tag) ;
  if isempty(hLine)
    continue ;  % line deleted, that's fine
  end
  isInvisible = strcmp(get(hLine, 'Visible'), 'off') ;
  xData = get(hLine, 'XData') ;
  yData = get(hLine, 'YData') ;
  isAllNaN = all(isnan(xData)) || all(isnan(yData)) ;
  if ~isInvisible && ~isAllNaN
    error('Tag %s: expected line to be invisible or all-NaN after clearing, but it is visible with non-NaN data', ...
          tag) ;
  end
end

end  % function
