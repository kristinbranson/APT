function test_compare_trackers()
% Test that the Compare Trackers window populates its dropdowns and
% listbox correctly, flags the reference (current) tracker pink in the
% test dropdown, and tracks the test-tracker selection by identity so
% that making the test tracker the current tracker is handled
% gracefully.

% Temporary location until the test project lands under
% /groups/branson/bransonlab/apt/unittest/.
linux_project_file_path = ...
  '/groups/branson/bransonlab/apt/unittest/htflies-10-with-trks-from-two-trackers-relocated.lbl' ;
[project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleaner = onCleanup(@()(delete(controller))) ;
cleaner2 = onCleanup(@()(delete(labeler))) ;

% The project should have at least two trained trackers.
trackerCount = numel(labeler.trackerHistory) ;
if trackerCount < 2
  error('Expected trackerHistory to have >= 2 entries, but got %d', trackerCount) ;
end

% Open the Compare Trackers window.
controller.controlActuated('menu_evaluate_compare_trackers') ;

% The reference dropdown lists every tracker; the test dropdown omits
% the current (reference) tracker, so it has one fewer item.
referenceDropdown = findall(0, 'Tag', 'compare_trackers_reference_dropdown') ;
testDropdown = findall(0, 'Tag', 'compare_trackers_test_dropdown') ;
currentTracker = labeler.tracker ;
if numel(referenceDropdown.Items) ~= trackerCount
  error('Reference dropdown has %d items, expected %d', ...
        numel(referenceDropdown.Items), trackerCount) ;
end
if numel(testDropdown.Items) ~= trackerCount - 1
  error('Test dropdown has %d items, expected %d', ...
        numel(testDropdown.Items), trackerCount - 1) ;
end
if any(cellfun(@(d)(d == currentTracker), testDropdown.ItemsData))
  error('Test dropdown should omit the current tracker when it is not selected as the test tracker') ;
end

% The reference dropdown is disabled and always shows the current
% tracker.
if ~strcmp(referenceDropdown.Enable, 'off')
  error('Expected reference dropdown to be disabled, but Enable=%s', referenceDropdown.Enable) ;
end
if ~(referenceDropdown.Value == currentTracker)
  error('Expected reference dropdown to show the current tracker') ;
end

% Defaults: reference = current tracker, test = a different tracker.  The
% two differ, so the listbox should contain at least one bout.
if referenceDropdown.Value == testDropdown.Value
  error('Expected reference and test selections to differ') ;
end
listbox = findall(0, 'Tag', 'compare_trackers_listbox') ;
expectedListboxItemCount = 1 ;
if numel(listbox.Items) < expectedListboxItemCount
  error('Expected >= %d bout items in listbox, but got %d', ...
        expectedListboxItemCount, numel(listbox.Items)) ;
end
if strcmp(listbox.FontAngle, 'italic')
  error('Listbox should be in normal font when bouts are present, but FontAngle is italic') ;
end

% The mode dropdown offers both modes and defaults to Maximum Landmark
% Distance.
modeDropdown = findall(0, 'Tag', 'compare_trackers_mode_dropdown') ;
if numel(modeDropdown.Items) ~= 2
  error('Mode dropdown has %d items, expected 2', numel(modeDropdown.Items)) ;
end
if modeDropdown.Value ~= CompareTrackersMode.MaximumLandmarkDistance
  error('Expected the mode dropdown to default to Maximum Landmark Distance') ;
end

% The model starts with no current bout, so no listbox item should be
% selected.
if ~isempty(listbox.ValueIndex)
  error('Expected no listbox item to be selected initially, but ValueIndex is %d', listbox.ValueIndex) ;
end

% With no bout selected, the preview image and pose overlays are hidden.
previewImage = findall(0, 'Tag', 'compare_trackers_preview_image') ;
refScatter = findall(0, 'Tag', 'compare_trackers_preview_ref_scatter') ;
testScatter = findall(0, 'Tag', 'compare_trackers_preview_test_scatter') ;
if strcmp(previewImage.Visible, 'on')
  error('Expected the preview image to be hidden when no bout is selected') ;
end
if strcmp(refScatter.Visible, 'on') || strcmp(testScatter.Visible, 'on')
  error('Expected the pose overlays to be hidden when no bout is selected') ;
end
placeholderText = findall(0, 'Tag', 'compare_trackers_preview_placeholder_text') ;
if ~strcmp(placeholderText.Visible, 'on')
  error('Expected the placeholder text to be shown when no bout is selected') ;
end

% Selecting a bout through the model should select the matching listbox
% item.
labeler.compareTrackersCurrentBoutIndexMaybe = 1 ;
if ~isequal(listbox.ValueIndex, 1)
  error('Expected listbox ValueIndex to be 1 after selecting bout 1') ;
end

% Selecting a bout populates the preview: the bout's max-distance frame
% image is shown and both trackers' poses are overlaid.
if ~strcmp(previewImage.Visible, 'on') || isempty(previewImage.CData)
  error('Expected the preview image to be shown after selecting a bout') ;
end
if strcmp(placeholderText.Visible, 'on')
  error('Expected the placeholder text to be hidden after selecting a bout') ;
end
if ~strcmp(refScatter.Visible, 'on')
  error('Expected the reference pose overlay to be shown after selecting a bout') ;
end
if ~strcmp(testScatter.Visible, 'on')
  error('Expected the test pose overlay to be shown after selecting a bout') ;
end

% Each landmark's ref and test predictions are joined by a connector
% line; at least one should be visible for the selected bout.
connectorLines = findall(0, 'Tag', 'compare_trackers_preview_connector_line') ;
isVisibleFromConnectorIndex = arrayfun(@(h)(strcmp(h.Visible, 'on')), connectorLines) ;
if isempty(connectorLines) || ~any(isVisibleFromConnectorIndex)
  error('Expected at least one visible connector line after selecting a bout') ;
end

% The preview axes limits should bound all the finite overlaid landmarks.
previewAxes = findall(0, 'Tag', 'compare_trackers_preview_axes') ;
overlayX = [refScatter.XData(:) ; testScatter.XData(:)] ;
overlayY = [refScatter.YData(:) ; testScatter.YData(:)] ;
isFinitePoint = isfinite(overlayX) & isfinite(overlayY) ;
if ~any(isFinitePoint)
  error('Expected at least one finite overlaid landmark in the preview') ;
end
if any(overlayX(isFinitePoint) < previewAxes.XLim(1)) || ...
   any(overlayX(isFinitePoint) > previewAxes.XLim(2)) || ...
   any(overlayY(isFinitePoint) < previewAxes.YLim(1)) || ...
   any(overlayY(isFinitePoint) > previewAxes.YLim(2))
  error('Expected the preview axes limits to bound all overlaid landmarks') ;
end

% Clicking the already-selected item should still navigate: this is how
% the user jumps back to the bout's frame after manually moving away.
% (ValueChangedFcn does not fire for such a click, but ClickedFcn does;
% simulate the latter's actuation.)
boutFrame = labeler.currFrame ;
if boutFrame == labeler.nframes
  awayFrame = boutFrame - 1 ;
else
  awayFrame = boutFrame + 1 ;
end
labeler.setFrame(awayFrame) ;
if labeler.currFrame ~= awayFrame
  error('Expected to navigate away to frame %d, but currFrame is %d', awayFrame, labeler.currFrame) ;
end
clickEvent = struct('InteractionInformation', struct('Item', 1)) ;
exceptionMaybe = controller.controlActuated('compare_trackers_listbox_clicked', listbox, clickEvent) ;
if ~isempty(exceptionMaybe)
  error('Simulated listbox click raised an exception: %s', exceptionMaybe{1}.message) ;
end
if labeler.currFrame ~= boutFrame
  error('Expected the re-click on the selected bout to navigate back to frame %d, but currFrame is %d', ...
        boutFrame, labeler.currFrame) ;
end

% No test-dropdown item is flagged pink while the current tracker is
% absent from the list.
pinkColor = [1 0.8 0.85] ;
if ~verLessThan('matlab', '9.14') && isReferenceItemPink_(testDropdown, pinkColor)  % R2023a
  error('Test dropdown should not flag any item pink when the current tracker is not in the list') ;
end

% Selecting the current tracker as the test tracker is the graceful
% case: the test dropdown then includes the current tracker (flagged
% pink), and the listbox shows a single italic warning entry.
model = labeler.compareTrackersModel_ ;
model.testTracker = model.referenceTracker ;
assertGracefulSameTrackerState_(testDropdown, listbox, trackerCount, pinkColor) ;

% Pick a distinct test tracker (the second tracker in the history); the
% current tracker drops out of the list and the listbox repopulates.
secondTracker = labeler.trackerHistory{2} ;
model.testTracker = secondTracker ;
if numel(testDropdown.Items) ~= trackerCount - 1
  error('Test dropdown should omit the current tracker for a distinct test; got %d items, expected %d', ...
        numel(testDropdown.Items), trackerCount - 1) ;
end
if numel(listbox.Items) < expectedListboxItemCount
  error('Expected >= %d bout items in listbox for a distinct test, but got %d', ...
        expectedListboxItemCount, numel(listbox.Items)) ;
end

% The bout list was rebuilt when the test tracker changed, so the bout
% selection should have been reset to none.
if ~isempty(listbox.ValueIndex)
  error('Expected listbox selection to reset after the bout list was rebuilt, but ValueIndex is %d', listbox.ValueIndex) ;
end

% Now make the selected test tracker the current tracker.  Because the
% test selection is tracked by identity, it should follow the same
% tracker -- which is now the current (reference) tracker -- so the
% window should switch to the graceful same-tracker state.  (With the
% old positional-index design, the test selection would instead have
% pointed at whatever tracker landed in that slot.)
labeler.trackMakeExistingTrackerCurrentGivenIndex(2) ;
if ~(labeler.tracker == secondTracker)
  error('Expected the second tracker to become the current tracker') ;
end
if ~(model.testTracker == secondTracker)
  error('Expected the test selection to still refer to the same tracker by identity') ;
end
assertGracefulSameTrackerState_(testDropdown, listbox, trackerCount, pinkColor) ;

% Make the other tracker current again.  The test selection still refers
% to secondTracker (now the non-current tracker), so the comparison is
% meaningful again and the listbox repopulates.
labeler.trackMakeExistingTrackerCurrentGivenIndex(2) ;
if labeler.tracker == secondTracker
  error('Expected the current tracker to change away from the second tracker') ;
end
if ~(model.testTracker == secondTracker)
  error('Expected the test selection to still refer to the same tracker by identity') ;
end
if ~(referenceDropdown.Value == labeler.tracker)
  error('Expected reference dropdown to show the new current tracker after the change') ;
end
if numel(listbox.Items) < expectedListboxItemCount
  error('Expected the listbox to repopulate after the current tracker changed, but got %d items', ...
        numel(listbox.Items)) ;
end
if strcmp(listbox.FontAngle, 'italic')
  error('Listbox should be in normal font after the current tracker changed') ;
end

% ---- Unmatched Animal Count mode ----
% Switching the mode through the dropdown rebuilds the listbox from the
% per-frame unmatched-animal count.  Exercise the full actuation path.
modeDropdown.Value = CompareTrackersMode.UnmatchedAnimalCount ;
modeSwitchException = ...
  controller.controlActuated('compare_trackers_mode_dropdown', modeDropdown, struct()) ;
if ~isempty(modeSwitchException)
  error('Switching to Unmatched Animal Count mode raised an exception: %s', modeSwitchException{1}.message) ;
end
if model.mode ~= CompareTrackersMode.UnmatchedAnimalCount
  error('Expected the model mode to be UnmatchedAnimalCount after switching the dropdown') ;
end
if numel(listbox.Items) < expectedListboxItemCount
  error('Expected >= %d bout items in the listbox in Unmatched Animal Count mode, but got %d', ...
        expectedListboxItemCount, numel(listbox.Items)) ;
end
if strcmp(listbox.FontAngle, 'italic')
  error('Listbox should be in normal font when unmatched-count bouts are present') ;
end

% Rebuilding the bout list resets the selection to none.
if ~isempty(listbox.ValueIndex)
  error('Expected the bout selection to reset after switching mode, but ValueIndex is %d', listbox.ValueIndex) ;
end

% Each line follows the "Frm <range>  UnmatchedCount <n>" format, with no
% tracklet/target index or distance.
firstUnmatchedLine = listbox.Items{1} ;
boutFrameTokens = ...
  regexp(firstUnmatchedLine, '^Frm (\d+)(?:-(\d+))?  UnmatchedCount (\d+)$', 'tokens', 'once') ;
if isempty(boutFrameTokens)
  error('Unexpected unmatched-count listbox line format: "%s"', firstUnmatchedLine) ;
end
if contains(firstUnmatchedLine, 'Trklet') || contains(firstUnmatchedLine, 'Tgt') || ...
   contains(firstUnmatchedLine, 'MaxDist')
  error('Unmatched-count listbox line should not contain a tracklet index or distance: "%s"', ...
        firstUnmatchedLine) ;
end

% Selecting a bout shows the whole peak frame with no pose decorations.
labeler.compareTrackersCurrentBoutIndexMaybe = 1 ;
if ~isequal(listbox.ValueIndex, 1)
  error('Expected listbox ValueIndex to be 1 after selecting bout 1 in unmatched-count mode') ;
end
if ~strcmp(previewImage.Visible, 'on') || isempty(previewImage.CData)
  error('Expected the preview image to be shown after selecting an unmatched-count bout') ;
end
if strcmp(placeholderText.Visible, 'on')
  error('Expected the placeholder text to be hidden after selecting an unmatched-count bout') ;
end
if strcmp(refScatter.Visible, 'on') || strcmp(testScatter.Visible, 'on')
  error('Expected no pose overlays to be shown in unmatched-count mode') ;
end
unmatchedConnectorLines = findall(0, 'Tag', 'compare_trackers_preview_connector_line') ;
if any(arrayfun(@(h)(strcmp(h.Visible, 'on')), unmatchedConnectorLines))
  error('Expected no visible connector lines in unmatched-count mode') ;
end

% The preview shows the whole frame: the axes limits span the full image.
imageHeight = size(previewImage.CData, 1) ;
imageWidth = size(previewImage.CData, 2) ;
expectedXLim = [0.5, imageWidth + 0.5] ;
expectedYLim = [0.5, imageHeight + 0.5] ;
if ~isequal(previewAxes.XLim, expectedXLim) || ~isequal(previewAxes.YLim, expectedYLim)
  error('Expected the preview axes to span the whole frame in unmatched-count mode') ;
end

% Navigation lands on a frame within the selected bout's range.  Parse the
% range with explicit single- and multi-frame patterns rather than one
% pattern with an optional group: regexp's 'once' option collapses a
% non-participating optional group, which would shift the token indices.
rangeTokens = regexp(firstUnmatchedLine, '^Frm (\d+)-(\d+)  UnmatchedCount \d+$', 'tokens', 'once') ;
singleTokens = regexp(firstUnmatchedLine, '^Frm (\d+)  UnmatchedCount \d+$', 'tokens', 'once') ;
if ~isempty(rangeTokens)
  boutStartFrame = str2double(rangeTokens{1}) ;
  boutEndFrame = str2double(rangeTokens{2}) ;
elseif ~isempty(singleTokens)
  boutStartFrame = str2double(singleTokens{1}) ;
  boutEndFrame = boutStartFrame ;
else
  error('Unexpected unmatched-count listbox line format: "%s"', firstUnmatchedLine) ;
end
if labeler.currFrame < boutStartFrame || labeler.currFrame > boutEndFrame
  error('Expected to navigate to a frame within bout 1''s range [%d, %d], but currFrame is %d', ...
        boutStartFrame, boutEndFrame, labeler.currFrame) ;
end

% Switching back to Maximum Landmark Distance restores that mode's
% distance-based listbox.
modeDropdown.Value = CompareTrackersMode.MaximumLandmarkDistance ;
modeRestoreException = ...
  controller.controlActuated('compare_trackers_mode_dropdown', modeDropdown, struct()) ;
if ~isempty(modeRestoreException)
  error('Switching back to Maximum Landmark Distance mode raised an exception: %s', modeRestoreException{1}.message) ;
end
if model.mode ~= CompareTrackersMode.MaximumLandmarkDistance
  error('Expected the model mode to be MaximumLandmarkDistance after switching back') ;
end
if numel(listbox.Items) < expectedListboxItemCount
  error('Expected >= %d bout items after switching back to Maximum Landmark Distance mode, but got %d', ...
        expectedListboxItemCount, numel(listbox.Items)) ;
end
if ~contains(listbox.Items{1}, 'MaxDist')
  error('Expected the distance-mode listbox lines to contain MaxDist, but got "%s"', listbox.Items{1}) ;
end

end  % function



function assertGracefulSameTrackerState_(testDropdown, listbox, trackerCount, pinkColor)
% Assert the "same tracker" graceful state: the test dropdown includes
% the current tracker (flagged pink) and the listbox shows a single
% italic, disabled warning entry.
if numel(testDropdown.Items) ~= trackerCount
  error('Test dropdown should include the current tracker when it is the test tracker; got %d items, expected %d', ...
        numel(testDropdown.Items), trackerCount) ;
end
if verLessThan('matlab', '9.14')  % R2023a
  if ~isequal(testDropdown.BackgroundColor, pinkColor)
    error('Expected test dropdown background to be pink when test == reference') ;
  end
else
  if ~isReferenceItemPink_(testDropdown, pinkColor)
    error('Expected the current tracker item in the test dropdown to be pink when it is the test tracker') ;
  end
end
if numel(listbox.Items) ~= 1
  error('Expected listbox to have a single warning entry when test == reference, but got %d items', ...
        numel(listbox.Items)) ;
end
if ~strcmp(listbox.FontAngle, 'italic')
  error('Expected listbox FontAngle to be italic when test == reference, but got %s', listbox.FontAngle) ;
end
if ~strcmp(listbox.Enable, 'off')
  error('Expected listbox to be disabled when test == reference, but got Enable=%s', listbox.Enable) ;
end
end  % function



function result = isReferenceItemPink_(dropdown, pinkColor)
% Return whether the dropdown has a per-item style giving its first item
% (the current/reference tracker, when present) the given background
% color.
result = false ;
styleConfigurations = dropdown.StyleConfigurations ;
for styleIndex = 1 : height(styleConfigurations)
  target = string(styleConfigurations.Target(styleIndex)) ;
  targetIndex = styleConfigurations.TargetIndex{styleIndex} ;
  style = styleConfigurations.Style(styleIndex) ;
  if target == "item" && isequal(targetIndex, 1) && isequal(style.BackgroundColor, pinkColor)
    result = true ;
    return
  end
end
end  % function
