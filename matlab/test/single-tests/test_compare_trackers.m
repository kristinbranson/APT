function test_compare_trackers()
% Test that the Compare Trackers window populates its dropdowns and
% listbox correctly, flags the reference (current) tracker pink in the
% test dropdown, and shows the same-tracker warning when the test
% selection coincides with the reference selection.

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

% Both dropdowns should be populated with one item per tracker.
referenceDropdown = findall(0, 'Tag', 'compare_trackers_reference_dropdown') ;
testDropdown = findall(0, 'Tag', 'compare_trackers_test_dropdown') ;
if numel(referenceDropdown.Items) ~= trackerCount
  error('Reference dropdown has %d items, expected %d', ...
        numel(referenceDropdown.Items), trackerCount) ;
end
if numel(testDropdown.Items) ~= trackerCount
  error('Test dropdown has %d items, expected %d', ...
        numel(testDropdown.Items), trackerCount) ;
end

% The reference dropdown is disabled and always shows the current
% tracker (trackerHistory index 1).
if ~strcmp(referenceDropdown.Enable, 'off')
  error('Expected reference dropdown to be disabled, but Enable=%s', referenceDropdown.Enable) ;
end
if referenceDropdown.Value ~= 1
  error('Expected reference dropdown to show the current tracker (index 1), but Value=%d', ...
        referenceDropdown.Value) ;
end

% Defaults: reference = first tracker, test = second.  The two should
% differ, so the listbox should contain at least one bout.
if referenceDropdown.Value == testDropdown.Value
  error('Expected default reference and test selections to differ') ;
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

% The reference (current) tracker should be flagged pink in the test
% dropdown.  In R2023a and later this is a per-item style on the first
% item, shown regardless of the current test selection; in older
% releases the whole control turns pink only when the test selection
% coincides with the reference.
pinkColor = [1 0.8 0.85] ;
model = labeler.compareTrackersModel_ ;
if verLessThan('matlab', '9.14')  % R2023a
  model.testTrackerHistoryIndex = model.referenceTrackerHistoryIndex ;
  if ~isequal(testDropdown.BackgroundColor, pinkColor)
    error('Expected test dropdown background to be pink when ref==test') ;
  end
else
  if ~isReferenceItemPink_(testDropdown, pinkColor)
    error('Expected the reference tracker item in the test dropdown to have a pink background') ;
  end
end

% Set the test selection equal to the reference; the listbox should
% switch to a single italic warning entry.
model.testTrackerHistoryIndex = model.referenceTrackerHistoryIndex ;
if numel(listbox.Items) ~= 1
  error('Expected listbox to have a single warning entry when ref==test, but got %d items', ...
        numel(listbox.Items)) ;
end
if ~strcmp(listbox.FontAngle, 'italic')
  error('Expected listbox FontAngle to be italic when ref==test, but got %s', listbox.FontAngle) ;
end
if ~strcmp(listbox.Enable, 'off')
  error('Expected listbox to be disabled when ref==test, but got Enable=%s', listbox.Enable) ;
end

% Restore the test selection to a different tracker; listbox should
% repopulate.
otherTrackerIndex = 1 + mod(model.referenceTrackerHistoryIndex, trackerCount) ;
model.testTrackerHistoryIndex = otherTrackerIndex ;
if numel(listbox.Items) < expectedListboxItemCount
  error('Expected >= %d bout items in listbox after restoring distinct test, but got %d', ...
        expectedListboxItemCount, numel(listbox.Items)) ;
end
if strcmp(listbox.FontAngle, 'italic')
  error('Listbox should be in normal font after restoring distinct test') ;
end

end  % function



function result = isReferenceItemPink_(dropdown, pinkColor)
% Return whether the dropdown has a per-item style giving the first
% (reference) item the given background color.
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
