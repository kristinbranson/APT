function test_gt_manager_frame_selection_with_multiple_targets()
% Test that the GT Manager selects the right frame-table row as the current frame
% changes, including when the table has more than one row for that frame.
%
% In a project without trx, GTManagerController looks up the frame-table row by
% frame only, ignoring the target.  So in a multianimal project, a frame with
% several targets yields several matching rows, and the lookup produces a column
% vector of row indices.  A uitable Selection must be empty or a scalar row
% index (these tables are 'Multiselect','off'), so setting it errored, which
% showed up as a warning every time the current frame changed, e.g. during video
% playback.

% Start APT with a multianimal project that has groundtruth labels
linux_project_file_path = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'four-points-testing-2024-11-19-with-gt-added-and-movie-paths-moved.lbl'] ;
[projectFile, replace_path] = localize_test_project_path(linux_project_file_path) ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                 'replace_path', replace_path, ...
                                 'isInDebugMode', true, ...
                                 'isInYodaMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

assert(labeler.maIsMA, 'Expected a multianimal project.') ;
assert(~labeler.projectHasTrx, 'Expected a project without trx.') ;

labeler.gtSetGTMode(true) ;
drawnow() ;

% Add a to-label list with two targets in one frame, so that the GT Manager's
% frame table is sure to have a frame with more than one row
frameWithTwoTargets = 101 ;
mov = MovieIndex([1 ; 1], true) ;
frm = [frameWithTwoTargets ; frameWithTwoTargets] ;
iTgt = [1 ; 2] ;
labeler.gtSetUserSuggestions(table(mov, frm, iTgt)) ;
drawnow() ;

controller.gtShowGTManager() ;
drawnow() ;
gtManagerController = controller.gtManagerController_ ;

% Navigate once to populate the frame table for the current movie
labeler.setFrame(1) ;
gtManagerController.currMovFrmTgtChanged_() ;
frameFromRow = cell2mat(gtManagerController.tblFrame_.Data(:,1)) ;
assert(~isempty(frameFromRow), 'GT Manager frame table is empty.') ;

% A frame the table has several rows for, and one it has exactly one row for
rowCountFromFrame = arrayfun(@(f)(nnz(frameFromRow==f)), frameFromRow) ;
assert(any(rowCountFromFrame>1), 'No frame with multiple rows, cannot test that case.') ;
assert(any(rowCountFromFrame==1), 'No frame with exactly one row, cannot test that case.') ;
multipleRowFrame = frameFromRow(find(rowCountFromFrame>1, 1)) ;
singleRowFrame = frameFromRow(find(rowCountFromFrame==1, 1)) ;

% In both cases a single row, for the current frame, should end up selected
for targetFrame = [singleRowFrame multipleRowFrame]
  labeler.setFrame(targetFrame) ;
  gtManagerController.currMovFrmTgtChanged_() ;
  selection = gtManagerController.tblFrame_.Selection ;
  assert(isscalar(selection), ...
         'Expected a scalar frame-table selection for frame %d, got %s.', ...
         targetFrame, mat2str(selection)) ;
  selectedFrame = gtManagerController.tblFrame_.Data{selection,1} ;
  assert(selectedFrame == targetFrame, ...
         'Expected the selected row to be frame %d, got %d.', ...
         targetFrame, selectedFrame) ;
end

% A frame that is not in the table at all should clear the selection
unlistedFrames = setdiff(1:labeler.nframes, frameFromRow) ;
assert(~isempty(unlistedFrames), 'Every frame is in the table, cannot test that case.') ;
unlistedFrame = unlistedFrames(1) ;
labeler.setFrame(unlistedFrame) ;
gtManagerController.currMovFrmTgtChanged_() ;
selection = gtManagerController.tblFrame_.Selection ;
assert(isempty(selection), ...
       'Expected an empty frame-table selection for frame %d, got %s.', ...
       unlistedFrame, mat2str(selection)) ;

end  % function
