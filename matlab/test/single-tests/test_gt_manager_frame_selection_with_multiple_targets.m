function test_gt_manager_frame_selection_with_multiple_targets()
% Test the Groundtruth Navigator's frame table in a multianimal project, where a
% frame can hold several labeled animals.
%
% The navigator's table is the to-label list unioned with the labeled rows.  In a
% multianimal project the labeled rows all carry iTgt==0 ("any target"), so a
% frame that is both in the to-label list and labeled used to be listed twice,
% and a to-label list with a row per animal listed a frame once per animal.
% Those duplicate rows looked identical, since the Target column is only shown
% for projects with trx.  They also broke navigation: the frame-table row lookup
% keys on frame only, so it turned up a column vector of row indices, and
% setting a uitable Selection to that errors, which showed up as a warning on
% every frame change, e.g. during video playback.

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

% A to-label list with a row per animal, which is what used to produce duplicate
% rows for frameWithTwoRows once unioned with the labeled rows
frameWithTwoRows = 101 ;
frameWithOneRow = 201 ;
mov = MovieIndex([1 ; 1 ; 1], true) ;
frm = [frameWithTwoRows ; frameWithTwoRows ; frameWithOneRow] ;
iTgt = [1 ; 2 ; 1] ;
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

% Every frame should be listed exactly once
assert(numel(unique(frameFromRow)) == numel(frameFromRow), ...
       'Frames listed more than once in the frame table: %s', ...
       mat2str(frameFromRow')) ;
assert(ismember(frameWithTwoRows, frameFromRow), ...
       'Frame %d missing from the frame table.', frameWithTwoRows) ;
assert(ismember(frameWithOneRow, frameFromRow), ...
       'Frame %d missing from the frame table.', frameWithOneRow) ;

% Navigating to a listed frame should select that frame's row
for targetFrame = [frameWithOneRow frameWithTwoRows]
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
