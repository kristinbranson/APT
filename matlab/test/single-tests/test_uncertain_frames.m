function test_uncertain_frames()
% Test that the Uncertain Frames window shows the correct number of items
% when switching between movies.
linux_project_file_path = ...
  '/groups/branson/bransonlab/apt/unittest/with-trx-project-with-short-movie-tracked.lbl' ;
if ispc()
  project_file_path = strrep(linux_project_file_path, '/groups/branson/bransonlab', 'Z:') ;
  replace_path = { '/groups/branson/bransonlab', 'Z:' } ;
else
  project_file_path = linux_project_file_path ;
  replace_path = [] ;
end

[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleaner = onCleanup(@()(delete(controller))) ;
cleaner2 = onCleanup(@()(delete(labeler))) ;

% Verify that movie 10 is the current movie
if labeler.currMovie ~= 10
  error('Expected current movie to be 10, but got %d', labeler.currMovie) ;
end

% Open the Uncertain Frames window
controller.controlActuated('menu_evaluate_show_uncertain_frames') ;

% Check the listbox has expectedListboxItemCount items
expectedListboxItemCount = 155 ;
listbox = findall(0, 'Tag', 'uncertain_frames_listbox') ;
itemCountForMovie10 = numel(listbox.String) ;
if itemCountForMovie10 ~= expectedListboxItemCount
  error('Expected %d items in listbox for movie 10, but got %d', expectedListboxItemCount, itemCountForMovie10) ;
end

% Switch to movie 8
labeler.movieSet(8) ;

% Check the listbox has 0 items
itemCountForMovie8 = numel(listbox.String) ;
if itemCountForMovie8 ~= 0
  error('Expected 0 items in listbox for movie 8, but got %d', itemCountForMovie8) ;
end

% Switch back to movie 10
labeler.movieSet(10) ;

% Check the listbox has expectedListboxItemCount items again
itemCountForMovie10Again = numel(listbox.String) ;
if itemCountForMovie10Again ~= expectedListboxItemCount
  error('Expected %d items in listbox for movie 10 after switching back, but got %d', ...
        expectedListboxItemCount, itemCountForMovie10Again) ;
end

end  % function
