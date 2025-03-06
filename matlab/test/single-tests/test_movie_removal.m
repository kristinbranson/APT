function test_movie_removal()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2024-11-19-with-gt-and-rois-added.lbl' ;
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
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too
  cleaner2 = onCleanup(@()(delete(labeler))) ;  % but just to be sure

  originalMovieCount = numel(labeler.movieFilesAll) ;

  firstMovieIndexToRemove = 7 ;
  labeler.movieRmGUI(firstMovieIndexToRemove, 'force', true) ;

  secondMovieIndexToRemove = 6 ;  % Want one in the middle
  labeler.movieRmGUI(secondMovieIndexToRemove, 'force', true) ;

  % Do verification
  finalMovieCount = numel(labeler.movieFilesAll) ; 
  if ~isequal(finalMovieCount, originalMovieCount-2)
    error('Final movie count is not expected value') ;
  end
end  % function    
