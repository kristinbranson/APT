function test_movie_removal()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
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

  firstMovieIndexToRemove = 3 ;
  labeler.movieRmGUI(firstMovieIndexToRemove, 'force', true) ;

  secondMovieIndexToRemove = 2 ;  % Want one in the middle
  labeler.movieRmGUI(secondMovieIndexToRemove, 'force', true) ;

  % Do verification
  finalMovieCount = numel(labeler.movieFilesAll) ; 
  if ~isequal(finalMovieCount, originalMovieCount-2)
    error('Final movie count is not expected value') ;
  end
end  % function    
