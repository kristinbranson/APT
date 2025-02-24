function test_movie_removal()
  [labeler, controller] = ...
    StartAPT('projfile', '/groups/branson/bransonlab/apt/unittest/four-points-testing-2024-11-19-with-gt-and-rois-added.lbl');
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

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
