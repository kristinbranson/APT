function bug()

[labeler, controller] = ...
  StartAPT('projfile', '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-11-19-with-gt-and-rois-added.lbl');
cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

originalMovieCount = numel(labeler.movieFilesAll) ;

firstMovieIndexToRemove = 7 ;
labeler.movieRm(firstMovieIndexToRemove, 'force', true) ;

secondMovieIndexToRemove = 6 ;  % Want one in the middle
labeler.movieRm(secondMovieIndexToRemove, 'force', true) ;

% Do verification
finalMovieCount = numel(labeler.movieFilesAll) ; 
assert(isequal(finalMovieCount, originalMovieCount-2), 'Final movie count is not expected value') ;

end
