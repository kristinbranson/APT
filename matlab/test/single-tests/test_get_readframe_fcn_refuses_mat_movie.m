function test_get_readframe_fcn_refuses_mat_movie()
  % Test that get_readframe_fcn() refuses a .mat "movie".  The one .mat
  % movie format it dispatched on, SingleLarvaTracker, read its frames
  % through ReadSingleLarvaTrackerFrame, which APT never shipped, so the
  % branch could open a movie whose frames could never be read.  With the
  % branch gone, a .mat falls through to VideoReader, which cannot open it.

  rowCount = 4 ;
  columnCount = 6 ;
  frameCount = 3 ;
  videofiletype = 'SingleLarvaTracker' ;
  firstframeim = zeros(rowCount, columnCount, 'uint8') ;
  imraw = repmat({firstframeim}, [1 frameCount]) ;
  finalbbox = [1 1 columnCount rowCount] ;
  fps = 30 ;
  fileName = [tempname() '.mat'] ;
  save(fileName, 'videofiletype', 'firstframeim', 'imraw', 'finalbbox', 'fps') ;
  cleaner = onCleanup(@()(delete(fileName))) ;

  didError = false ;
  try
    get_readframe_fcn(fileName) ;
  catch exception
    didError = true ;
    assert(startsWith(exception.message, 'Could not open file'), ...
           'A .mat movie should be refused by the VideoReader fallback, got: %s', exception.message) ;
  end
  assert(didError, 'A .mat movie should be refused') ;
end  % function
