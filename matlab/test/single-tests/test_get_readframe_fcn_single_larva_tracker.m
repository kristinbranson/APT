function test_get_readframe_fcn_single_larva_tracker()
  % get_readframe_fcn() must open a SingleLarvaTracker .mat movie: the frame
  % count is the number of raw images and the header carries the first
  % frame's size.
  %
  % Regression test: that branch read imraw and firstframeim as bare variables
  % rather than as fields of the loaded videodata struct, and switched on the
  % struct that load() returns rather than on the videofiletype string inside
  % it, so any .mat movie errored on open.

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

  [readframe, nframes, fid, headerinfo] = get_readframe_fcn(fileName) ;
  assert(isa(readframe, 'function_handle'), 'No readframe function returned') ;
  assert(nframes == frameCount, 'Expected %d frames, got %d', frameCount, nframes) ;
  assert(fid == 0, 'A .mat movie should not leave a file open') ;
  assert(headerinfo.nr == rowCount && headerinfo.nc == columnCount, ...
         'Header has the wrong frame size') ;
  assert(strcmp(headerinfo.type, 'SingleLarvaTracker'), 'Header has the wrong type') ;
end  % function
