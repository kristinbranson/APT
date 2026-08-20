function test_mjpg_reading()
% Test that reading frames from an indexed .mjpg movie is order-independent: a
% frame read out of order must return the same pixels as the same frame read in
% order.  get_readframe_fcn reads .mjpg via the companion <basename>.txt index,
% seeking to each frame's byte offset, so a frame's content must not depend on
% what was read before it.  We read 10 evenly-spaced frames in ascending order,
% then read the same 10 in a fixed randomly-shuffled order, and require each
% shuffled read to match.

  % Fix the RNG so the shuffle (and thus the test) is deterministic.
  rng(42) ;

  linuxPath = '/groups/branson/bransonlab/apt/unittest/roian/190523_m164301sbpb_no_odor_m164564_ft164992.mjpg' ;
  moviePath = localize_test_project_path(linuxPath) ;

  [readframe, nframes, fid] = get_readframe_fcn(moviePath) ;
  cleaner = onCleanup(@()(closeIfOpen(fid))) ;

  frameCount = 10 ;
  frameIndices = round(linspace(1, nframes, frameCount)) ;

  % Read the frames in ascending index order.
  inOrderFrames = cell(1, frameCount) ;
  for orderedIndex = 1 : frameCount ,
    inOrderFrames{orderedIndex} = readframe(frameIndices(orderedIndex)) ;
  end

  % Read the same frames in a shuffled order and compare each to the in-order read.
  permutation = randperm(frameCount) ;
  for shuffledIndex = 1 : frameCount ,
    frameNumber = frameIndices(permutation(shuffledIndex)) ;
    shuffledFrame = readframe(frameNumber) ;
    if ~isequal(shuffledFrame, inOrderFrames{permutation(shuffledIndex)}) ,
      error('MJPG frame %d read out of order differs from the same frame read in order', ...
            frameNumber) ;
    end
  end

end  % function

function closeIfOpen(fid)
% Close the file identifier fid if it refers to an open file.
  if fid > 0 ,
    fclose(fid) ;
  end
end  % function
