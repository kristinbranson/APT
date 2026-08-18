function test_mp4_read_nframes()
% Test that mp4_read_nframes() reads the exact frame count from MP4 container
% metadata, for files with the moov box at the end (the common case, and how
% the cameras in the lab write files) and at the front (faststart).  The
% metadata count is cross-checked against VideoReader's NumberOfFrames, which
% is the authoritative-but-slow count that mp4_read_nframes() exists to avoid.

  cases = struct('linuxPath', ...
                   { '/groups/branson/bransonlab/apt/unittest/mp4_nframes_test_moov_at_end.mp4', ...
                     '/groups/branson/bransonlab/apt/unittest/mp4_nframes_test_moov_at_front.mp4' }, ...
                 'expectedFrameCount', ...
                   { 137, ...
                     89 }) ;

  for caseIndex = 1 : numel(cases) ,
    linuxPath = cases(caseIndex).linuxPath ;
    expectedFrameCount = cases(caseIndex).expectedFrameCount ;
    moviePath = localize_test_project_path(linuxPath) ;

    metadataFrameCount = mp4_read_nframes(moviePath) ;
    if ~isequal(metadataFrameCount, expectedFrameCount) ,
      error('mp4_read_nframes returned %d frames for %s, expected %d', ...
            metadataFrameCount, moviePath, expectedFrameCount) ;
    end

    readerObj = VideoReader(moviePath) ;
    videoReaderFrameCount = get(readerObj, 'NumberOfFrames') ;
    if ~isequal(metadataFrameCount, videoReaderFrameCount) ,
      error('mp4_read_nframes returned %d frames for %s, but VideoReader reports %d', ...
            metadataFrameCount, moviePath, videoReaderFrameCount) ;
    end

    % get_readframe_fcn() should return the same frame count (via the metadata
    % fast path) and a headerinfo struct populated with the movie's metadata,
    % without having called VideoReader's slow NumberOfFrames.
    [readframe, readframeFrameCount, fid, headerinfo] = get_readframe_fcn(moviePath) ;
    cleaner = onCleanup(@()(closeIfOpen(fid))) ;
    if ~isequal(readframeFrameCount, expectedFrameCount) ,
      error('get_readframe_fcn returned %d frames for %s, expected %d', ...
            readframeFrameCount, moviePath, expectedFrameCount) ;
    end
    requiredFields = { 'FrameRate', 'VideoFormat', 'nr', 'nc', 'nframes' } ;
    for fieldIndex = 1 : numel(requiredFields) ,
      if ~isfield(headerinfo, requiredFields{fieldIndex}) ,
        error('get_readframe_fcn headerinfo for %s is missing field %s', ...
              moviePath, requiredFields{fieldIndex}) ;
      end
    end
    firstImage = readframe(1) ;
    if size(firstImage, 1) ~= headerinfo.nr || size(firstImage, 2) ~= headerinfo.nc ,
      error('get_readframe_fcn frame size does not match headerinfo for %s', moviePath) ;
    end
    clear cleaner ;
  end
end  % function

function closeIfOpen(fid)
% Close the file identifier fid if it refers to an open file.
  if fid > 0 ,
    fclose(fid) ;
  end
end  % function
