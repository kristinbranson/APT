function test_ufmf_mean_orientation()
  % Check that a .ufmf's mean (background) image comes back in the same
  % orientation as the frames it is the background for, and that the
  % header's row/column counts agree with the frame size.  The movie is
  % synthesized here, and is deliberately non-square: a square one would
  % pass even with rows and columns transposed.

  rowCount = 7 ;
  columnCount = 5 ;
  background = uint8(reshape(1 : rowCount*columnCount, [rowCount columnCount])) ;

  % One foreground box, itself non-square, well away from the image edges
  % and from the diagonal, so a transposed read cannot go unnoticed.
  boxX = 1 ;  % zero-based, as stored in the file
  boxY = 3 ;
  boxColumnCount = 2 ;
  boxRowCount = 3 ;
  boxImage = uint8(reshape(201 : 200+boxRowCount*boxColumnCount, [boxRowCount boxColumnCount])) ;

  expectedFrame = background ;
  expectedFrame(boxY+1 : boxY+boxRowCount, boxX+1 : boxX+boxColumnCount) = boxImage ;

  movieFilePath = horzcat(tempname(), '.ufmf') ;
  cleaner = onCleanup(@()(delete_file_if_it_exists(movieFilePath))) ;
  write_test_ufmf(movieFilePath, background, boxX, boxY, boxImage) ;

  header = ufmf_read_header(movieFilePath) ;
  headerCleaner = onCleanup(@()(fclose(header.fid))) ;

  if ~isequal([header.nr header.nc], [rowCount columnCount]) ,
    error('APT:test', ...
          'ufmf header reports nr=%d, nc=%d for a %dx%d movie', ...
          header.nr, header.nc, rowCount, columnCount) ;
  end

  [meanImage, header] = ufmf_read_mean(header, 'meani', 1) ;
  if ~isequal(size(meanImage), [rowCount columnCount]) ,
    error('APT:test', ...
          'ufmf mean image is %s for a %dx%d movie', ...
          mat2str(size(meanImage)), rowCount, columnCount) ;
  end
  if ~isequal(meanImage, background) ,
    error('APT:test', 'ufmf mean image does not match the background it was written from') ;
  end

  [frameImage, header] = ufmf_read_frame(header, 1) ;  %#ok<ASGLU>
  if ~isequal(size(frameImage), size(meanImage)) ,
    error('APT:test', ...
          'ufmf frame is %s but its mean image is %s', ...
          mat2str(size(frameImage)), mat2str(size(meanImage))) ;
  end
  if ~isequal(frameImage, expectedFrame) ,
    error('APT:test', 'ufmf frame does not match the background plus foreground box it was written from') ;
  end
end  % function



function write_test_ufmf(filePath, background, boxX, boxY, boxImage)
  % Write a minimal one-frame MONO8 .ufmf holding the given background as
  % its single mean keyframe and the given box as its single foreground
  % patch.  UFMF stores images sideways -- the on-disk order runs over
  % colors fastest, then columns, then rows -- so each image is
  % transposed on the way out.

  [rowCount, columnCount] = size(background) ;
  [boxRowCount, boxColumnCount] = size(boxImage) ;

  fid = fopen(filePath, 'wb', 'ieee-le') ;
  if fid < 0 ,
    error('APT:test', 'Unable to open %s for writing', filePath) ;
  end
  cleaner = onCleanup(@()(fclose(fid))) ;

  % File header
  fwrite(fid, 'ufmf', 'char') ;
  fwrite(fid, 4, 'uint') ;  % version
  indexLocationLocation = ftell(fid) ;
  fwrite(fid, 0, 'uint64') ;  % index location, backpatched below
  fwrite(fid, [columnCount rowCount], 'ushort') ;  % max box size, sideways
  fwrite(fid, 0, 'uchar') ;  % is_fixed_size
  fwrite(fid, length('MONO8'), 'uchar') ;
  fwrite(fid, 'MONO8', 'char') ;

  % The mean keyframe
  meanLocation = ftell(fid) ;
  fwrite(fid, 0, 'uchar') ;  % keyframe chunk
  fwrite(fid, length('mean'), 'uchar') ;
  fwrite(fid, 'mean', 'char') ;
  fwrite(fid, 'B', 'char') ;  % uint8 data
  fwrite(fid, [columnCount rowCount], 'ushort') ;  % width, height
  fwrite(fid, 0, 'double') ;  % timestamp
  fwrite(fid, background', 'uint8') ;

  % The single frame, holding the single foreground box
  frameLocation = ftell(fid) ;
  fwrite(fid, 1, 'uchar') ;  % frame chunk
  fwrite(fid, 0, 'double') ;  % timestamp
  fwrite(fid, 1, 'uint32') ;  % box count
  fwrite(fid, [boxX boxY boxColumnCount boxRowCount], 'ushort') ;
  fwrite(fid, boxImage', 'uint8') ;

  % The index
  index = struct() ;
  index.frame = struct() ;
  index.frame.loc = int64(frameLocation) ;
  index.frame.timestamp = 0 ;
  index.keyframe = struct() ;
  index.keyframe.mean = struct() ;
  index.keyframe.mean.loc = int64(meanLocation) ;
  index.keyframe.mean.timestamp = 0 ;
  indexLocation = ftell(fid) ;
  ufmf_write_struct(fid, index) ;

  % Backpatch the index location into the file header
  fseek(fid, indexLocationLocation, 'bof') ;
  fwrite(fid, indexLocation, 'uint64') ;
end  % function



function delete_file_if_it_exists(filePath)
  % Delete the named file, if it is there.
  if exist(filePath, 'file') ,
    delete(filePath) ;
  end
end  % function
