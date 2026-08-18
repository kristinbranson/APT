function nframes = mp4_read_nframes(filename)
% Return the exact frame count of an ISO base-media file (.mp4/.mov/.m4v) by
% traversing its container metadata, without decoding any frames.  Walks the
% box tree to the first video track and reads the sample count from its
% stsz/stz2 box (or, failing that, sums its stts box).  This is fast even for
% very large files because only box headers and a few metadata boxes are read;
% the media data (mdat) is skipped by its size field and never touched.
%
% Errors if the file cannot be opened, cannot be parsed as ISO base-media, or
% has no video track with a frame count.  Callers that want a graceful
% fallback (e.g. to VideoReader) should wrap the call in try/catch.

  fid = fopen(filename, 'r', 'ieee-be') ;
  if fid < 0 ,
    error('mp4_read_nframes:cannotOpenFile', 'Could not open file %s', filename) ;
  end
  cleaner = onCleanup(@()(fclose(fid))) ;
  fileSize = fileSize_(fid) ;
  moovBox = findBox_(fid, 0, fileSize, 'moov') ;
  if isempty(moovBox) ,
    error('mp4_read_nframes:noMoovBox', 'No moov box found in %s', filename) ;
  end
  nframes = videoTrackFrameCount_(fid, moovBox.contentStart, moovBox.contentEnd) ;
  if isempty(nframes) ,
    error('mp4_read_nframes:noVideoTrack', 'No video track with a frame count found in %s', filename) ;
  end
end  % function

function result = fileSize_(fid)
% Return the size of the open file, in bytes.
  fseek(fid, 0, 'eof') ;
  result = ftell(fid) ;
  fseek(fid, 0, 'bof') ;
end  % function

function box = readBoxHeader_(fid, pos, endPos)
% Read the header of the ISO-BMFF box starting at byte offset pos, which must
% lie within [startOfLevel, endPos).  Returns a struct describing the box, or
% [] if the header is truncated or the box overruns endPos (i.e. corrupt).
  box = [] ;
  if pos + 8 > endPos ,
    return
  end
  fseek(fid, pos, 'bof') ;
  size32 = fread(fid, 1, 'uint32') ;
  type = fread(fid, 4, '*char')' ;
  if isempty(size32) || numel(type) < 4 ,
    return
  end
  headerSize = 8 ;
  boxSize = size32 ;
  if size32 == 1 ,
    % 64-bit "largesize" follows the type field.
    boxSize = fread(fid, 1, 'uint64') ;
    headerSize = 16 ;
    if isempty(boxSize) ,
      box = [] ;
      return
    end
  elseif size32 == 0 ,
    % Box extends to the end of the enclosing extent.
    boxSize = endPos - pos ;
  end
  if boxSize < headerSize || pos + boxSize > endPos ,
    return
  end
  box = struct('type', type, ...
               'contentStart', pos + headerSize, ...
               'contentEnd', pos + boxSize) ;
end  % function

function box = findBox_(fid, startPos, endPos, wantedType)
% Return the first child box of the given type within [startPos, endPos), or []
% if none is found.  Only box headers are read, so unrelated large boxes (such
% as mdat) are skipped over by their size field.
  box = [] ;
  pos = startPos ;
  while pos + 8 <= endPos ,
    thisBox = readBoxHeader_(fid, pos, endPos) ;
    if isempty(thisBox) ,
      return
    end
    if strcmp(thisBox.type, wantedType) ,
      box = thisBox ;
      return
    end
    pos = thisBox.contentEnd ;
  end
end  % function

function nframes = videoTrackFrameCount_(fid, moovStart, moovEnd)
% Return the frame (sample) count of the first video track within the moov box
% spanning [moovStart, moovEnd), or [] if there is no such track.
  nframes = [] ;
  pos = moovStart ;
  while pos + 8 <= moovEnd ,
    thisBox = readBoxHeader_(fid, pos, moovEnd) ;
    if isempty(thisBox) ,
      return
    end
    if strcmp(thisBox.type, 'trak') ,
      count = trakFrameCountIfVideo_(fid, thisBox.contentStart, thisBox.contentEnd) ;
      if ~isempty(count) ,
        nframes = count ;
        return
      end
    end
    pos = thisBox.contentEnd ;
  end
end  % function

function count = trakFrameCountIfVideo_(fid, trakStart, trakEnd)
% Return the sample count of the given trak box if it is a video track, or []
% if it is not a video track or its sample table cannot be read.
  count = [] ;
  mdiaBox = findBox_(fid, trakStart, trakEnd, 'mdia') ;
  if isempty(mdiaBox) ,
    return
  end
  hdlrBox = findBox_(fid, mdiaBox.contentStart, mdiaBox.contentEnd, 'hdlr') ;
  if isempty(hdlrBox) ,
    return
  end
  % hdlr is a full box: version+flags (4 bytes), pre_defined (4 bytes), then
  % the four-character handler_type.
  fseek(fid, hdlrBox.contentStart + 8, 'bof') ;
  handlerType = fread(fid, 4, '*char')' ;
  if numel(handlerType) < 4 || ~strcmp(handlerType, 'vide') ,
    return
  end
  minfBox = findBox_(fid, mdiaBox.contentStart, mdiaBox.contentEnd, 'minf') ;
  if isempty(minfBox) ,
    return
  end
  stblBox = findBox_(fid, minfBox.contentStart, minfBox.contentEnd, 'stbl') ;
  if isempty(stblBox) ,
    return
  end
  count = sampleCountFromStbl_(fid, stblBox.contentStart, stblBox.contentEnd) ;
end  % function

function count = sampleCountFromStbl_(fid, stblStart, stblEnd)
% Return the number of samples described by the sample table box spanning
% [stblStart, stblEnd), or [] if it cannot be determined.  Prefers the stsz (or
% stz2) sample_count field, falling back to summing the stts sample counts.
  count = [] ;
  stszBox = findBox_(fid, stblStart, stblEnd, 'stsz') ;
  if ~isempty(stszBox) ,
    % stsz is a full box: version+flags (4 bytes), sample_size (4 bytes), then
    % sample_count.
    fseek(fid, stszBox.contentStart + 8, 'bof') ;
    count = fread(fid, 1, 'uint32') ;
    return
  end
  stz2Box = findBox_(fid, stblStart, stblEnd, 'stz2') ;
  if ~isempty(stz2Box) ,
    % stz2 is a full box: version+flags (4 bytes), reserved+field_size (4
    % bytes), then sample_count.
    fseek(fid, stz2Box.contentStart + 8, 'bof') ;
    count = fread(fid, 1, 'uint32') ;
    return
  end
  sttsBox = findBox_(fid, stblStart, stblEnd, 'stts') ;
  if ~isempty(sttsBox) ,
    % stts is a full box: version+flags (4 bytes), entry_count (4 bytes), then
    % entry_count pairs of (sample_count, sample_delta).
    fseek(fid, sttsBox.contentStart + 4, 'bof') ;
    entryCount = fread(fid, 1, 'uint32') ;
    if isempty(entryCount) || entryCount == 0 ,
      return
    end
    entries = fread(fid, [2 entryCount], 'uint32') ;
    count = sum(entries(1, :)) ;
    return
  end
end  % function
