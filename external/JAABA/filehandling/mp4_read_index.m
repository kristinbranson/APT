function index = mp4_read_index(filename)
% Read a seek index for an ISO base-media file (.mp4/.mov/.m4v) by traversing
% its container metadata, without decoding any frames.  Returns a struct with:
%
%   nframes           - number of frames (samples) in the first video track
%   timescale         - media timescale, in ticks per second
%   sampleStartTicks  - (nframes+1)-by-1 double; sampleStartTicks(i+1) is the
%                       decode start time of 0-based sample i, in ticks.  The
%                       final element is the end time of the last sample.
%   isUniform         - true if every sample has the same duration
%   uniformDelta      - the common per-sample duration in ticks (when isUniform)
%   syncSamples       - sorted 0-based indices of sync samples (keyframes), or
%                       [] meaning every sample is a sync sample
%   isFastSeekable    - true if the track can be seeked frame-accurately from
%                       these tables alone: a video track with a valid stts and
%                       no composition offsets (ctts) or edit list (elst).  When
%                       false, only nframes/timescale are guaranteed meaningful.
%
% This is fast even for very large files because only box headers and a few
% metadata boxes are read; the media data (mdat) is never touched.  Errors if
% the file cannot be opened, cannot be parsed as ISO base-media, or has no
% video track with a frame count.  Callers wanting a graceful fallback should
% wrap the call in try/catch.

  fid = fopen(filename, 'r', 'ieee-be') ;
  if fid < 0 ,
    error('mp4_read_index:cannotOpenFile', 'Could not open file %s', filename) ;
  end
  cleaner = onCleanup(@()(fclose(fid))) ;
  fileSize = fileSize_(fid) ;
  moovBox = findBox_(fid, 0, fileSize, 'moov') ;
  if isempty(moovBox) ,
    error('mp4_read_index:noMoovBox', 'No moov box found in %s', filename) ;
  end
  trakBox = findVideoTrakBox_(fid, moovBox.contentStart, moovBox.contentEnd) ;
  if isempty(trakBox) ,
    error('mp4_read_index:noVideoTrack', 'No video track found in %s', filename) ;
  end
  index = readTrakIndex_(fid, trakBox.contentStart, trakBox.contentEnd) ;
  if isempty(index) || isempty(index.nframes) ,
    error('mp4_read_index:noSampleTable', 'Could not read the sample table in %s', filename) ;
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

function trakBox = findVideoTrakBox_(fid, moovStart, moovEnd)
% Return the first trak box that is a video track within the moov box spanning
% [moovStart, moovEnd), or [] if there is no video track.
  trakBox = [] ;
  pos = moovStart ;
  while pos + 8 <= moovEnd ,
    thisBox = readBoxHeader_(fid, pos, moovEnd) ;
    if isempty(thisBox) ,
      return
    end
    if strcmp(thisBox.type, 'trak') && isVideoTrak_(fid, thisBox.contentStart, thisBox.contentEnd) ,
      trakBox = thisBox ;
      return
    end
    pos = thisBox.contentEnd ;
  end
end  % function

function result = isVideoTrak_(fid, trakStart, trakEnd)
% Return true if the given trak box is a video track (mdia/hdlr handler 'vide').
  result = false ;
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
  result = (numel(handlerType) == 4) && strcmp(handlerType, 'vide') ;
end  % function

function index = readTrakIndex_(fid, trakStart, trakEnd)
% Read the seek index from a video trak box spanning [trakStart, trakEnd).
% Returns the index struct, or [] if the sample table cannot be read.
  index = [] ;
  mdiaBox = findBox_(fid, trakStart, trakEnd, 'mdia') ;
  if isempty(mdiaBox) ,
    return
  end
  timescale = readTimescale_(fid, mdiaBox.contentStart, mdiaBox.contentEnd) ;
  if isempty(timescale) ,
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

  nframes = readSampleCount_(fid, stblBox.contentStart, stblBox.contentEnd) ;
  if isempty(nframes) ,
    return
  end

  % Detect composition offsets (B-frame reordering) and non-identity edit lists,
  % which would make frame-index-to-time mapping from stts alone unreliable.  A
  % single identity edit (media_time 0, rate 1.0) does not shift the timeline and
  % is treated as benign.
  hasCtts = ~isempty(findBox_(fid, stblBox.contentStart, stblBox.contentEnd, 'ctts')) ;
  hasElst = isProblematicEditList_(fid, trakStart, trakEnd) ;

  [sampleStartTicks, isUniform, uniformDelta, sttsSampleCount] = ...
    readSampleTimes_(fid, stblBox.contentStart, stblBox.contentEnd) ;
  syncSamples = readSyncSamples_(fid, stblBox.contentStart, stblBox.contentEnd) ;

  index = struct() ;
  index.nframes = nframes ;
  index.timescale = timescale ;
  index.sampleStartTicks = sampleStartTicks ;
  index.isUniform = isUniform ;
  index.uniformDelta = uniformDelta ;
  index.syncSamples = syncSamples ;
  index.isFastSeekable = ~hasCtts && ~hasElst && ~isempty(sampleStartTicks) && ...
                         isequal(sttsSampleCount, nframes) ;
end  % function

function timescale = readTimescale_(fid, mdiaStart, mdiaEnd)
% Return the media timescale from the mdhd box, in ticks per second, or [].
  timescale = [] ;
  mdhdBox = findBox_(fid, mdiaStart, mdiaEnd, 'mdhd') ;
  if isempty(mdhdBox) ,
    return
  end
  fseek(fid, mdhdBox.contentStart, 'bof') ;
  version = fread(fid, 1, 'uint8') ;
  fread(fid, 3, 'uint8') ;  % flags
  if version == 1 ,
    fread(fid, 2, 'uint64') ;  % creation_time, modification_time (64-bit)
    timescale = fread(fid, 1, 'uint32') ;
  else
    fread(fid, 2, 'uint32') ;  % creation_time, modification_time (32-bit)
    timescale = fread(fid, 1, 'uint32') ;
  end
  if isempty(timescale) || timescale == 0 ,
    timescale = [] ;
  end
end  % function

function nframes = readSampleCount_(fid, stblStart, stblEnd)
% Return the number of samples from the stsz (or stz2) box, or [].
  nframes = [] ;
  stszBox = findBox_(fid, stblStart, stblEnd, 'stsz') ;
  if ~isempty(stszBox) ,
    % stsz full box: version+flags (4), sample_size (4), then sample_count.
    fseek(fid, stszBox.contentStart + 8, 'bof') ;
    nframes = fread(fid, 1, 'uint32') ;
    return
  end
  stz2Box = findBox_(fid, stblStart, stblEnd, 'stz2') ;
  if ~isempty(stz2Box) ,
    % stz2 full box: version+flags (4), reserved+field_size (4), sample_count.
    fseek(fid, stz2Box.contentStart + 8, 'bof') ;
    nframes = fread(fid, 1, 'uint32') ;
    return
  end
end  % function

function [sampleStartTicks, isUniform, uniformDelta, sampleCount] = readSampleTimes_(fid, stblStart, stblEnd)
% Read the stts box and return the cumulative sample start times (in ticks),
% whether the per-sample durations are all equal, that common duration, and the
% total number of samples described.  Returns empties if stts is absent.
  sampleStartTicks = [] ;
  isUniform = false ;
  uniformDelta = [] ;
  sampleCount = [] ;
  sttsBox = findBox_(fid, stblStart, stblEnd, 'stts') ;
  if isempty(sttsBox) ,
    return
  end
  % stts full box: version+flags (4), entry_count (4), then entry_count pairs
  % of (sample_count, sample_delta).
  fseek(fid, sttsBox.contentStart + 4, 'bof') ;
  entryCount = fread(fid, 1, 'uint32') ;
  if isempty(entryCount) || entryCount == 0 ,
    return
  end
  entries = fread(fid, [2 entryCount], 'uint32') ;
  counts = entries(1, :)' ;
  deltas = entries(2, :)' ;
  sampleCount = sum(counts) ;

  sampleStartTicks = zeros(sampleCount + 1, 1) ;
  tick = 0 ;
  writtenCount = 0 ;
  for entryIndex = 1 : entryCount ,
    thisCount = counts(entryIndex) ;
    thisDelta = deltas(entryIndex) ;
    if thisCount > 0 ,
      sampleStartTicks(writtenCount + 2 : writtenCount + thisCount + 1) = ...
        tick + (1 : thisCount)' * thisDelta ;
      tick = tick + thisCount * thisDelta ;
      writtenCount = writtenCount + thisCount ;
    end
  end

  isUniform = (entryCount == 1) || (numel(unique(deltas(counts > 0))) == 1) ;
  if isUniform ,
    nonEmptyDeltas = deltas(counts > 0) ;
    if ~isempty(nonEmptyDeltas) ,
      uniformDelta = nonEmptyDeltas(1) ;
    end
  end
end  % function

function result = isProblematicEditList_(fid, trakStart, trakEnd)
% Return true if the track has an edit list that shifts or reorders the media
% timeline.  A single identity edit (one entry with media_time 0 and rate 1.0)
% is treated as benign, since it leaves sample times unchanged.
  result = false ;
  edtsBox = findBox_(fid, trakStart, trakEnd, 'edts') ;
  if isempty(edtsBox) ,
    return
  end
  elstBox = findBox_(fid, edtsBox.contentStart, edtsBox.contentEnd, 'elst') ;
  if isempty(elstBox) ,
    return
  end
  fseek(fid, elstBox.contentStart, 'bof') ;
  version = fread(fid, 1, 'uint8') ;
  fread(fid, 3, 'uint8') ;  % flags
  entryCount = fread(fid, 1, 'uint32') ;
  if isempty(entryCount) || entryCount ~= 1 ,
    result = true ;  % no entries, or multiple segments: not a plain identity edit
    return
  end
  if version == 1 ,
    fread(fid, 1, 'uint64') ;         % segment_duration
    mediaTime = fread(fid, 1, 'int64') ;
  else
    fread(fid, 1, 'uint32') ;         % segment_duration
    mediaTime = fread(fid, 1, 'int32') ;
  end
  mediaRateInteger = fread(fid, 1, 'int16') ;
  fread(fid, 1, 'int16') ;            % media_rate_fraction
  isIdentityEdit = ~isempty(mediaTime) && (mediaTime == 0) && (mediaRateInteger == 1) ;
  result = ~isIdentityEdit ;
end  % function

function syncSamples = readSyncSamples_(fid, stblStart, stblEnd)
% Return the 0-based indices of sync samples (keyframes) from the stss box.
% Returns [] if there is no stss box, which by convention means every sample is
% a sync sample.
  syncSamples = [] ;
  stssBox = findBox_(fid, stblStart, stblEnd, 'stss') ;
  if isempty(stssBox) ,
    return
  end
  % stss full box: version+flags (4), entry_count (4), then entry_count 1-based
  % sample numbers.
  fseek(fid, stssBox.contentStart + 4, 'bof') ;
  entryCount = fread(fid, 1, 'uint32') ;
  if isempty(entryCount) || entryCount == 0 ,
    return
  end
  oneBasedSamples = fread(fid, entryCount, 'uint32') ;
  syncSamples = oneBasedSamples - 1 ;
end  % function
