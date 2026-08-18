function [im, timestamp] = mp4_seek_read_frame(readerobj, index, f)
% Read frame f (1-based) from an already-open VideoReader, using the seek index
% from mp4_read_index for fast, frame-accurate random access.
%
% MATLAB's read(readerobj, f) decodes every frame from the start of the file to
% reach frame f, so its cost grows linearly with f (over two minutes for a
% frame ~200k into a large movie).  Instead, this seeks CurrentTime to the exact
% keyframe at or before f -- using the container's true sample timestamps, not
% VideoReader's rounded FrameRate -- and then reads forward a handful of frames
% to f.  Each read frame is identified by the CurrentTime value just before the
% read (readFrame consumes the frame whose decode time equals CurrentTime and
% advances CurrentTime past it), so the mapping is exact regardless of any
% frame-boundary rounding in readFrame.  If anything unexpected happens, it
% falls back to the slow-but-exact read(readerobj, f).
%
% index must be from mp4_read_index for this movie, with isFastSeekable true.

  targetSample = f - 1 ;  % 0-based

  if ~index.isFastSeekable || targetSample < 0 || targetSample >= index.nframes ,
    [im, timestamp] = slowReadFrame_(readerobj, index, f) ;
    return
  end

  keyframeSample = keyframeAtOrBefore_(index, targetSample) ;
  readerobj.CurrentTime = index.sampleStartTicks(keyframeSample + 1) / index.timescale ;

  maxStepCount = (targetSample - keyframeSample) + 16 ;  % a little slack
  for stepIndex = 1 : maxStepCount ,
    currentTick = round(readerobj.CurrentTime * index.timescale) ;
    currentSample = tickToSample_(index, currentTick) ;
    if currentSample > targetSample ,
      break  % overshot; fall back below
    end
    im = readFrame(readerobj) ;
    if currentSample == targetSample ,
      timestamp = index.sampleStartTicks(targetSample + 1) / index.timescale ;
      return
    end
  end

  % Stepping did not land on the target frame; use the slow exact read.
  [im, timestamp] = slowReadFrame_(readerobj, index, f) ;
end  % function

function keyframeSample = keyframeAtOrBefore_(index, targetSample)
% Return the 0-based index of the sync sample (keyframe) at or before
% targetSample.  With no sync-sample table every sample is a keyframe.
  if isempty(index.syncSamples) ,
    keyframeSample = targetSample ;
    return
  end
  position = find(index.syncSamples <= targetSample, 1, 'last') ;
  if isempty(position) ,
    keyframeSample = 0 ;
  else
    keyframeSample = index.syncSamples(position) ;
  end
end  % function

function sample = tickToSample_(index, tick)
% Return the 0-based sample whose decode interval contains the given tick.
  if index.isUniform && ~isempty(index.uniformDelta) && index.uniformDelta > 0 ,
    sample = round(tick / index.uniformDelta) ;
  else
    sample = find(index.sampleStartTicks <= tick + 0.5, 1, 'last') - 1 ;
    if isempty(sample) ,
      sample = 0 ;
    end
  end
  if sample < 0 ,
    sample = 0 ;
  elseif sample > index.nframes - 1 ,
    sample = index.nframes - 1 ;
  end
end  % function

function [im, timestamp] = slowReadFrame_(readerobj, index, f)
% Read frame f the slow-but-exact way, via read(readerobj, f).
  im = read(readerobj, f) ;
  if index.isFastSeekable && f >= 1 && f <= index.nframes ,
    timestamp = index.sampleStartTicks(f) / index.timescale ;
  else
    timestamp = (f - 1) / readerobj.FrameRate ;
  end
end  % function
