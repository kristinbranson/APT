function [startFrameFromBoutIndex, endFrameFromBoutIndex, extremeFrameFromBoutIndex, ...
         trackletIndexFromBoutIndex, targetIndexFromBoutIndex, extremeConfidenceFromBoutIndex, ...
         overallMinConfidence, overallMaxConfidence] = ...
  confidenceBoutsFromTrkFile(trkFile, threshold, isConfidenceLackThereof)
% Find bouts of consecutive frames whose confidence passes a threshold.
%
% A bout is a maximal contiguous run of consecutive frames (within a single
% tracklet) where confidence stays below (or above, in lack-thereof mode)
% the threshold.  Each bout is represented by the frame achieving the
% extreme confidence within the run.
%
% Returns empty column vectors and NaN extremes when there are no bouts.

% Gather per-frame confidence data from all tracklets.
frameIndexFromPairIndex = [] ;
trackletIndexFromPairIndex = [] ;
targetIndexFromPairIndex = [] ;
minConfFromPairIndex = [] ;
maxConfFromPairIndex = [] ;

trackletCount = trkFile.ntracklets ;
for trackletIndex = 1 : trackletCount
  [xy, ~, fr, aux] = trkFile.getPTrkTgt(trackletIndex, 'auxflds', {'pTrkConf'}) ;
  if isempty(xy) || isempty(aux)
    continue
  end
  % aux is [npt x numfrm x 1 x 1]
  confPerPointAndFrame = reshape(aux, size(aux, 1), size(aux, 2)) ;  % [npt x numfrm]
  minConfPerFrame = min(confPerPointAndFrame, [], 1) ;  % [1 x numfrm]
  minConfPerFrame = minConfPerFrame(:) ;  % [numfrm x 1]
  maxConfPerFrame = max(confPerPointAndFrame, [], 1) ;  % [1 x numfrm]
  maxConfPerFrame = maxConfPerFrame(:) ;  % [numfrm x 1]
  fr = fr(:) ;  % [numfrm x 1]

  % Filter out NaN confidence frames
  isFinite = isfinite(minConfPerFrame) ;
  fr = fr(isFinite) ;
  minConfPerFrame = minConfPerFrame(isFinite) ;
  maxConfPerFrame = maxConfPerFrame(isFinite) ;
  if isempty(fr)
    continue
  end

  targetIndex = trkFile.pTrkiTgt(trackletIndex) ;

  frameCount = numel(fr) ;
  frameIndexFromPairIndex = [frameIndexFromPairIndex ; fr] ;  %#ok<AGROW>
  trackletIndexFromPairIndex = [trackletIndexFromPairIndex ; repmat(trackletIndex, frameCount, 1)] ;  %#ok<AGROW>
  targetIndexFromPairIndex = [targetIndexFromPairIndex ; repmat(targetIndex, frameCount, 1)] ;  %#ok<AGROW>
  minConfFromPairIndex = [minConfFromPairIndex ; minConfPerFrame] ;  %#ok<AGROW>
  maxConfFromPairIndex = [maxConfFromPairIndex ; maxConfPerFrame] ;  %#ok<AGROW>
end

if isempty(frameIndexFromPairIndex)
  startFrameFromBoutIndex = zeros(0, 1) ;
  endFrameFromBoutIndex = zeros(0, 1) ;
  extremeFrameFromBoutIndex = zeros(0, 1) ;
  trackletIndexFromBoutIndex = zeros(0, 1) ;
  targetIndexFromBoutIndex = zeros(0, 1) ;
  extremeConfidenceFromBoutIndex = zeros(0, 1) ;
  overallMinConfidence = nan ;
  overallMaxConfidence = nan ;
  return
end

% Record overall extremes before filtering.
overallMinConfidence = min(minConfFromPairIndex) ;
overallMaxConfidence = max(maxConfFromPairIndex) ;

% Build bouts: contiguous runs of consecutive frames (within a single
% tracklet) that pass the confidence threshold.
uniqueTracklets = unique(trackletIndexFromPairIndex) ;
trackletGroupCount = numel(uniqueTracklets) ;
startFrameCell = cell(trackletGroupCount, 1) ;
endFrameCell = cell(trackletGroupCount, 1) ;
extremeFrameCell = cell(trackletGroupCount, 1) ;
trackletCell = cell(trackletGroupCount, 1) ;
targetCell = cell(trackletGroupCount, 1) ;
extremeConfCell = cell(trackletGroupCount, 1) ;

for iU = 1 : trackletGroupCount
  thisTrackletIndex = uniqueTracklets(iU) ;
  isThisTrackletFromPairIndex = (trackletIndexFromPairIndex == thisTrackletIndex) ;
  tltFrames = frameIndexFromPairIndex(isThisTrackletFromPairIndex) ;
  tltMinConf = minConfFromPairIndex(isThisTrackletFromPairIndex) ;
  tltMaxConf = maxConfFromPairIndex(isThisTrackletFromPairIndex) ;
  tltTarget = targetIndexFromPairIndex(find(isThisTrackletFromPairIndex, 1)) ;

  [tltStartFrames, tltEndFrames, tltExtremeFrames, tltExtremeConfs] = ...
    confidenceBoutsForSingleTracklet(tltFrames, tltMinConf, tltMaxConf, threshold, isConfidenceLackThereof) ;

  boutCount = numel(tltStartFrames) ;
  startFrameCell{iU} = tltStartFrames ;
  endFrameCell{iU} = tltEndFrames ;
  extremeFrameCell{iU} = tltExtremeFrames ;
  trackletCell{iU} = repmat(thisTrackletIndex, boutCount, 1) ;
  targetCell{iU} = repmat(tltTarget, boutCount, 1) ;
  extremeConfCell{iU} = tltExtremeConfs ;
end  % for

boutStartFrames = vertcat(startFrameCell{:}) ;
boutExtremeConfs = vertcat(extremeConfCell{:}) ;

if isempty(boutStartFrames)
  startFrameFromBoutIndex = zeros(0, 1) ;
  endFrameFromBoutIndex = zeros(0, 1) ;
  extremeFrameFromBoutIndex = zeros(0, 1) ;
  trackletIndexFromBoutIndex = zeros(0, 1) ;
  targetIndexFromBoutIndex = zeros(0, 1) ;
  extremeConfidenceFromBoutIndex = zeros(0, 1) ;
  return
end

% Sort bouts by extreme confidence.
if isConfidenceLackThereof
  [~, sortOrder] = sort(boutExtremeConfs, 'descend') ;
else
  [~, sortOrder] = sort(boutExtremeConfs, 'ascend') ;
end

startFrameFromBoutIndex = boutStartFrames(sortOrder) ;
endFrameFromBoutIndex = vertcat(endFrameCell{:}) ;
endFrameFromBoutIndex = endFrameFromBoutIndex(sortOrder) ;
extremeFrameFromBoutIndex = vertcat(extremeFrameCell{:}) ;
extremeFrameFromBoutIndex = extremeFrameFromBoutIndex(sortOrder) ;
trackletIndexFromBoutIndex = vertcat(trackletCell{:}) ;
trackletIndexFromBoutIndex = trackletIndexFromBoutIndex(sortOrder) ;
targetIndexFromBoutIndex = vertcat(targetCell{:}) ;
targetIndexFromBoutIndex = targetIndexFromBoutIndex(sortOrder) ;
extremeConfidenceFromBoutIndex = boutExtremeConfs(sortOrder) ;

end  % function
