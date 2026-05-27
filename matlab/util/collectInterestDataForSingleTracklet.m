function [frameIndexFromValidFrameIndex, ...
          trackletIndex, ...
          targetIndex, ...
          interestFromValidFrameIndex] = ...
    collectInterestDataForSingleTracklet(trkFile, trackletIndex)
% Gather per-frame max-interest data for one tracklet.
%
% Interest is the negation of confidence: a higher interest
% means a less confident, more uncertain frame.  The per-frame
% interest is taken over all landmark points (and any views) as
% the max, so the most-interesting (least confident) landmark dictates
% the frame's value.  This is equivalent to taking the min over
% per-landmark confidence and then negating.
%
% Frames with NaN confidence (and thus NaN interest) are filtered
% out.

% Read the file.
[~, ~, frameIndexFromFrameIndexAsRow, rawConfidence] = ...
  trkFile.getPTrkTgt(trackletIndex, 'auxflds', {'pTrkConf'}) ;

% Get the target index for this tracklet up front, since it is returned
% along every early-return path.
targetIndex = trkFile.pTrkiTgt(trackletIndex) ;

% Do an early return if no data.
if isempty(frameIndexFromFrameIndexAsRow) || isempty(rawConfidence)
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  interestFromValidFrameIndex = zeros(0, 1) ;
  return
end

% Translate per-landmark confidence to interest.  rawConfidence is
% [labelPointCount x frameCount x 1 x 1].
frameIndexFromFrameIndex = frameIndexFromFrameIndexAsRow(:) ;  % [frameCount x 1]
confidenceFromLandmarkIndexFromFrameIndex = ...
  reshape(rawConfidence, size(rawConfidence, 1), size(rawConfidence, 2)) ;  % [labelPointCount x frameCount]
interestFromLandmarkIndexFromFrameIndex = ...
  -confidenceFromLandmarkIndexFromFrameIndex ;

% Per frame, take the max interest across landmarks.
interestFromFrameIndexAsRow = ...
  max(interestFromLandmarkIndexFromFrameIndex, [], 1) ;  % [1 x frameCount]
interestFromFrameIndex = interestFromFrameIndexAsRow(:) ;  % [frameCount x 1]

% Find frames whose interest is finite (i.e. whose confidence was
% not NaN).
isValidFromFrameIndex = isfinite(interestFromFrameIndex) ;

% Do an early return if no valid frames.
validFrameCount = sum(isValidFromFrameIndex) ;
if validFrameCount == 0
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  interestFromValidFrameIndex = zeros(0, 1) ;
  return
end

% Filter out NaN-interest frames.
frameIndexFromValidFrameIndex = frameIndexFromFrameIndex(isValidFromFrameIndex) ;
interestFromValidFrameIndex = interestFromFrameIndex(isValidFromFrameIndex) ;

end  % function
