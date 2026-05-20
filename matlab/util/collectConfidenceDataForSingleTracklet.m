function [frameIndexFromValidFrameIndex, ...
          trackletIndex, ...
          targetIndex, ...
          minConfFromValidFrameIndex, ...
          maxConfFromValidFrameIndex] = ...
    collectConfidenceDataForSingleTracklet(trkFile, trackletIndex)
% Gather flattened confidence-pair data for one tracklet.

% Read the file
[~, ~, frameIndexFromFrameIndexAsRow, rawConfidence] = ...
  trkFile.getPTrkTgt(trackletIndex, 'auxflds', {'pTrkConf'}) ;

% Do an early return if no data
if isempty(frameIndexFromFrameIndexAsRow) || isempty(rawConfidence)
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  minConfFromValidFrameIndex = zeros(0, 1) ;
  maxConfFromValidFrameIndex = zeros(0, 1) ;
  return
end

% Compute min/max confidences across landmark points and across views.
frameIndexFromFrameIndex = frameIndexFromFrameIndexAsRow(:) ;  % [frameCount x 1]
confFromLandmarkIndexFromFrameIndex = ...
  reshape(rawConfidence, size(rawConfidence, 1), size(rawConfidence, 2)) ;  % [labelPointCount x frameCount]
  % rawConfidence is [labelPointCount x frameCount x 1 x 1]
minConfFromFrameIndexAsRow = min(confFromLandmarkIndexFromFrameIndex, [], 1) ;  % [1 x frameCount]
minConfFromFrameIndex = minConfFromFrameIndexAsRow(:) ;  % [frameCount x 1]
maxConfFromFrameIndexAsRow = max(confFromLandmarkIndexFromFrameIndex, [], 1) ;  % [1 x frameCount]
maxConfFromFrameIndex = maxConfFromFrameIndexAsRow(:) ;  % [frameCount x 1]

% Find NaN confidence frames.
isValidFromFrameIndex = isfinite(minConfFromFrameIndex) ;

% Do an early return if no valid frames.
validFrameCount = sum(isValidFromFrameIndex) ;
if validFrameCount == 0
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  minConfFromValidFrameIndex = zeros(0, 1) ;
  maxConfFromValidFrameIndex = zeros(0, 1) ;
  return
end

% Filter out NaN confidence frames.
frameIndexFromValidFrameIndex = frameIndexFromFrameIndex(isValidFromFrameIndex) ;
minConfFromValidFrameIndex = minConfFromFrameIndex(isValidFromFrameIndex) ;
maxConfFromValidFrameIndex = maxConfFromFrameIndex(isValidFromFrameIndex) ;

% Get the target index for this tracklet.
targetIndex = trkFile.pTrkiTgt(trackletIndex) ;

end  % function
