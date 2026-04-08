function [frameIndexFromValidFrameIndex, ...
          trackletIndexFromValidFrameIndex, ...
          targetIndexFromValidFrameIndex, ...
          minConfFromValidFrameIndex, ...
          maxConfFromValidFrameIndex] = ...
    collectConfidenceDataForSingleTracklet(trkFile, trackletIndex)
% Gather flattened confidence-pair data for one tracklet.

% Read the file
[~, ~, frameIndexFromFrameIndexAsRow, aux] = ...
  trkFile.getPTrkTgt(trackletIndex, 'auxflds', {'pTrkConf'}) ;

% Do an early return if no data
if isempty(frameIndexFromFrameIndexAsRow) || isempty(aux)
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  trackletIndexFromValidFrameIndex = zeros(0, 1) ;
  targetIndexFromValidFrameIndex = zeros(0, 1) ;
  minConfFromValidFrameIndex = zeros(0, 1) ;
  maxConfFromValidFrameIndex = zeros(0, 1) ;
  return
end

% Compute min/max confidences across landmark points and across views.
frameIndexFromFrameIndex = frameIndexFromFrameIndexAsRow(:) ;  % [frameCount x 1]
confFromPointIndexFromFrameIndex = reshape(aux, size(aux, 1), size(aux, 2)) ;  % [labelPointCount x frameCount]
  % aux is [labelPointCount x frameCount x 1 x 1]
minConfFromFrameIndexAsRow = min(confFromPointIndexFromFrameIndex, [], 1) ;  % [1 x frameCount]
minConfFromFrameIndex = minConfFromFrameIndexAsRow(:) ;  % [frameCount x 1]
maxConfFromFrameIndexAsRow = max(confFromPointIndexFromFrameIndex, [], 1) ;  % [1 x frameCount]
maxConfFromFrameIndex = maxConfFromFrameIndexAsRow(:) ;  % [frameCount x 1]

% Find NaN confidence frames.
isValidFromFrameIndex = isfinite(minConfFromFrameIndex) ;

% Do an early return if no valid frames.
validFrameCount = sum(isValidFromFrameIndex) ;
if validFrameCount == 0
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  trackletIndexFromValidFrameIndex = zeros(0, 1) ;
  targetIndexFromValidFrameIndex = zeros(0, 1) ;
  minConfFromValidFrameIndex = zeros(0, 1) ;
  maxConfFromValidFrameIndex = zeros(0, 1) ;
  return
end

% Filter out NaN confidence frames.
frameIndexFromValidFrameIndex = frameIndexFromFrameIndex(isValidFromFrameIndex) ;
minConfFromValidFrameIndex = minConfFromFrameIndex(isValidFromFrameIndex) ;
maxConfFromValidFrameIndex = maxConfFromFrameIndex(isValidFromFrameIndex) ;

% Make arrays with the tracketIndex and 
targetIndex = trkFile.pTrkiTgt(trackletIndex) ;
trackletIndexFromValidFrameIndex = repmat(trackletIndex, validFrameCount, 1) ;
targetIndexFromValidFrameIndex = repmat(targetIndex, validFrameCount, 1) ;

end  % function
