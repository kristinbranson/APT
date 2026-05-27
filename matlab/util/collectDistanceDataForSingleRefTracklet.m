function [frameIndexFromValidFrameIndex, ...
          trackletIndex, ...
          targetIndex, ...
          distanceFromValidFrameIndex] = ...
    collectDistanceDataForSingleRefTracklet(refTrkFile, ...
                                            refTrackletIndex, ...
                                            testTrackletsAtFrame, ...
                                            centroidMatchThreshold)
% Gather per-frame max-landmark-distance data for one reference tracklet,
% using a precomputed per-frame index of test tracklets.
%
% For each frame in the reference tracklet, find the test tracklet whose
% centroid (at that frame) is closest to the reference centroid, accept
% it as a match if that centroid distance is less than
% centroidMatchThreshold, then compute the max distance over landmarks
% between the two tracklets at that frame.  Frames with no matching test
% tracklet are dropped (this mirrors the way the confidence pipeline
% drops NaN-confidence frames).
%
% Inputs:
%   refTrkFile -- a TrkFile for the reference tracker
%   refTrackletIndex -- scalar index into refTrkFile
%   testTrackletsAtFrame -- cell array of length nframes; cell f contains
%     a struct array with fields trackletIndex (scalar), centroid (2x1),
%     and xy (npt x 2) for each test tracklet that exists at frame f
%   centroidMatchThreshold -- scalar pixel distance threshold for the
%     centroid match
%
% Outputs:
%   frameIndexFromValidFrameIndex -- movie-frame indices for frames that
%     matched a test tracklet (column vector)
%   trackletIndex -- scalar, == refTrackletIndex (for downstream
%     bookkeeping)
%   targetIndex -- scalar target index for this tracklet
%   distanceFromValidFrameIndex -- max landmark distance at each valid
%     frame (column vector, parallel to frameIndexFromValidFrameIndex)

% Read the ref tracklet's data.
[refXyByLandmarkAndAxisAndFrame, ~, refFrameIndices] = ...
  refTrkFile.getPTrkTgt(refTrackletIndex) ;

% Get the target index for this ref tracklet up front, since it is
% returned along every early-return path.
targetIndex = refTrkFile.pTrkiTgt(refTrackletIndex) ;
trackletIndex = refTrackletIndex ;

% Early return if the ref tracklet has no data.
if isempty(refFrameIndices) || isempty(refXyByLandmarkAndAxisAndFrame)
  frameIndexFromValidFrameIndex = zeros(0, 1) ;
  distanceFromValidFrameIndex = zeros(0, 1) ;
  return
end

% refXy is [npt x 2 x numfrm].  Compute per-frame centroid (mean over
% landmarks, ignoring NaN landmarks).
refCentroidByAxisAndFrame = ...
  squeeze(mean(refXyByLandmarkAndAxisAndFrame, 1, 'omitnan')) ;  % [2 x numfrm]
% squeeze on a [1 x 2 x N] array gives [2 x N]; safe for numfrm >= 2.
% Handle the numfrm == 1 case where squeeze collapses too far.
if size(refXyByLandmarkAndAxisAndFrame, 3) == 1
  refCentroidByAxisAndFrame = refCentroidByAxisAndFrame(:) ;  % [2 x 1]
end

refFrameCount = numel(refFrameIndices) ;
isValidFromRefFrameIndex = false(refFrameCount, 1) ;
distanceFromRefFrameIndex = nan(refFrameCount, 1) ;
nframes = numel(testTrackletsAtFrame) ;

for refLocalFrameIndex = 1 : refFrameCount
  movieFrame = refFrameIndices(refLocalFrameIndex) ;
  if movieFrame < 1 || movieFrame > nframes
    continue
  end
  refCentroid = refCentroidByAxisAndFrame(:, refLocalFrameIndex) ;  % [2 x 1]
  if any(~isfinite(refCentroid))
    % Ref tracklet has no usable position at this frame (e.g. all
    % landmarks NaN).  Skip.
    continue
  end
  candidates = testTrackletsAtFrame{movieFrame} ;
  if isempty(candidates)
    continue
  end
  candidateCount = numel(candidates) ;
  centroidDistanceFromCandidateIndex = nan(candidateCount, 1) ;
  for candidateIndex = 1 : candidateCount
    candidateCentroid = candidates(candidateIndex).centroid ;
    centroidDistanceFromCandidateIndex(candidateIndex) = ...
      norm(candidateCentroid - refCentroid) ;
  end
  [bestCentroidDistance, bestCandidateIndex] = min(centroidDistanceFromCandidateIndex) ;
  if ~isfinite(bestCentroidDistance) || bestCentroidDistance >= centroidMatchThreshold
    continue
  end
  % Got a match -- compute per-landmark distances.
  refXyAtFrame = refXyByLandmarkAndAxisAndFrame(:, :, refLocalFrameIndex) ;  % [npt x 2]
  testXyAtFrame = candidates(bestCandidateIndex).xy ;  % [npt x 2]
  landmarkOffsets = testXyAtFrame - refXyAtFrame ;  % [npt x 2]
  landmarkDistances = sqrt(sum(landmarkOffsets .^ 2, 2)) ;  % [npt x 1]
  maxLandmarkDistance = max(landmarkDistances, [], 'omitnan') ;
  if isempty(maxLandmarkDistance) || ~isfinite(maxLandmarkDistance)
    continue
  end
  isValidFromRefFrameIndex(refLocalFrameIndex) = true ;
  distanceFromRefFrameIndex(refLocalFrameIndex) = maxLandmarkDistance ;
end

frameIndexFromValidFrameIndex = refFrameIndices(isValidFromRefFrameIndex) ;
frameIndexFromValidFrameIndex = frameIndexFromValidFrameIndex(:) ;
distanceFromValidFrameIndex = distanceFromRefFrameIndex(isValidFromRefFrameIndex) ;

end  % function
