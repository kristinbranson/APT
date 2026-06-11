function [distanceFromFrameIndexAndTrackletIndex, ...
          testTrackletIndexFromFrameIndexAndTrackletIndex, ...
          refPoseCountFromFrameIndex, ...
          unmatchedRefCountFromFrameIndex] = ...
  perFrameTrackerMatch(refTrkFile, ...
                       testTrkFile, ...
                       nframes, ...
                       matchDistanceThreshold)
% Hungarian-match a reference tracker's poses to a test tracker's poses,
% frame by frame, for the current movie.
%
% At each frame, the ref poses and test poses present at that frame are
% put into one-to-one correspondence by Hungarian matching (matchpairs),
% as in computeMAErr().  The assignment cost for a (ref, test) pose pair
% is the mean over landmarks of the landmark-to-landmark distance, and
% pairs whose cost is matchDistanceThreshold or more are left unmatched.
%
% Inputs:
%   refTrkFile -- TrkFile for the reference tracker (current movie)
%   testTrkFile -- TrkFile for the test tracker (current movie)
%   nframes -- number of frames in the current movie
%   matchDistanceThreshold -- mean-over-landmarks pixel distance at or
%     above which a ref pose and a test pose are considered unrelated and
%     left unmatched
%
% Outputs (rows indexed by movie frame, columns by ref-tracklet index):
%   distanceFromFrameIndexAndTrackletIndex -- [nframes x refTrackletCount]
%     the max-over-landmarks ref-to-test distance for each matched ref
%     tracklet at each frame; NaN where the ref tracklet has no matched
%     test pose (or is absent)
%   testTrackletIndexFromFrameIndexAndTrackletIndex --
%     [nframes x refTrackletCount] the test-tracklet index matched to each
%     ref tracklet at each frame; NaN where unmatched
%   refPoseCountFromFrameIndex -- [nframes x 1] number of ref poses present
%     at each frame
%   unmatchedRefCountFromFrameIndex -- [nframes x 1] number of ref poses
%     present at each frame that the matcher left without a test partner

% Precompute per-frame pose data for both trackers.
refPosesFromFrameIndex = buildPosesFromFrameIndex_(refTrkFile, nframes) ;
testPosesFromFrameIndex = buildPosesFromFrameIndex_(testTrkFile, nframes) ;

% At each frame, Hungarian-match the ref poses to the test poses, and
% record the max landmark distance for each matched ref pose.
% matchpairs() leaves a pair unmatched when its cost exceeds twice the
% cost of non-assignment, so this choice of costOfNonAssignment leaves
% pairs with mean landmark distance >= matchDistanceThreshold unmatched.
refTrackletCount = refTrkFile.ntracklets ;
distanceFromFrameIndexAndTrackletIndex = nan(nframes, refTrackletCount) ;
testTrackletIndexFromFrameIndexAndTrackletIndex = nan(nframes, refTrackletCount) ;
refPoseCountFromFrameIndex = zeros(nframes, 1) ;
unmatchedRefCountFromFrameIndex = zeros(nframes, 1) ;
costOfNonAssignment = matchDistanceThreshold / 2 ;
for frameIndex = 1 : nframes
  refPoses = refPosesFromFrameIndex{frameIndex} ;
  testPoses = testPosesFromFrameIndex{frameIndex} ;
  refPoseCount = numel(refPoses) ;
  refPoseCountFromFrameIndex(frameIndex) = refPoseCount ;
  if isempty(refPoses) || isempty(testPoses)
    % With no test poses to match against, every present ref pose is
    % unmatched.  (With no ref poses, the count is trivially zero.)
    unmatchedRefCountFromFrameIndex(frameIndex) = refPoseCount ;
    continue
  end
  % Two distances are computed per (ref, test) pose pair, each answering
  % a different question.  The mean over landmarks ("are these the same
  % animal?") is the matching cost: it represents the pose as a whole and
  % is robust to a single wild landmark, so one bad point can't block an
  % otherwise-obvious correspondence.  The max over landmarks ("how badly
  % do the trackers disagree about this animal?") is the per-frame
  % interest signal used downstream to form bouts: a single flipped or
  % swapped landmark is exactly the kind of disagreement we want to
  % surface, and the mean would dilute it.
  testPoseCount = numel(testPoses) ;
  meanDistanceFromRefAndTestPoseIndex = nan(refPoseCount, testPoseCount) ;
  maxDistanceFromRefAndTestPoseIndex = nan(refPoseCount, testPoseCount) ;
  for refPoseIndex = 1 : refPoseCount
    refXy = refPoses(refPoseIndex).xy ;  % [landmarkCount x 2]
    for testPoseIndex = 1 : testPoseCount
      testXy = testPoses(testPoseIndex).xy ;  % [landmarkCount x 2]
      landmarkDistances = sqrt(sum((testXy - refXy) .^ 2, 2)) ;  % [landmarkCount x 1]
      meanDistanceFromRefAndTestPoseIndex(refPoseIndex, testPoseIndex) = ...
        mean(landmarkDistances, 'omitnan') ;
      maxDistanceFromRefAndTestPoseIndex(refPoseIndex, testPoseIndex) = ...
        max(landmarkDistances, [], 'omitnan') ;
    end
  end
  % matchpairs() does not accept NaN costs.  A NaN mean distance means the
  % two poses share no commonly-valid landmarks, so make such pairs
  % unmatchable.
  costFromRefAndTestPoseIndex = meanDistanceFromRefAndTestPoseIndex ;
  isUnmatchable = ~isfinite(costFromRefAndTestPoseIndex) ;
  costFromRefAndTestPoseIndex(isUnmatchable) = 2 * costOfNonAssignment + 1 ;
  matchedPoseIndexPairs = matchpairs(costFromRefAndTestPoseIndex, costOfNonAssignment) ;
  matchedRefPoseCount = size(matchedPoseIndexPairs, 1) ;
  unmatchedRefCountFromFrameIndex(frameIndex) = refPoseCount - matchedRefPoseCount ;
  for matchIndex = 1 : matchedRefPoseCount
    refPoseIndex = matchedPoseIndexPairs(matchIndex, 1) ;
    testPoseIndex = matchedPoseIndexPairs(matchIndex, 2) ;
    maxLandmarkDistance = maxDistanceFromRefAndTestPoseIndex(refPoseIndex, testPoseIndex) ;
    if ~isfinite(maxLandmarkDistance)
      continue
    end
    trackletIndex = refPoses(refPoseIndex).trackletIndex ;
    distanceFromFrameIndexAndTrackletIndex(frameIndex, trackletIndex) = maxLandmarkDistance ;
    testTrackletIndexFromFrameIndexAndTrackletIndex(frameIndex, trackletIndex) = ...
      testPoses(testPoseIndex).trackletIndex ;
  end
end

end  % function



function posesFromFrameIndex = buildPosesFromFrameIndex_(trkFile, nframes)
% Build a per-frame lookup of the tracklet poses present at each frame.
% Each cell holds a struct array with fields trackletIndex (scalar) and
% xy (landmarkCount x 2).  Poses with no finite landmark are omitted.

emptyEntry = struct('trackletIndex', {}, 'xy', {}) ;
posesFromFrameIndex = repmat({emptyEntry}, nframes, 1) ;

if isempty(trkFile) || trkFile.ntracklets == 0
  return
end

trackletCount = trkFile.ntracklets ;
for trackletIndex = 1 : trackletCount
  [xyByLandmarkAndAxisAndFrame, ~, frameIndices] = ...
    trkFile.getPTrkTgt(trackletIndex) ;
  if isempty(frameIndices) || isempty(xyByLandmarkAndAxisAndFrame)
    continue
  end
  for localFrameIndex = 1 : numel(frameIndices)
    movieFrame = frameIndices(localFrameIndex) ;
    if movieFrame < 1 || movieFrame > nframes
      continue
    end
    xy = xyByLandmarkAndAxisAndFrame(:, :, localFrameIndex) ;  % [landmarkCount x 2]
    if ~any(isfinite(xy(:)))
      continue
    end
    entry = struct('trackletIndex', trackletIndex, 'xy', xy) ;
    posesFromFrameIndex{movieFrame}(end+1) = entry ;
  end
end

end  % function
