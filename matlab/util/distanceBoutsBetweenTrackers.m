function [firstFrameIndexFromSortedBoutIndex, ...
          lastFrameIndexFromSortedBoutIndex, ...
          maxDistanceFrameIndexFromSortedBoutIndex, ...
          trackletIndexFromSortedBoutIndex, ...
          targetIndexFromSortedBoutIndex, ...
          maxDistanceFromSortedBoutIndex, ...
          absoluteDistanceThreshold] = ...
  distanceBoutsBetweenTrackers(refTrkFile, ...
                               testTrkFile, ...
                               nframes, ...
                               quantileThreshold, ...
                               matchDistanceThreshold)
% Find bouts of consecutive frames where the per-frame max landmark
% distance between a reference tracker's tracklet and its matched test
% tracker tracklet is at or above a quantile-derived threshold.
%
% At each frame, the ref poses and test poses present at that frame are
% put into one-to-one correspondence by Hungarian matching (matchpairs),
% as in computeMAErr().  The assignment cost for a (ref, test) pose pair
% is the mean over landmarks of the landmark-to-landmark distance, and
% pairs whose cost is matchDistanceThreshold or more are left unmatched.
% The per-frame interest signal is the max over landmarks of the
% ref-to-test landmark distance.  Ref frames with no matched test pose
% are dropped.  Bouts are returned sorted by per-bout max distance,
% descending (largest disagreement first).
%
% Inputs:
%   refTrkFile -- TrkFile for the reference tracker (current movie)
%   testTrkFile -- TrkFile for the test tracker (current movie)
%   nframes -- number of frames in the current movie
%   quantileThreshold -- in [0, 1]; higher keeps only the largest
%     per-frame distances
%   matchDistanceThreshold -- mean-over-landmarks pixel distance at or
%     above which a ref pose and a test pose are considered unrelated
%     and left unmatched
%
% Returns empty column vectors and NaN threshold when there are no bouts.

% Sanity-check the inputs.
assert(isa(refTrkFile, 'TrkFile'), ...
       'refTrkFile must be a TrkFile') ;
assert(isa(testTrkFile, 'TrkFile'), ...
       'testTrkFile must be a TrkFile') ;
assert(isscalar(nframes) && isnumeric(nframes) && isreal(nframes) && ...
       isfinite(nframes) && nframes == round(nframes) && nframes >= 0, ...
       'nframes must be a nonnegative integer') ;
assert(isscalar(quantileThreshold) && isnumeric(quantileThreshold) && ...
       isreal(quantileThreshold) && isfinite(quantileThreshold) && ...
       0 <= quantileThreshold && quantileThreshold <= 1, ...
       'quantileThreshold must be a finite scalar in [0, 1]') ;
assert(isscalar(matchDistanceThreshold) && isnumeric(matchDistanceThreshold) && ...
       isreal(matchDistanceThreshold) && isfinite(matchDistanceThreshold) && ...
       matchDistanceThreshold >= 0, ...
       'matchDistanceThreshold must be a nonnegative finite scalar') ;

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
costOfNonAssignment = matchDistanceThreshold / 2 ;
for frameIndex = 1 : nframes
  refPoses = refPosesFromFrameIndex{frameIndex} ;
  testPoses = testPosesFromFrameIndex{frameIndex} ;
  if isempty(refPoses) || isempty(testPoses)
    continue
  end
  refPoseCount = numel(refPoses) ;
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
  for matchIndex = 1 : size(matchedPoseIndexPairs, 1)
    refPoseIndex = matchedPoseIndexPairs(matchIndex, 1) ;
    testPoseIndex = matchedPoseIndexPairs(matchIndex, 2) ;
    maxLandmarkDistance = maxDistanceFromRefAndTestPoseIndex(refPoseIndex, testPoseIndex) ;
    if ~isfinite(maxLandmarkDistance)
      continue
    end
    trackletIndex = refPoses(refPoseIndex).trackletIndex ;
    distanceFromFrameIndexAndTrackletIndex(frameIndex, trackletIndex) = maxLandmarkDistance ;
  end
end

% Repackage the matched distances as per-tracklet cell arrays for the
% shared bout-formation pipeline.
frameIndexFromTrackletFrameIndexFromTrackletIndex = cell(refTrackletCount, 1) ;
trackletIndexFromTrackletIndex = cell(refTrackletCount, 1) ;
targetIndexFromTrackletIndex = cell(refTrackletCount, 1) ;
interestFromTrackletFrameIndexFromTrackletIndex = cell(refTrackletCount, 1) ;
for trackletIndex = 1 : refTrackletCount
  isValidFromFrameIndex = isfinite(distanceFromFrameIndexAndTrackletIndex(:, trackletIndex)) ;
  frameIndexFromTrackletFrameIndexFromTrackletIndex{trackletIndex} = find(isValidFromFrameIndex) ;
  trackletIndexFromTrackletIndex{trackletIndex} = trackletIndex ;
  targetIndexFromTrackletIndex{trackletIndex} = refTrkFile.pTrkiTgt(trackletIndex) ;
  interestFromTrackletFrameIndexFromTrackletIndex{trackletIndex} = ...
    distanceFromFrameIndexAndTrackletIndex(isValidFromFrameIndex, trackletIndex) ;
end

% Run the shared bout-formation / quantile / sort pipeline.  Distance is
% already an interest signal (higher = more interesting), so no
% sign-flipping at the boundary.
[firstFrameIndexFromSortedBoutIndex, ...
 lastFrameIndexFromSortedBoutIndex, ...
 maxDistanceFrameIndexFromSortedBoutIndex, ...
 trackletIndexFromSortedBoutIndex, ...
 targetIndexFromSortedBoutIndex, ...
 maxDistanceFromSortedBoutIndex, ...
 absoluteDistanceThreshold] = ...
  interestBoutsFromPerTrackletData(frameIndexFromTrackletFrameIndexFromTrackletIndex, ...
                                   trackletIndexFromTrackletIndex, ...
                                   targetIndexFromTrackletIndex, ...
                                   interestFromTrackletFrameIndexFromTrackletIndex, ...
                                   quantileThreshold) ;

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
