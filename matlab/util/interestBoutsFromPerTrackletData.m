function [firstFrameIndexFromSortedBoutIndex, ...
          lastFrameIndexFromSortedBoutIndex, ...
          maxInterestFrameIndexFromSortedBoutIndex, ...
          trackletIndexFromSortedBoutIndex, ...
          targetIndexFromSortedBoutIndex, ...
          maxInterestFromSortedBoutIndex, ...
          absoluteInterestThreshold] = ...
  interestBoutsFromPerTrackletData(frameIndexFromTrackletFrameIndexFromTrackletIndex, ...
                                   trackletIndexFromTrackletIndex, ...
                                   targetIndexFromTrackletIndex, ...
                                   interestFromTrackletFrameIndexFromTrackletIndex, ...
                                   quantileThreshold)
% Find bouts of consecutive frames whose per-frame interest is at or above
% a quantile-derived threshold, given pre-collected per-tracklet interest
% data.
%
% This is the bout-formation, quantile, and sort core shared by the
% uncertain-frames (interest = -confidence) and compare-trackers
% (interest = max landmark distance) pipelines.  Callers do the
% data-collection step that turns their source into per-tracklet
% interest cell arrays, then call this function.
%
% Inputs (one cell per tracklet, all parallel):
%   frameIndexFromTrackletFrameIndexFromTrackletIndex -- movie-frame
%     indices for each valid sample in the tracklet
%   trackletIndexFromTrackletIndex -- scalar tracklet index
%   targetIndexFromTrackletIndex -- scalar target index
%   interestFromTrackletFrameIndexFromTrackletIndex -- interest values for
%     each valid sample (parallel to the frame-index vector)
%   quantileThreshold -- in [0, 1]; a higher value keeps only the most
%     interesting frames.
%
% Returns sorted-by-descending-max-interest per-bout vectors in the
% interest domain, plus the absolute interest threshold derived from the
% quantile.  Returns empty column vectors and NaN threshold when there
% are no valid (frame, tracklet) pairs.

% Concatenate all the *FromTrackletIndex cell arrays together, thus
% collecting all valid (frame, tracklet) pairs into a single flattened
% interest vector.
interestFromPairIndex = vertcat(interestFromTrackletFrameIndexFromTrackletIndex{:}) ;

% Do an early return if there are no pairs.
if isempty(interestFromPairIndex)
  firstFrameIndexFromSortedBoutIndex = zeros(0, 1) ;
  lastFrameIndexFromSortedBoutIndex = zeros(0, 1) ;
  maxInterestFrameIndexFromSortedBoutIndex = zeros(0, 1) ;
  trackletIndexFromSortedBoutIndex = zeros(0, 1) ;
  targetIndexFromSortedBoutIndex = zeros(0, 1) ;
  maxInterestFromSortedBoutIndex = zeros(0, 1) ;
  absoluteInterestThreshold = nan ;
  return
end

% Compute the absolute interest threshold corresponding to the given
% quantile.  Higher quantileThreshold filters out more (only the most-
% interesting frames survive).
absoluteInterestThreshold = quantile(interestFromPairIndex, quantileThreshold) ;

% Build bouts: contiguous runs of consecutive frames (within a single
% tracklet) whose interest is at or above the threshold.
[firstFrameIndexFromTrackletBoutIndexFromTrackletIndex, ...
 lastFrameIndexFromTrackletBoutIndexFromTrackletIndex, ...
 maxInterestFrameIndexFromTrackletBoutIndexFromTrackletIndex, ...
 trackletIndexFromTrackletBoutIndexFromTrackletIndex, ...
 targetIndexFromTrackletBoutIndexFromTrackletIndex, ...
 maxInterestFromTrackletBoutIndexFromTrackletIndex] = ...
  cellfun(@(frameIndexFromTrackletFrameIndex, ...
            trackletIndex, ...
            targetIndex, ...
            interestFromTrackletFrameIndex) ...
            collectInterestBoutsForSingleTracklet(frameIndexFromTrackletFrameIndex, ...
                                                  trackletIndex, ...
                                                  targetIndex, ...
                                                  interestFromTrackletFrameIndex, ...
                                                  absoluteInterestThreshold), ...
          frameIndexFromTrackletFrameIndexFromTrackletIndex, ...
          trackletIndexFromTrackletIndex, ...
          targetIndexFromTrackletIndex, ...
          interestFromTrackletFrameIndexFromTrackletIndex, ...
          'UniformOutput', false) ;

% Concatenate all the *FromTrackletIndex cell arrays together, thus
% collecting all the bouts from all the tracklets.
firstFrameIndexFromBoutIndex = vertcat(firstFrameIndexFromTrackletBoutIndexFromTrackletIndex{:}) ;
lastFrameIndexFromBoutIndex = vertcat(lastFrameIndexFromTrackletBoutIndexFromTrackletIndex{:}) ;
trackletIndexFromBoutIndex = vertcat(trackletIndexFromTrackletBoutIndexFromTrackletIndex{:}) ;
targetIndexFromBoutIndex = vertcat(targetIndexFromTrackletBoutIndexFromTrackletIndex{:}) ;
maxInterestFromBoutIndex = vertcat(maxInterestFromTrackletBoutIndexFromTrackletIndex{:}) ;
maxInterestFrameIndexFromBoutIndex = vertcat(maxInterestFrameIndexFromTrackletBoutIndexFromTrackletIndex{:}) ;

% Sort bouts by max interest, descending: most interesting first.
[~, boutIndexFromSortedBoutIndex] = sort(maxInterestFromBoutIndex, 'descend') ;

% Sort everything else according to the sort order.
firstFrameIndexFromSortedBoutIndex = firstFrameIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
lastFrameIndexFromSortedBoutIndex = lastFrameIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
maxInterestFrameIndexFromSortedBoutIndex = maxInterestFrameIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
trackletIndexFromSortedBoutIndex = trackletIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
targetIndexFromSortedBoutIndex = targetIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
maxInterestFromSortedBoutIndex = maxInterestFromBoutIndex(boutIndexFromSortedBoutIndex) ;

end  % function
