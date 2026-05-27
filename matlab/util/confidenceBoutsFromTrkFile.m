function [firstFrameIndexFromSortedBoutIndex, lastFrameIndexFromSortedBoutIndex, extremalConfFrameIndexFromSortedBoutIndex, ...
          trackletIndexFromSortedBoutIndex, targetIndexFromSortedBoutIndex, extremalConfFromSortedBoutIndex, ...
          minConf, maxConf, ...
          absoluteThreshold] = ...
  confidenceBoutsFromTrkFile(trkFile, naiveQuantileThreshold)
% Find bouts of consecutive frames whose max-landmark interest
% (interest = -confidence) is at or above a quantile-derived threshold
% (i.e. uncertain bouts).
%
% Internally the bout-formation, quantile, and sort calculations all
% happen in the interest domain: a higher interest means a less
% confident, more uncertain frame.  A bout is a maximal contiguous run
% of consecutive frames (within a single tracklet) whose per-frame
% max-interest stays at or above the quantile-derived threshold.  Each
% bout is represented by the frame achieving the maximum interest within
% the run, and bouts are returned in descending order of that extremal
% interest (i.e. ascending order of extremal confidence) -- most
% uncertain first.
%
% Inputs and outputs are expressed in the confidence domain so callers
% that present results to the user in terms of confidence don't need to
% flip signs.  In particular, naiveQuantileThreshold is interpreted in
% the interest domain (so 0.99 keeps the top 1% most-interesting
% frames), and the returned absoluteThreshold, extremalConf, minConf,
% and maxConf are all confidence values.
%
% Returns empty column vectors and NaN extremes when there are no bouts.

% Gather per-frame max-interest data from all tracklets.
trackletCount = trkFile.ntracklets ;
trackletIndices = (1 : trackletCount)' ;
[frameIndexFromTrackletFrameIndexFromTrackletIndex, ...
 trackletIndexFromTrackletIndex, ...
 targetIndexFromTrackletIndex, ...
 interestFromTrackletFrameIndexFromTrackletIndex] = ...
  arrayfun(@(trackletIndex) collectInterestDataForSingleTracklet(trkFile, trackletIndex), ...
           trackletIndices, ...
           'UniformOutput', false) ;

% Concatenate all the *FromTrackletIndex cell arrays together, thus
% collecting all valid (frame, tracklet) pairs into a single flattened
% interest vector.
interestFromPairIndex = vertcat(interestFromTrackletFrameIndexFromTrackletIndex{:}) ;

% Do an early return if there are no pairs.
if isempty(interestFromPairIndex)
  firstFrameIndexFromSortedBoutIndex = zeros(0, 1) ;
  lastFrameIndexFromSortedBoutIndex = zeros(0, 1) ;
  extremalConfFrameIndexFromSortedBoutIndex = zeros(0, 1) ;
  trackletIndexFromSortedBoutIndex = zeros(0, 1) ;
  targetIndexFromSortedBoutIndex = zeros(0, 1) ;
  extremalConfFromSortedBoutIndex = zeros(0, 1) ;
  minConf = nan ;
  maxConf = nan ;
  absoluteThreshold = nan ;
  return
end

% Record overall extremes before filtering.  Higher interest maps to
% lower confidence and vice versa.
minConf = -max(interestFromPairIndex) ;
maxConf = -min(interestFromPairIndex) ;

% Compute the absolute interest threshold corresponding to the given
% quantile.  Higher naiveQuantileThreshold filters out more (only the
% most-interesting frames survive).
absoluteInterestThreshold = quantile(interestFromPairIndex, naiveQuantileThreshold) ;

% Build bouts: contiguous runs of consecutive frames (within a single
% tracklet) whose interest is at or above the threshold.  The per-bout
% frame-index cell array drops "Interest" from its name to stay under
% the 63-character identifier limit.
[firstFrameIndexFromTrackletBoutIndexFromTrackletIndex, ...
 lastFrameIndexFromTrackletBoutIndexFromTrackletIndex, ...
 extremalFrameIndexFromTrackletBoutIndexFromTrackletIndex, ...
 trackletIndexFromTrackletBoutIndexFromTrackletIndex, ...
 targetIndexFromTrackletBoutIndexFromTrackletIndex, ...
 extremalInterestFromTrackletBoutIndexFromTrackletIndex] = ...
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
extremalInterestFromBoutIndex = vertcat(extremalInterestFromTrackletBoutIndexFromTrackletIndex{:}) ;
extremalFrameIndexFromBoutIndex = vertcat(extremalFrameIndexFromTrackletBoutIndexFromTrackletIndex{:}) ;

% Sort bouts by extremal interest, descending: most interesting (least
% confident) first.
[~, boutIndexFromSortedBoutIndex] = sort(extremalInterestFromBoutIndex, 'descend') ;

% Sort everything else according to the sort order, and translate the
% extremal interest back to extremal confidence.
firstFrameIndexFromSortedBoutIndex = firstFrameIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
lastFrameIndexFromSortedBoutIndex = lastFrameIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
extremalConfFrameIndexFromSortedBoutIndex = extremalFrameIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
trackletIndexFromSortedBoutIndex = trackletIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
targetIndexFromSortedBoutIndex = targetIndexFromBoutIndex(boutIndexFromSortedBoutIndex) ;
extremalConfFromSortedBoutIndex = -extremalInterestFromBoutIndex(boutIndexFromSortedBoutIndex) ;

% Translate the absolute interest threshold back to confidence.
absoluteThreshold = -absoluteInterestThreshold ;

end  % function
