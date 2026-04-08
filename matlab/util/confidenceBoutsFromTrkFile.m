function [startFrameFromSortedBoutIndex, endFrameFromSortedBoutIndex, extremeFrameFromSortedBoutIndex, ...
         trackletIndexFromSortedBoutIndex, targetIndexFromSortedBoutIndex, extremeConfidenceFromSortedBoutIndex, ...
         minConf, maxConf] = ...
  confidenceBoutsFromTrkFile(trkFile, threshold, isConfidenceLackThereof, isQuantile)
% Find bouts of consecutive frames whose confidence is above a threshold.
%
% A bout is a maximal contiguous run of consecutive frames (within a single
% tracklet) where confidence stays below (or above, in lack-thereof mode)
% the threshold.  Each bout is represented by the frame achieving the
% extreme confidence within the run.
%
% Returns empty column vectors and NaN extremes when there are no bouts.

% Gather per-frame confidence data from all tracklets.
trackletCount = trkFile.ntracklets ;
trackletIndexFromTrackletIndex = (1 : trackletCount)' ;
[frameIndexFromTrackletFrameIndexFromTrackletIndex, ...
 trackletIndexFromTrackletFrameIndexFromTrackletIndex, ...
 targetIndexFromTrackletFrameIndexFromTrackletIndex, ...
 minConfFromTrackletFrameIndexFromTrackletIndex, ...
 maxConfFromTrackletFrameIndexFromTrackletIndex] = ...
  arrayfun(@(trackletIndex) collectConfidenceDataForSingleTracklet(trkFile, trackletIndex), ...
           trackletIndexFromTrackletIndex, ...
           'UniformOutput', false) ;

% Concatenate all the *FromTrackletIndex cell arrays together, thus
% collecting all valid (frame, tracklet) pairs into flattened vectors.
frameIndexFromPairIndex = vertcat(frameIndexFromTrackletFrameIndexFromTrackletIndex{:}) ;
trackletIndexFromPairIndex = vertcat(trackletIndexFromTrackletFrameIndexFromTrackletIndex{:}) ;
targetIndexFromPairIndex = vertcat(targetIndexFromTrackletFrameIndexFromTrackletIndex{:}) ;
minConfFromPairIndex = vertcat(minConfFromTrackletFrameIndexFromTrackletIndex{:}) ;
maxConfFromPairIndex = vertcat(maxConfFromTrackletFrameIndexFromTrackletIndex{:}) ;

% Do an early return if there are no pairs
if isempty(frameIndexFromPairIndex)
  startFrameFromSortedBoutIndex = zeros(0, 1) ;
  endFrameFromSortedBoutIndex = zeros(0, 1) ;
  extremeFrameFromSortedBoutIndex = zeros(0, 1) ;
  trackletIndexFromSortedBoutIndex = zeros(0, 1) ;
  targetIndexFromSortedBoutIndex = zeros(0, 1) ;
  extremeConfidenceFromSortedBoutIndex = zeros(0, 1) ;
  minConf = nan ;
  maxConf = nan ;
  return
end

% Record overall extremes before filtering.
minConf = min(minConfFromPairIndex) ;
maxConf = max(maxConfFromPairIndex) ;

% If in quantile mode, convert the quantile threshold to an absolute one.
if isQuantile
  if isConfidenceLackThereof
    threshold = quantile(maxConfFromPairIndex, 1 - threshold) ;
  else
    threshold = quantile(minConfFromPairIndex, threshold) ;
  end
end

% Build bouts: contiguous runs of consecutive frames (within a single
% tracklet) that pass the confidence threshold.
trackletIndexFromFoundTrackletIndex = unique(trackletIndexFromPairIndex) ;
trackletIndexFromFoundTrackletIndexAsCell = num2cell(trackletIndexFromFoundTrackletIndex) ;
[startFrameFromTrackletBoutIndexFromFoundTrackletIndex, ...
 endFrameFromTrackletBoutIndexFromFoundTrackletIndex, ...
 extremalFrameIndexFromTrackletBoutIndexFromFoundTrackletIndex, ...
 trackletIndexFromTrackletBoutIndexFromFoundTrackletIndex, ...
 targetIndexFromTrackletBoutIndexFromFoundTrackletIndex, ...
 extremalConfFromTrackletBoutIndexFromFoundTrackletIndex] = ...
  cellfun(@(trackletIndex) ...
            collectBoutsForSingleTracklet(trackletIndex, ...
                                          frameIndexFromPairIndex, ...
                                          trackletIndexFromPairIndex, ...
                                          targetIndexFromPairIndex, ...
                                          minConfFromPairIndex, ...
                                          maxConfFromPairIndex, ...
                                          threshold, ...
                                          isConfidenceLackThereof), ...
          trackletIndexFromFoundTrackletIndexAsCell, ...
          'UniformOutput', false) ;

% Concatenate all the *FromFoundTrackletIndex cell arrays together, thus
% collecting all the bouts from all the tracklets.
startFrameFromBoutIndex = vertcat(startFrameFromTrackletBoutIndexFromFoundTrackletIndex{:}) ;
endFrameFromBoutIndex = vertcat(endFrameFromTrackletBoutIndexFromFoundTrackletIndex{:}) ;
trackletIndexFromBoutIndex = vertcat(trackletIndexFromTrackletBoutIndexFromFoundTrackletIndex{:}) ;
targetIndexFromBoutIndex = vertcat(targetIndexFromTrackletBoutIndexFromFoundTrackletIndex{:}) ;
extremalConfFromBoutIndex = vertcat(extremalConfFromTrackletBoutIndexFromFoundTrackletIndex{:}) ;
extremeConfFrameIndexFromBoutIndex = vertcat(extremalFrameIndexFromTrackletBoutIndexFromFoundTrackletIndex{:}) ;

% Sort bouts by extreme confidence.
if isConfidenceLackThereof
  [~, sortOrder] = sort(extremalConfFromBoutIndex, 'descend') ;
else
  [~, sortOrder] = sort(extremalConfFromBoutIndex, 'ascend') ;
end

% Sort everything else according to the sort order
startFrameFromSortedBoutIndex = startFrameFromBoutIndex(sortOrder) ;
endFrameFromSortedBoutIndex = endFrameFromBoutIndex(sortOrder) ;
extremeFrameFromSortedBoutIndex = extremeConfFrameIndexFromBoutIndex(sortOrder) ;
trackletIndexFromSortedBoutIndex = trackletIndexFromBoutIndex(sortOrder) ;
targetIndexFromSortedBoutIndex = targetIndexFromBoutIndex(sortOrder) ;
extremeConfidenceFromSortedBoutIndex = extremalConfFromBoutIndex(sortOrder) ;

end  % function
