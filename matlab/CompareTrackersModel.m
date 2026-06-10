classdef CompareTrackersModel < handle
  % Holds computed data about bouts of frames where a reference tracker
  % and a test tracker disagree, for the current movie.

  properties (Access=private)
    quantileThreshold_ = 0.99
      % scalar double, the quantile threshold (0-1) for filtering bouts.
    matchDistanceThreshold_ = 50
      % scalar double, mean-over-landmarks pixel distance at or above
      % which a ref pose and a test pose at a given frame are considered
      % unrelated and left unmatched by the per-frame Hungarian matching.
      % Hardcoded; no UI control for now.
  end

  properties (Transient, Access=private)
    labeler_  % back-reference to Labeler
    testTracker_ = []
      % Handle to the tracker selected as the test tracker, or [] for
      % none.  Stored by identity (not by index) so the selection
      % survives reordering of labeler.trackerHistory.  The reference
      % tracker is always the current tracker (trackerHistory{1}), so it
      % is not stored here.
    startFrameFromBoutIndex_  % [N x 1] first frame of each bout
    endFrameFromBoutIndex_  % [N x 1] last frame of each bout
    maxDistanceFrameFromBoutIndex_  % [N x 1] frame where the per-bout max distance occurs
    trackletIndexFromBoutIndex_  % [N x 1] ref-tracklet indices (into the ref TrkFile)
    targetIndexFromBoutIndex_  % [N x 1] target indices (for navigation)
    maxDistanceFromBoutIndex_  % [N x 1] per-bout max landmark distance
    absoluteDistanceThreshold_ = nan
      % scalar double, the quantile-derived absolute pixel-distance
      % threshold used to filter bouts.
    isVisible_ = false  % scalar logical, whether the compare-trackers figure is visible
    isFresh_ = false
      % scalar logical, whether the data in this object is up-to-date with the
      % data in the rest of the Labeler.  (Opposite of stale.)
    currentBoutIndexMaybe_ = []  % the currently selected bout index, or empty
  end

  properties (Dependent)
    isLaden
    isVisible
    absoluteDistanceThreshold
    quantileThreshold
    referenceTracker
    testTracker
    currentBoutIndexMaybe
  end

  properties (Dependent, Hidden)
    displayStringFromBoutIndex
      % cellstr showing the bout frame ranges and max distances (used for
      % listbox display).
    isTestTrackerChoiceValid
      % logical: true when the test-tracker selection differs from the
      % reference-tracker selection (so a meaningful comparison can be
      % computed), false when they coincide.
  end

  methods
    function obj = CompareTrackersModel(labeler)
      % Construct a CompareTrackersModel with a back-reference to the
      % given Labeler.
      obj.labeler_ = labeler ;
      obj.startFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.endFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.maxDistanceFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.trackletIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.targetIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.maxDistanceFromBoutIndex_ = zeros(0, 1) ;
    end  % function

    function result = get.isLaden(obj)
      % Return whether there is anything to show.
      result = ~isempty(obj.maxDistanceFromBoutIndex_) ;
    end  % function

    function result = get.isVisible(obj)
      % Return whether the compare-trackers figure is visible.
      result = obj.isVisible_ ;
    end  % function

    function set.isVisible(obj, newValue)
      obj.isVisible_ = newValue ;
      obj.syncFromPredictionsIfStaleAndVisible_() ;
      obj.labeler_.notifyRetrograde('updateCompareTrackers') ;
      obj.labeler_.notifyRetrograde('didSetCompareTrackersIsVisible') ;
    end  % function

    function result = get.absoluteDistanceThreshold(obj)
      % Return the absolute pixel-distance threshold used for filtering.
      result = obj.absoluteDistanceThreshold_ ;
    end  % function

    function result = get.quantileThreshold(obj)
      result = obj.quantileThreshold_ ;
    end  % function

    function set.quantileThreshold(obj, newValue)
      % Set the quantile threshold, then resync and notify.
      isValid = isscalar(newValue) && ...
                isnumeric(newValue) && ...
                isreal(newValue) && ...
                isfinite(newValue) && ...
                0 <= newValue && ...
                newValue <= 1 ;
      if isValid
        obj.quantileThreshold_ = newValue ;
        obj.isFresh_ = false ;
        obj.syncFromPredictionsIfStaleAndVisible_() ;
      end
      obj.labeler_.notifyRetrograde('didSetCompareTrackersThreshold') ;
      if ~isValid
        error('APT:invalidPropertyValue', ...
              'Threshold must be a finite scalar between 0 and 1') ;
      end
    end  % function

    function result = get.referenceTracker(obj)
      % The reference tracker is always the current tracker.  Read-only.
      % Returns [] if there are no trackers.
      result = obj.labeler_.tracker ;
    end  % function

    function result = get.testTracker(obj)
      % The tracker selected as the test tracker, or [] if none.  The
      % default is filled in (by identity) during
      % syncFromPredictionsIfStaleAndVisible_.
      result = obj.testTracker_ ;
    end  % function

    function set.testTracker(obj, newValue)
      % Set the test tracker, by identity.  A valid value is [] or a
      % tracker currently in the labeler's trackerHistory.
      isValid = isempty(newValue) || ...
                ( isscalar(newValue) && ...
                  isa(newValue, 'LabelTracker') && ...
                  isvalid(newValue) && ...
                  isTrackerInHistory_(newValue, obj.labeler_.trackerHistory) ) ;
      if isValid
        if isempty(newValue)
          obj.testTracker_ = [] ;
        else
          obj.testTracker_ = newValue ;
        end
        obj.isFresh_ = false ;
        obj.syncFromPredictionsIfStaleAndVisible_() ;
      end
      obj.labeler_.notifyRetrograde('didSetCompareTrackersTrackerSelection') ;
      if ~isValid
        error('APT:invalidPropertyValue', ...
              'Test tracker must be [] or a tracker in the labeler''s trackerHistory') ;
      end
    end  % function

    function syncFromPredictions(obj)
      % Force a resync from the current tracking predictions.
      obj.isFresh_ = false ;
      obj.syncFromPredictionsIfStaleAndVisible_() ;
      obj.labeler_.notifyRetrograde('updateCompareTrackers') ;
    end  % function

    function result = get.isTestTrackerChoiceValid(obj)
      % Return whether the test tracker differs from the reference
      % tracker (compared by identity).
      result = ~areSameTracker_(obj.testTracker, obj.referenceTracker) ;
    end  % function

    function result = get.displayStringFromBoutIndex(obj)
      % Return a cellstr describing each bout (suitable for use in a
      % listbox).
      if ~obj.isLaden
        result = {} ;
        return
      end
      boutCount = numel(obj.startFrameFromBoutIndex_) ;
      result = cell(boutCount, 1) ;
      isMA = obj.labeler_.maIsMA ;
      isMultiTarget = obj.labeler_.hasTrx || isMA ;
      for boutIndex = 1 : boutCount
        startFrameIndex = obj.startFrameFromBoutIndex_(boutIndex) ;
        endFrameIndex = obj.endFrameFromBoutIndex_(boutIndex) ;
        distance = obj.maxDistanceFromBoutIndex_(boutIndex) ;
        isSingleFrame = (startFrameIndex == endFrameIndex) ;
        if isMultiTarget
          if isMA
            label = 'Trklet' ;
            tragletIndex = obj.trackletIndexFromBoutIndex_(boutIndex) ;
          else
            label = 'Tgt' ;
            tragletIndex = obj.targetIndexFromBoutIndex_(boutIndex) ;
          end
          if isSingleFrame
            result{boutIndex} = sprintf('Frm %d  %s %d  MaxDist %.2f', startFrameIndex, label, tragletIndex, distance) ;
          else
            result{boutIndex} = sprintf('Frm %d-%d  %s %d  MaxDist %.2f', startFrameIndex, endFrameIndex, label, tragletIndex, distance) ;
          end
        else
          if isSingleFrame
            result{boutIndex} = sprintf('Frm %d  MaxDist %.2f', startFrameIndex, distance) ;
          else
            result{boutIndex} = sprintf('Frm %d-%d  MaxDist %.2f', startFrameIndex, endFrameIndex, distance) ;
          end
        end
      end
    end  % function

    function [frameIndex, trackletIndex, targetIndex] = frameTrackletAndTargetIndexFromCurrentBoutIndex(obj)
      % Return the max-distance frame, ref-tracklet index, and target
      % index for the currently selected bout.  Errors if no bout is
      % selected.
      boutIndex = obj.currentBoutIndexMaybe_ ;
      if isempty(boutIndex)
        error('APT:invalidPropertyValue', ...
              'No current bout index is set') ;
      end
      [frameIndex, trackletIndex, targetIndex] = obj.frameTrackletAndTargetIndexFromBoutIndex_(boutIndex) ;
    end  % function

    function [frameIndex, trackletIndex, targetIndex] = frameTrackletAndTargetIndexFromBoutIndex_(obj, boutIndex)
      % Return the max-distance frame and ref-tracklet index for the
      % given bout.
      frameIndex = obj.maxDistanceFrameFromBoutIndex_(boutIndex) ;
      trackletIndex = obj.trackletIndexFromBoutIndex_(boutIndex) ;
      targetIndex = obj.targetIndexFromBoutIndex_(boutIndex) ;
    end  % function

    function result = get.currentBoutIndexMaybe(obj)
      result = obj.currentBoutIndexMaybe_ ;
    end  % function

    function set.currentBoutIndexMaybe(obj, newValue)
      % Setter method for currentBoutIndexMaybe.  A valid value is a
      % positive-integer scalar in 1:nBouts.
      nBouts = numel(obj.startFrameFromBoutIndex_) ;
      isValid = ...
        isscalar(newValue) && ...
        isnumeric(newValue) && ...
        isreal(newValue) && ...
        isfinite(newValue) && ...
        newValue == round(newValue) && ...
        1 <= newValue && ...
        newValue <= nBouts ;
      if isValid
        obj.currentBoutIndexMaybe_ = newValue ;
      end
      obj.labeler_.notifyRetrograde('updateCompareTrackers') ;
      if ~isValid
        error('APT:invalidPropertyValue', ...
              'Current bout index must be a positive integer in 1:%d', nBouts) ;
      end
    end  % function
  end  % methods

  methods (Access=private)
    function syncFromPredictionsIfStaleAndVisible_(obj)
      % Compute bouts of disagreement between the reference and test
      % trackers for the current movie.
      if ~obj.isVisible_ || obj.isFresh_
        return
      end

      labeler = obj.labeler_ ;
      trackerHistory = labeler.trackerHistory ;
      trackerCount = numel(trackerHistory) ;

      if labeler.currMovie == 0
        obj.clear_() ;
        return
      end
      if trackerCount < 2
        obj.clear_() ;
        return
      end

      % The reference tracker is always the current tracker (history
      % index 1).  The test tracker is tracked by identity.
      referenceTracker = obj.referenceTracker ;
      if isempty(obj.testTracker_)
        % If no test tracker is selected, adopt the default: the first
        % non-current tracker.
        obj.testTracker_ = defaultTestTrackerFromHistory_(trackerHistory) ;
      end
      testTracker = obj.testTracker ;      

      % Short-circuit: comparing a tracker to itself (or having no
      % distinct test tracker) is meaningless and would just be a flat
      % zero-distance result.  The controller flags this state visually.
      % This is reachable when the user makes the test tracker the
      % current (reference) tracker.
      if isempty(testTracker) || areSameTracker_(testTracker, referenceTracker)
        obj.clear_() ;
        return
      end

      testIndex = trackerHistoryIndexFromTracker_(trackerHistory, testTracker) ;
      if isempty(testIndex)
        obj.clear_() ;
        return
      end
      refIndex = 1 ;

      refTrkFile = trkFileForTrackerHistoryIndex_(labeler, refIndex) ;
      testTrkFile = trkFileForTrackerHistoryIndex_(labeler, testIndex) ;
      if isempty(refTrkFile) || isempty(testTrkFile)
        obj.clear_() ;
        return
      end

      [obj.startFrameFromBoutIndex_, ...
       obj.endFrameFromBoutIndex_, ...
       obj.maxDistanceFrameFromBoutIndex_, ...
       obj.trackletIndexFromBoutIndex_, ...
       obj.targetIndexFromBoutIndex_, ...
       obj.maxDistanceFromBoutIndex_, ...
       obj.absoluteDistanceThreshold_] = ...
        distanceBoutsBetweenTrackers(refTrkFile, ...
                                     testTrkFile, ...
                                     labeler.nframes, ...
                                     obj.quantileThreshold_, ...
                                     obj.matchDistanceThreshold_) ;
      % The bout list has been rebuilt, so any previously selected bout
      % index refers to the old list.  Reset to no selection.
      obj.currentBoutIndexMaybe_ = [] ;
      obj.isFresh_ = true ;
    end  % function

    function clear_(obj)
      % Reset to empty state.
      obj.startFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.endFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.maxDistanceFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.trackletIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.targetIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.maxDistanceFromBoutIndex_ = zeros(0, 1) ;
      obj.absoluteDistanceThreshold_ = nan ;
      obj.currentBoutIndexMaybe_ = [] ;
      obj.isFresh_ = true ;
    end  % function
  end  % methods
end  % classdef



function result = areSameTracker_(a, b)
% Identity comparison for tracker handles that tolerates [] operands.
% Two trackers are the same iff they are the same handle; two empties are
% considered the same.
if isempty(a) || isempty(b)
  result = isempty(a) && isempty(b) ;
else
  result = (a == b) ;
end
end  % function



function index = trackerHistoryIndexFromTracker_(trackerHistory, tracker)
% Return the index of the given tracker within trackerHistory, or [] if
% it is not present.  Compared by identity.
index = find(cellfun(@(t)(t == tracker), trackerHistory), 1) ;
end  % function



function result = isTrackerInHistory_(tracker, trackerHistory)
% Return whether the given tracker handle is one of the trackers in
% trackerHistory (compared by identity).
result = ~isempty(trackerHistoryIndexFromTracker_(trackerHistory, tracker)) ;
end  % function



function result = defaultTestTrackerFromHistory_(trackerHistory)
% Return the default test tracker (the first non-current tracker), or []
% if there is none.  The current tracker is trackerHistory{1}.
if numel(trackerHistory) >= 2
  result = trackerHistory{2} ;
else
  result = [] ;
end
end  % function



function trkFile = trkFileForTrackerHistoryIndex_(labeler, trackerHistoryIndex)
% Return the merged TrkFile for the tracker at the given trackerHistory
% index, for the labeler's current movie.  Returns [] if no usable
% predictions are available.

trackerHistory = labeler.trackerHistory ;
if trackerHistoryIndex < 1 || trackerHistoryIndex > numel(trackerHistory)
  trkFile = [] ;
  return
end
tracker = trackerHistory{trackerHistoryIndex} ;

% The current tracker (index 1) already has its merged trkP cached.
if trackerHistoryIndex == 1 && ~isempty(tracker.trkP) && ...
    isa(tracker.trkP, 'TrkFile') && tracker.trkP.hasdata()
  trkFile = tracker.trkP ;
  return
end

% Otherwise (or if the cached one is empty), load from disk and merge
% views, following DeepTracker.trackCurrResUpdate.
mIdx = labeler.currMovIdx ;
if isempty(mIdx) || mIdx == 0
  trkFile = [] ;
  return
end
[trks, tfHasRes] = tracker.getTrackingResults(mIdx) ;
if ~tfHasRes
  trkFile = [] ;
  return
end
viewCount = size(trks, 2) ;
if viewCount == 0 || isempty(trks{1, 1})
  trkFile = [] ;
  return
end
mergedTrk = trks{1, 1} ;
if viewCount > 1
  mergedTrk.mergeMultiView(trks{1, 2:end}) ;
end
mergedTrk.initFrm2Tlt(labeler.nframes) ;
trkFile = mergedTrk ;
end  % function
