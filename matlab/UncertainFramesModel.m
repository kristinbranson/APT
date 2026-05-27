classdef UncertainFramesModel < handle
  % Holds computed data about bouts of frames with low tracker confidence
  % for the current movie.

  properties (Access=private)
    quantileConfidenceThreshold_ = 0.99
      % scalar double, the quantile threshold (0-1) for filtering bouts.
  end

  properties (Transient, Access=private)
    labeler_  % back-reference to Labeler (Transient in spirit)
    startFrameFromBoutIndex_  % [N x 1] first frame of each bout
    endFrameFromBoutIndex_  % [N x 1] last frame of each bout
    extremeFrameFromBoutIndex_  % [N x 1] frame with min/max confidence in each bout
    trackletIndexFromBoutIndex_  % [N x 1] tracklet indices (into TrkFile)
    targetIndexFromBoutIndex_  % [N x 1] target indices (for navigation)
    extremeConfidenceFromBoutIndex_  % [N x 1] min- or max-confidence values per bout
    absoluteConfidenceThreshold_ = nan  % scalar double, quantile-derived absolute threshold
    overallMinConfidence_ = nan  % scalar double, min of allMinConf across all frames
    overallMaxConfidence_ = nan  % scalar double, max of allMaxConf across all frames
    % isLaden_ = false  % scalar logical, true if there is anything to show
    isVisible_ = false  % scalar logical, whether the UFC figure is visible
    isFresh_ = false
      % scalar logical, whether the data in this object is up-to-date with the
      % data in the rest of the Labeler.  (Opposite of stale.)
    currentBoutIndexMaybe_ = []  % the curently selected bout index, or empty
  end

  properties (Dependent)
    isLaden
    isVisible
    absoluteConfidenceThreshold
    quantileConfidenceThreshold
    overallMinConfidence
    overallMaxConfidence
    currentBoutIndexMaybe
  end

  properties (Dependent, Hidden)
    displayStringFromBoutIndex  % cellstr showing the uncertain frame-target pairs (used for listbox display)
  end

  methods
    function obj = UncertainFramesModel(labeler)
      % Construct an UncertainFramesModel with a back-reference to the
      % given Labeler.
      obj.labeler_ = labeler ;
      obj.startFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.endFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.extremeFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.trackletIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.targetIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.extremeConfidenceFromBoutIndex_ = zeros(0, 1) ;
    end  % function

    function result = get.isLaden(obj)
      % Return whether there is anything to show.
      result = ~isempty(obj.extremeConfidenceFromBoutIndex_) ;
    end  % function

    function result = get.isVisible(obj)
      % Return whether the uncertain-frames figure is visible.
      result = obj.isVisible_ ;
    end  % function

    function set.isVisible(obj, newValue)
      obj.isVisible_ = newValue ;
      obj.syncFromPredictionsIfStaleAndVisible_() ;
      obj.labeler_.notifyRetrograde('updateUncertainFrames') ;
      obj.labeler_.notifyRetrograde('didSetUncertainFramesIsVisible') ;
    end  % function

    function result = get.absoluteConfidenceThreshold(obj)
      % Return the absolute confidence threshold used for filtering.
      result = obj.absoluteConfidenceThreshold_ ;
    end  % function

    function result = get.quantileConfidenceThreshold(obj)
      % Return the confidence threshold corresponding to the current quantile.
      result = obj.quantileConfidenceThreshold_ ;
    end  % function

    function set.quantileConfidenceThreshold(obj, newValue)
      % Set the quantile threshold, then resync and notify.
      isValid = isscalar(newValue) && ...
                isnumeric(newValue) && ...
                isreal(newValue) && ...
                isfinite(newValue) && ...
                0 <= newValue && ...
                newValue <= 1 ;
      if isValid
        obj.quantileConfidenceThreshold_ = newValue ;
        obj.isFresh_ = false ;
        obj.syncFromPredictionsIfStaleAndVisible_() ;
      end
      obj.labeler_.notifyRetrograde('didSetUncertainFramesThreshold') ;
      if ~isValid
        error('APT:invalidPropertyValue', ...
              'Threshold must be a finite scalar between 0 and 1') ;
      end
    end  % function

    function result = get.overallMinConfidence(obj)
      % Return the min of per-frame min-confidence across all frames.
      result = obj.overallMinConfidence_ ;
    end  % function

    function result = get.overallMaxConfidence(obj)
      % Return the max of per-frame max-confidence across all frames.
      result = obj.overallMaxConfidence_ ;
    end  % function

    function syncFromPredictions(obj)
      % Force a resync from the current tracking predictions.
      obj.isFresh_ = false ;
      obj.syncFromPredictionsIfStaleAndVisible_() ;
      obj.labeler_.notifyRetrograde('updateUncertainFrames') ;
    end

    function result = get.displayStringFromBoutIndex(obj)
      % Return a cellstr showing the frame-target pairs (suitable for use in a
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
        confidence = obj.extremeConfidenceFromBoutIndex_(boutIndex) ;
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
            result{boutIndex} = sprintf('Frm %d  %s %d  Conf %.3f', startFrameIndex, label, tragletIndex, confidence) ;
          else
            result{boutIndex} = sprintf('Frm %d-%d  %s %d  Conf %.3f', startFrameIndex, endFrameIndex, label, tragletIndex, confidence) ;
          end
        else
          if isSingleFrame
            result{boutIndex} = sprintf('Frm %d  Conf %.3f', startFrameIndex, confidence) ;
          else
            result{boutIndex} = sprintf('Frm %d-%d  Conf %.3f', startFrameIndex, endFrameIndex, confidence) ;
          end
        end
      end
    end  % function

    function [frameIndex, trackletIndex, targetIndex] = frameTrackletAndTargetIndexFromCurrentBoutIndex(obj)
      % Return the extreme-confidence frame, tracklet index, and target index
      % for the currently selected bout.  Errors if no bout is currently selected.
      boutIndex = obj.currentBoutIndexMaybe_ ;
      if isempty(boutIndex)
        error('APT:invalidPropertyValue', ...
              'No current bout index is set') ;
      end
      [frameIndex, trackletIndex, targetIndex] = obj.frameTrackletAndTargetIndexFromBoutIndex_(boutIndex) ;
    end  % function

    function [frameIndex, trackletIndex, targetIndex] = frameTrackletAndTargetIndexFromBoutIndex_(obj, boutIndex)
      % Return the extreme-confidence frame and tracklet index for the given bout.
      frameIndex = obj.extremeFrameFromBoutIndex_(boutIndex) ;
      trackletIndex = obj.trackletIndexFromBoutIndex_(boutIndex) ;
      targetIndex = obj.targetIndexFromBoutIndex_(boutIndex) ;
    end  % function

    function result = get.currentBoutIndexMaybe(obj)
      % Return the currently selected bout index, or [] if none is selected.
      result = obj.currentBoutIndexMaybe_ ;
    end  % function

    function set.currentBoutIndexMaybe(obj, newValue)
      % Setter method for currentBoutIndexMaybe.  A valid value is a positive-integer scalar in 1:nBouts.
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
      obj.labeler_.notifyRetrograde('updateUncertainFrames') ;
      if ~isValid
        error('APT:invalidPropertyValue', ...
              'Current bout index must be a positive integer in 1:%d', nBouts) ;
      end
    end  % function
  end  % methods

  methods (Access=private)
    function syncFromPredictionsIfStaleAndVisible_(obj)
      % Compute bouts of uncertain frames from the current movie's
      % tracking results.
      if ~obj.isVisible_ || obj.isFresh_
        return
      end

      labeler = obj.labeler_ ;

      if labeler.currMovie == 0
        obj.clear_() ;
        return
      end

      tracker = labeler.tracker ;
      if isempty(tracker)
        obj.clear_() ;
        return
      end

      trkFile = tracker.trkP ;
      if isempty(trkFile) || ~isa(trkFile, 'TrkFile') || ~trkFile.hasdata()
        obj.clear_() ;
        return
      end

      if ~isprop(trkFile, 'pTrkConf')
        obj.clear_() ;
        return
      end

      [obj.startFrameFromBoutIndex_, ...
       obj.endFrameFromBoutIndex_, ...
       obj.extremeFrameFromBoutIndex_, ...
       obj.trackletIndexFromBoutIndex_, ...
       obj.targetIndexFromBoutIndex_, ...
       obj.extremeConfidenceFromBoutIndex_, ...
       obj.overallMinConfidence_, ...
       obj.overallMaxConfidence_, ...
       obj.absoluteConfidenceThreshold_] = ...
        confidenceBoutsFromTrkFile(trkFile, obj.quantileConfidenceThreshold_) ;
      obj.isFresh_ = true ;
    end  % function

    function clear_(obj)
      % Reset to empty state.
      obj.startFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.endFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.extremeFrameFromBoutIndex_ = zeros(0, 1) ;
      obj.trackletIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.targetIndexFromBoutIndex_ = zeros(0, 1) ;
      obj.extremeConfidenceFromBoutIndex_ = zeros(0, 1) ;
      obj.absoluteConfidenceThreshold_ = nan ;
      % obj.isLaden_ = false ;
      obj.isFresh_ = true ;
    end  % function
  end  % methods
end  % classdef
