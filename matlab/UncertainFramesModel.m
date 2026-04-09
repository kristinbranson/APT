classdef UncertainFramesModel < handle
  % Holds computed data about bouts of frames with low tracker confidence
  % for the current movie.

  properties (Access=private)
    isConfidenceLackThereof_ = false
      % scalar logical, when true the UI finds high-confidence bouts instead of
      % low-confidence bouts.
    quantileConfidenceThreshold_ = 0.95
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
    overallMinConfidence_ = nan  % scalar double, min of allMinConf across all frames
    overallMaxConfidence_ = nan  % scalar double, max of allMaxConf across all frames
    % isLaden_ = false  % scalar logical, true if there is anything to show
    isVisible_ = false  % scalar logical, whether the UFC figure is visible
    isFresh_ = false
      % scalar logical, whether the data in this object is up-to-date with the
      % data in the rest of the Labeler.  (Opposite of stale.)
  end

  properties (Dependent)
    isLaden
    isVisible
    isConfidenceLackThereof
    confidenceThreshold
    quantileConfidenceThreshold
    overallMinConfidence
    overallMaxConfidence
  end

  properties (Dependent, Hidden)
    listboxString  % cellstr for listbox display
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
      obj.labeler_.notify_('updateUncertainFrames') ;
      obj.labeler_.notify_('didSetUncertainFramesIsVisible') ;
    end  % function

    function result = get.isConfidenceLackThereof(obj)
      % Return whether confidence values are treated as lack-of-confidence.
      result = obj.isConfidenceLackThereof_ ;
    end  % function

    function set.isConfidenceLackThereof(obj, newValue)
      % Set whether confidence values are treated as lack-of-confidence,
      % then resync and notify.
      obj.isConfidenceLackThereof_ = newValue ;
      obj.isFresh_ = false ;
      obj.syncFromPredictionsIfStaleAndVisible_() ;
      obj.labeler_.notify_('updateUncertainFrames') ;
    end  % function

    function result = get.confidenceThreshold(obj)
      % Return the confidence threshold for filtering.
      result = obj.quantileConfidenceThreshold_ ;
    end  % function

    function set.confidenceThreshold(obj, newValue)
      % Set the quantile confidence threshold.
      obj.quantileConfidenceThreshold = newValue ;
    end  % function

    function result = get.quantileConfidenceThreshold(obj)
      % Return the confidence threshold corresponding to the current quantile.
      result = obj.quantileConfidenceThreshold_ ;
    end  % function

    function set.quantileConfidenceThreshold(obj, newValue)
      % Set the quantile threshold, then resync and notify.
      obj.quantileConfidenceThreshold_ = newValue ;
      obj.isFresh_ = false ;
      obj.syncFromPredictionsIfStaleAndVisible_() ;
      obj.labeler_.notify_('didSetUncertainFramesThreshold') ;
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
      obj.labeler_.notify_('updateUncertainFrames') ;
    end

    function result = get.listboxString(obj)
      % Return a cellstr suitable for display in a listbox.
      if ~obj.isLaden
        result = {} ;
        return
      end
      nBouts = numel(obj.startFrameFromBoutIndex_) ;
      result = cell(nBouts, 1) ;
      isMultiTarget = obj.labeler_.hasTrx || obj.labeler_.maIsMA ;
      for iBout = 1 : nBouts
        startFrm = obj.startFrameFromBoutIndex_(iBout) ;
        endFrm = obj.endFrameFromBoutIndex_(iBout) ;
        conf = obj.extremeConfidenceFromBoutIndex_(iBout) ;
        isSingleFrame = (startFrm == endFrm) ;
        if isMultiTarget
          tgt = obj.targetIndexFromBoutIndex_(iBout) ;
          if isSingleFrame
            result{iBout} = sprintf('Frm %d  Tgt %d  Conf %.3f', startFrm, tgt, conf) ;
          else
            result{iBout} = sprintf('Frm %d-%d  Tgt %d  Conf %.3f', startFrm, endFrm, tgt, conf) ;
          end
        else
          if isSingleFrame
            result{iBout} = sprintf('Frm %d  Conf %.3f', startFrm, conf) ;
          else
            result{iBout} = sprintf('Frm %d-%d  Conf %.3f', startFrm, endFrm, conf) ;
          end
        end
      end
    end  % function

    function [frameIndex, trackletIndex, targetIndex] = frameTrackletAndTargetIndexFromBoutIndex(obj, boutIndex)
      % Return the extreme-confidence frame and tracklet index for the given bout.
      frameIndex = obj.extremeFrameFromBoutIndex_(boutIndex) ;
      trackletIndex = obj.trackletIndexFromBoutIndex_(boutIndex) ;
      targetIndex = obj.targetIndexFromBoutIndex_(boutIndex) ;
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
       obj.overallMaxConfidence_] = ...
        confidenceBoutsFromTrkFile(trkFile, obj.quantileConfidenceThreshold_, obj.isConfidenceLackThereof_) ;
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
      % obj.isLaden_ = false ;
      obj.isFresh_ = true ;
    end  % function
  end  % methods
end  % classdef
