classdef UncertainFramesModel < handle
  % Holds computed data about frames with low tracker confidence for the
  % current movie.

  properties (Access=private)  % private by convention
    labeler_  % back-reference to Labeler (Transient in spirit)
    frameIndexFromPairIndex_  % [N x 1] frame numbers
    tragletIndexFromPairIndex_  % [N x 1] traglet indices (into TrkFile)
    targetIndexFromPairIndex_  % [N x 1] target indices (for navigation)
    minConfidenceFromPairIndex_  % [N x 1] min-confidence values
    isValid_ = false  % scalar logical, true if data has been computed
    isVisible_ = false  % scalar logical, whether the UFC figure is visible
  end

  properties (Dependent)
    isValid
    isVisible
  end

  properties (Dependent, Hidden)
    listboxString  % cellstr for listbox display
  end

  methods
    function obj = UncertainFramesModel(labeler)
      % Construct an UncertainFramesModel with a back-reference to the
      % given Labeler.
      obj.labeler_ = labeler ;
      obj.frameIndexFromPairIndex_ = zeros(0, 1) ;
      obj.tragletIndexFromPairIndex_ = zeros(0, 1) ;
      obj.targetIndexFromPairIndex_ = zeros(0, 1) ;
      obj.minConfidenceFromPairIndex_ = zeros(0, 1) ;
    end  % function

    function result = get.isValid(obj)
      % Return whether computed data is valid.
      result = obj.isValid_ ;
    end  % function

    function result = get.isVisible(obj)
      % Return whether the uncertain-frames figure is visible.
      result = obj.isVisible_ ;
    end  % function

    function set.isVisible(obj, newValue)
      obj.isVisible_ = newValue ;
      obj.syncFromPredictions_() ;
    end  % function

    function result = get.listboxString(obj)
      % Return a cellstr suitable for display in a listbox.
      if ~obj.isValid_
        result = {} ;
        return
      end
      nPairs = numel(obj.frameIndexFromPairIndex_) ;
      result = cell(nPairs, 1) ;
      isMultiTarget = obj.labeler_.hasTrx || obj.labeler_.maIsMA ;
      for iPair = 1 : nPairs
        frm = obj.frameIndexFromPairIndex_(iPair) ;
        conf = obj.minConfidenceFromPairIndex_(iPair) ;
        if isMultiTarget
          tgt = obj.targetIndexFromPairIndex_(iPair) ;
          result{iPair} = sprintf('Frm %d  Tgt %d  Conf %.3f', frm, tgt, conf) ;
        else
          result{iPair} = sprintf('Frm %d  Conf %.3f', frm, conf) ;
        end
      end
    end  % function

    function [frameIndex, tragletIndex] = frameAndTragletIndexFromPairIndex(obj, pairIndex)
      frameIndex = obj.frameIndexFromPairIndex(pairIndex) ;
      tragletIndex = obj.tragletIndexFromPairIndex(pairIndex) ;
    end  % function
  end  % methods

  methods (Access=private)
    function syncFromPredictions_(obj)
      % Compute the most uncertain frame-traglet pairs from the current
      % movie's tracking results.
      if ~obj.isVisible_ 
        return
      end

      labeler = obj.labeler_ ;

      if labeler.currMovie == 0
        obj.clear_() ;
        return
      end

      trkResAll = labeler.trkResGTaware ;
      if isempty(trkResAll) || size(trkResAll, 1) < labeler.currMovie
        obj.clear_() ;
        return
      end

      trkFile = trkResAll{labeler.currMovie, 1, 1} ;
      if isempty(trkFile) || ~isa(trkFile, 'TrkFile') || ~trkFile.hasdata()
        obj.clear_() ;
        return
      end

      if ~isprop(trkFile, 'pTrkConf')
        obj.clear_() ;
        return
      end

      allFrames = [] ;
      allTraglets = [] ;
      allTargets = [] ;
      allMinConf = [] ;

      nTraglets = trkFile.ntracklets ;
      for iTlt = 1 : nTraglets
        [xy, ~, fr, aux] = trkFile.getPTrkTgt(iTlt, 'auxflds', {'pTrkConf'}) ;
        if isempty(xy) || isempty(aux)
          continue
        end
        % aux is [npt x numfrm x 1 x 1]
        confPerPointAndFrame = squeeze(aux) ;  % [npt x numfrm] or [numfrm] if npt==1
        if isvector(confPerPointAndFrame)
          confPerPointAndFrame = confPerPointAndFrame(:)' ;  % ensure [1 x numfrm]
        end
        minConfPerFrame = min(confPerPointAndFrame, [], 1) ;  % [1 x numfrm]
        minConfPerFrame = minConfPerFrame(:) ;  % [numfrm x 1]
        fr = fr(:) ;  % [numfrm x 1]

        % Filter out NaN confidence frames
        isFinite = isfinite(minConfPerFrame) ;
        fr = fr(isFinite) ;
        minConfPerFrame = minConfPerFrame(isFinite) ;
        if isempty(fr)
          continue
        end

        targetIndex = trkFile.pTrkiTgt(iTlt) ;

        nFrames = numel(fr) ;
        allFrames = [allFrames ; fr] ;  %#ok<AGROW>
        allTraglets = [allTraglets ; repmat(iTlt, nFrames, 1)] ;  %#ok<AGROW>
        allTargets = [allTargets ; repmat(targetIndex, nFrames, 1)] ;  %#ok<AGROW>
        allMinConf = [allMinConf ; minConfPerFrame] ;  %#ok<AGROW>
      end

      if isempty(allFrames)
        obj.clear_() ;
        return
      end

      % Sort ascending by min confidence, keep top 100
      [~, sortOrder] = sort(allMinConf, 'ascend') ;
      nKeepMax = 100 ;
      nKeep = min(nKeepMax, numel(sortOrder)) ;
      keepIndices = sortOrder(1:nKeep) ;

      obj.frameIndexFromPairIndex_ = allFrames(keepIndices) ;
      obj.tragletIndexFromPairIndex_ = allTraglets(keepIndices) ;
      obj.targetIndexFromPairIndex_ = allTargets(keepIndices) ;
      obj.minConfidenceFromPairIndex_ = allMinConf(keepIndices) ;
      obj.isValid_ = true ;

      obj.labeler_.notify('updateUncertainFrames') ;
    end  % function

    function clear_(obj)
      % Reset to empty state and notify listeners.
      obj.frameIndexFromPairIndex_ = zeros(0, 1) ;
      obj.tragletIndexFromPairIndex_ = zeros(0, 1) ;
      obj.targetIndexFromPairIndex_ = zeros(0, 1) ;
      obj.minConfidenceFromPairIndex_ = zeros(0, 1) ;
      obj.isValid_ = false ;
      obj.labeler_.notify('updateUncertainFrames') ;
    end  % function
  end  % methods
end  % classdef
