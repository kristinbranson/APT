classdef TrackingVisualizerMTFastModel < TrackingVisualizerModel
% Model layer for TrackingVisualizerMTFast.
%
% Holds non-gobject properties extracted from TrackingVisualizerMTFast:
% tracking data, current-frame xy cache, show/hide flags, etc.

  properties
    lObj % Labeler reference

    trk % scalar TrkFile, views merged
    xyCurr % [npts x 2 x nTgts] current pred coords
    occCurr % [npts x nTgts] logical, current occludedness
    xyCurrITgts % [nTgts] target indices/labels for 3rd dim of xyCurr

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptsPlotInfoFld % eg 'labelPointsPlotInfo'

    mrkrReg % char, regular marker
    mrkrOcc % char, marker for est-occ
    txtOffPx % scalar, px offset for landmark text labels
    skelEdges % [nEdges x 2], skeleton edges
    skelIDataPt1 % pre-computed indexing prop for fast skeleton update
    skelIDataPt2 % "

    tfHideViz % scalar logical
    tfHideTxt % scalar logical
    tfShowOnlyPrimary % logical scalar
    tfShowPch % scalar logical
    tfShowSkel % scalar logical

    handleTagPfix % char, prefix for handle tags

    doPch = false % logical

    iTgtPrimary % [nprimary] target indices
  end

  properties (Dependent)
    nPts
    nTgts
  end

  methods
    function v = get.nPts(obj)
      v = numel(obj.ipt2vw) ;
    end  % function
    function v = get.nTgts(obj)
      v = size(obj.xyCurr, 3) ;
    end  % function
  end  % methods

  methods
    function obj = TrackingVisualizerMTFastModel(lObj, ptsPlotInfoField, handleTagPfix)
      % Construct a TrackingVisualizerMTFastModel.

      if nargin == 0
        return
      end

      obj.lObj = lObj ;
      obj.ipt2vw = lObj.labeledposIPt2View ;
      obj.ptsPlotInfoFld = ptsPlotInfoField ;

      obj.tfHideTxt = false ;
      obj.tfHideViz = false ;
      obj.tfShowOnlyPrimary = false ;
      obj.tfShowPch = false ;
      obj.tfShowSkel = false ;

      obj.handleTagPfix = handleTagPfix ;
    end  % function

    function trkInit(obj, trk)
      % Initialize tracking data from a TrkFile.
      assert(isscalar(trk) && isa(trk, 'TrkFile')) ;
      assert(trk.nframes == obj.lObj.nframes) ;
      obj.trk = trk ;
    end  % function

    function [tfhaspred, xy, tfocc] = newFrame(obj, frm)
      % Return per-frame tracking data and update internal xy cache.
      [tfhaspred, xy, tfocc] = obj.trk.getPTrkFrame(frm, 'collapse', true) ;
      itgts = find(tfhaspred) ;
      obj.xyCurr = xy(:, :, tfhaspred) ;
      obj.occCurr = tfocc(:, tfhaspred) ;
      obj.xyCurrITgts = itgts ;
    end  % function
  end  % methods

end  % classdef
