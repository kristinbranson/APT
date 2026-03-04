classdef TrackingVisualizerMTModel < TrackingVisualizerModel
% Model layer for TrackingVisualizerMT.
%
% Holds non-gobject properties extracted from TrackingVisualizerMT: tracking
% data, point colors, show/hide flags, cosmetic state, etc.

  properties
    lObj % Labeler reference

    trk % scalar TrkFile, views merged

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptsPlotInfoFld % eg 'labelPointsPlotInfo'
    mrkrReg % char, regular marker
    mrkrOcc % char, marker for est-occ
    ptClrs % [npts x 3]

    txtOffPx % scalar, px offset for landmark text labels

    tfHideViz % scalar logical
    tfHideTxt % scalar logical

    tfShowPch % scalar logical
    tfShowSkel % scalar logical

    handleTagPfix % char, prefix for handle tags

    iTgtPrimary % [nprimary] target indices for 'primary' targets
    showOnlyPrimary = false % logical scalar

    iTgtHide % [nhide] target indices for hidden targets

    skel_linestyle = '-'
    doPch % logical
    pchColor = [0.3 0.3 0.3]
    pchFaceAlpha = 0.15
  end

  properties (Dependent)
    nPts
  end

  methods
    function v = get.nPts(obj)
      v = numel(obj.ipt2vw) ;
    end  % function
  end  % methods

  methods
    function obj = TrackingVisualizerMTModel(lObj, ptsPlotInfoField, handleTagPfix, varargin)
      % Construct a TrackingVisualizerMTModel.

      obj.tfHideTxt = false ;
      obj.tfHideViz = false ;

      if nargin == 0
        return
      end

      [skel_linestyle] = myparse(varargin, 'skel_linestyle', '-') ;
      obj.lObj = lObj ;
      obj.ipt2vw = lObj.labeledposIPt2View ;
      obj.ptsPlotInfoFld = ptsPlotInfoField ;
      obj.handleTagPfix = handleTagPfix ;
      obj.skel_linestyle = skel_linestyle ;
    end  % function

    function trkInit(obj, trk)
      % Initialize tracking data from a TrkFile.
      assert(isscalar(trk) && isa(trk, 'TrkFile')) ;
      assert(trk.nframes == obj.lObj.nframes) ;
      obj.trk = trk ;
    end  % function

    function [tfhaspred, xy, tfocc] = newFrame(obj, frm)
      % Return per-frame tracking data for the TV to render.
      [tfhaspred, xy, tfocc] = obj.trk.getPTrkFrame(frm, 'collapse', true) ;
    end  % function
  end  % methods

end  % classdef
