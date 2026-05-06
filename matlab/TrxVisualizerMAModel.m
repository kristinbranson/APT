classdef TrxVisualizerMAModel < TrackingVisualizerModel
% Model layer for TrxVisualizerMA.
%
% Holds non-gobject properties extracted from TrxVisualizerMA:
% trajectory parameters, color state, and show/hide flags.

  properties
    lObj % Labeler reference

    showTrxPreNFrm = 15 % number of preceding frames to show in traj
    showTrxPostNFrm = 5 % number of following frames to show in traj

    currTrx % index into hTrx; 0 <-> none currently
    nTrxLive % current number of live trx

    clrsTrx % [nTrx x 3] colors
    clrTrxCurrent % [1x3] color for current trx

    trxClickable = true

    tfHideViz = false
    showOnlyPrimary = false
  end

  methods
    function obj = TrxVisualizerMAModel(lObj)
      % Construct a TrxVisualizerMAModel.
      if nargin == 0
        return
      end
      obj.lObj = lObj ;
    end  % function

    function trkInit(obj, trk) %#ok<INUSD>
      % no-op for TrxMA model; trx data comes via updateLiveTrx
    end  % function

    function didSetCurrFrame(obj, frm) %#ok<INUSD>
      % no-op; frame updates driven by TrackingVisualizerTrackletsModel
    end  % function
  end  % methods

end  % classdef
