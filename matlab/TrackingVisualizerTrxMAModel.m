classdef TrackingVisualizerTrxMAModel < TrackingVisualizerModel
% Model layer for TrackingVisualizerTrxMA.
%
% Holds non-gobject properties extracted from TrackingVisualizerTrxMA:
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
    trxSelectCbk % function handle

    tfHideViz = false
    showOnlyPrimary = false
  end

  methods
    function obj = TrackingVisualizerTrxMAModel(lObj)
      % Construct a TrackingVisualizerTrxMAModel.
      if nargin == 0
        return
      end
      obj.lObj = lObj ;
    end  % function

    function trkInit(obj, trk) %#ok<INUSD>
      % no-op for TrxMA model; trx data comes via updateLiveTrx
    end  % function

    function newFrame(obj, frm) %#ok<INUSD>
      % no-op; frame updates driven by TrackingVisualizerTrackletsModel
    end  % function
  end  % methods

end  % classdef
