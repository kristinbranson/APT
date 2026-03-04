classdef TrackingVisualizerTrackletsModel < TrackingVisualizerModel
% Model layer for TrackingVisualizerTracklets.
%
% Holds non-gobject properties extracted from TrackingVisualizerTracklets:
% ptrx tracklet data, mapping arrays, sub-model references, etc.

  properties
    lObj % Labeler reference

    tvmt % scalar TrackingVisualizerMTModel
    tvtrx % scalar TrackingVisualizerTrxMAModel

    ptrx % ptrx structure: has landmarks in addition to .x, .y

    currTrklet % scalar int; index into .ptrx

    npts % scalar int
    ntrxmax % scalar int

    iTrxViz2iTrx % [ntrxmax] mapping from trx in tvtrx -> ptrx

    tfShowTrxTraj = true

    hudModel % AxisHUDModel
  end

  methods
    function obj = TrackingVisualizerTrackletsModel(lObj, ptsPlotInfoFld, handleTagPfix)
      % Construct a TrackingVisualizerTrackletsModel.

      if nargin == 0
        return
      end

      obj.tvmt = TrackingVisualizerMTModel(lObj, ptsPlotInfoFld, handleTagPfix, ...
                                           'skel_linestyle', '-') ;
      obj.tvtrx = TrackingVisualizerTrxMAModel(lObj) ;
      obj.npts = lObj.nLabelPoints ;
      obj.ntrxmax = 0 ;

      obj.currTrklet = nan ;
      obj.iTrxViz2iTrx = zeros(obj.ntrxmax, 1) ;
      obj.hudModel = AxisHUDModel() ;
      obj.lObj = lObj ;
    end  % function

    function init(obj, ntgtmax)
      % Initialize model for given target capacity.
      if nargin < 2
        ntgtmax = 20 ;
      end
      obj.ntrxmax = ntgtmax * 2 ;
      obj.iTrxViz2iTrx = zeros(obj.ntrxmax, 1) ;
      obj.hudModel.hasTrklet = true ;
    end  % function

    function trkInit(obj, trk)
      % Initialize tracking data from a TrkFile.  Builds ptrx.
      assert(isscalar(trk) && isa(trk, 'TrkFile')) ;
      assert(trk.nframes == obj.lObj.nframes) ;

      ptrxs = load_tracklet(trk) ;
      ptrxs = TrxUtil.ptrxAddXY(ptrxs) ;
      obj.ptrx = ptrxs ;

      obj.tvmt.trkInit(trk) ;
    end  % function

    function iTrx = frm2trx(obj, frm)
      % Compute which tracklets are live at frame frm.
      assert(numel(frm) == 1) ;
      iTrx = find([obj.ptrx.firstframe] <= frm & [obj.ptrx.endframe] >= frm) ;
    end  % function

    function [xy, tfeo, iTrx, iTrx2Viz2iTrxNew] = newFrame(obj, frm)
      % Compute per-frame tracklet data for the TV to render.
      ptrx = obj.ptrx ;
      if isempty(ptrx)
        xy = [] ;
        tfeo = [] ;
        iTrx = [] ;
        iTrx2Viz2iTrxNew = zeros(obj.ntrxmax, 1) ;
        return
      end

      iTrx = obj.frm2trx(frm) ;
      nTrx = numel(iTrx) ;
      if nTrx > obj.ntrxmax
        isalive = false(1, nTrx) ;
        for n = 1:nTrx
          trxn = iTrx(n) ;
          isalive(n) = ~isnan(ptrx(trxn).x(frm + ptrx(trxn).off)) ;
        end
        isalive = find(isalive) ;
        if numel(isalive) > obj.ntrxmax
          warningNoTrace('Number of targets to display (%d) is much more than max number of animals (%d). Showing first %d targets.', ...
            nTrx, obj.ntrxmax, obj.ntrxmax) ;
          nTrx = obj.ntrxmax ;
          iTrx = iTrx(isalive(1:nTrx)) ;
        else
          iTrx = iTrx(isalive) ;
          nTrx = numel(isalive) ;
        end
      end
      npts = obj.npts ;

      % get landmarks
      xy = nan(npts, 2, nTrx) ;
      tfeo = false(npts, nTrx) ;
      has_occ = isfield(ptrx, 'pocc') ;
      sel_pts = min(npts, size(ptrx(1).p, 1)) ;
      for j = 1:nTrx
        ptrxJ = ptrx(iTrx(j)) ;
        if ~isempty(ptrxJ.p)
          idx = frm + ptrxJ.off ;
          xy(1:sel_pts, :, j) = ptrxJ.p(1:sel_pts, :, idx) ;
          if has_occ
            tfeo(1:sel_pts, j) = ptrxJ.pocc(1:sel_pts, idx) ;
          end
        end
      end

      nLive = numel(iTrx) ;
      iTrx2Viz2iTrxNew = zeros(obj.ntrxmax, 1) ;
      iTrx2Viz2iTrxNew(1:nLive) = iTrx ;
      obj.iTrxViz2iTrx = iTrx2Viz2iTrxNew ;
    end  % function

    function iTrxViz = iTrx2iTrxViz(obj, iTrx)
      % Map a tracklet index to its visualization index.
      [~, iTrxViz] = ismember(iTrx, obj.iTrxViz2iTrx) ;
    end  % function
  end  % methods

end  % classdef
