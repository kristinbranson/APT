classdef TrackingVisualizerTracklets < TrackingVisualizerBase
  % Tracklet visualization
  % - landmarks via TVMT (TrackingVisualizerMT)
  % - trx/target label via tvtrx (TrackingVisualizerTrxMA)
  %
  % Non-gobject model state lives on the associated
  % TrackingVisualizerTrackletsModel (accessed via obj.tvm_).

  properties
    parent_ % LabelerController reference
    tvm_ % TrackingVisualizerTrackletsModel reference, set by creator

    tvmt % scalar TrackingVisualizerMT (view)
    tvtrx % scalar TrackingVisualizerTrxMA (view)

    hud % AxisHUD (view)
  end

  methods
    function obj = TrackingVisualizerTracklets(parent, tvm)
      % Construct a TrackingVisualizerTracklets.
      %
      % parent: LabelerController
      % tvm: TrackingVisualizerTrackletsModel

      if nargin == 0
        return
      end

      obj.parent_ = parent ;
      obj.tvm_ = tvm ;

      obj.tvmt = TrackingVisualizerMT(parent, tvm.tvmt) ;
      obj.tvtrx = TrackingVisualizerTrxMA(parent, tvm.tvtrx) ;
      obj.hud = parent.currImHud ;
    end

    function vizInit(obj, varargin)
      % Initialize graphics handles and cosmetics.
      ntgtmax = myparse(varargin,...
        'ntgtmax',20 ...
        );

      tvm = obj.tvm_ ;
      tvm.init(ntgtmax) ;

      obj.tvmt.vizInit('ntgts', ntgtmax) ;
      obj.tvtrx.init(@(iTrx)(obj.didSelectTrx(iTrx)), tvm.ntrxmax) ;
      obj.hud.updateReadoutFields('hasTrklet', true) ;
    end

    function newFrame(obj, frm)
      % Display tracking results for given/new frame.

      tvm = obj.tvm_ ;
      ptrx = tvm.ptrx ;
      if isempty(ptrx)
        return;
      end

      [xy, tfeo, iTrx, iTrx2Viz2iTrxNew] = tvm.newFrame(frm) ;
      trxMappingChanged = ~isequal(iTrx2Viz2iTrxNew, tvm.iTrxViz2iTrx) ;
      % Note: newFrame already set tvm.iTrxViz2iTrx

      tvtrx = obj.tvtrx ; %#ok<*PROPLC>
      tvtrx_primary = find(iTrx2Viz2iTrxNew == tvm.currTrklet) ;
      tvmt_primary = tvtrx_primary ;
      if isempty(tvtrx_primary)
        tvtrx_primary = 0 ;
      end

      % update tvmt
      obj.tvmt.updateTrackRes(xy, tfeo) ;
      obj.tvmt.updatePrimary(tvmt_primary) ;

      % update tvtrx
      tvtrx.updatePrimaryTrx(tvtrx_primary) ;
      tvtrx.updateLiveTrx(ptrx(iTrx), frm, trxMappingChanged) ;
    end
    function iTrxViz = iTrx2iTrxViz(obj, iTrx)
       [~, iTrxViz] = ismember(iTrx, obj.tvm_.iTrxViz2iTrx) ;
    end

    function didSelectTrx(obj, iTrxViz)
      % Callback when a trx marker is clicked.
      tvm = obj.tvm_ ;      
      tvm.setSelectedTrackletFromITrxViz(iTrxViz) ;
    end

    function updateSelectedTrxID(obj)
      % Update the view to reflect the current tracklet selection in the model.
      tvm = obj.tvm_ ;
      iTrklet = tvm.currTrklet ;
      trkletID = tvm.ptrx(iTrklet).id ;
      nTrkletTot = numel(tvm.ptrx) ;
      obj.hud.updateTrklet(trkletID, nTrkletTot) ;
      obj.parent_.updateTimelineTraces() ;
      iviz = find(tvm.iTrxViz2iTrx == iTrklet) ;
      if isempty(iviz)
        warning('This should not happen. Not setting primary trx') ;
      else
        obj.tvtrx.updatePrimaryTrx(iviz) ;
        obj.tvmt.updatePrimary(iviz) ;
      end
    end

    function centerPrimary(obj)
      % Center the view on the primary target.
      lObj = obj.parent_.labeler_ ;
      tvm = obj.tvm_ ;
      currframe = lObj.currFrame ;
      trx_curr = tvm.ptrx(tvm.currTrklet) ;
      if (currframe < trx_curr.firstframe) || (currframe > trx_curr.endframe)
        warning('Frame is outside current tracklets range. Not centering on the animal') ;
        return;
      end

      ndx_fr = currframe + trx_curr.off ;
      pts_curr = trx_curr.p(:,:,ndx_fr) ;
      if all(isnan(pts_curr(:)))
        warning('No data for primary animal for current frame. Not centering on the animal') ;
        return;
      end
      minx = nanmin(pts_curr(:,1)) ; maxx = nanmax(pts_curr(:,1)) ;  %#ok<NANMAX,NANMIN>
      miny = nanmin(pts_curr(:,2)) ; maxy = nanmax(pts_curr(:,2)) ;  %#ok<NANMAX,NANMIN>

      controller = obj.parent_ ;
      v = controller.videoCurrentAxis() ;
      x_ok = (minx >= (v(1)-1)) && (maxx <= (v(2)+1)) ;
      y_ok = (miny >= (v(3)-1)) && (maxy <= (v(4)+1)) ;
      if x_ok && y_ok
        return;
      end
      controller.videoCenterOn(trx_curr.x(ndx_fr), trx_curr.y(ndx_fr)) ;
    end
    function updatePrimary(obj, iTgtPrimary) %#ok<INUSD>
      % currently unused. Labeler/iTgtPrimary does not know about tracklet
      % indices.
    end
    function setShowOnlyPrimary(obj, tf)
      obj.tvmt.setShowOnlyPrimary(tf) ;
      obj.tvtrx.setShowOnlyPrimary(tf) ;
    end
    function setShowSkeleton(obj, tf)
      obj.tvmt.setShowSkeleton(tf) ;
    end
    function setHideViz(obj, tf)
      obj.tvmt.setHideViz(tf) ;
      obj.tvtrx.setHideViz(tf) ;
    end
    function setHideTextLbls(obj, tf)
      obj.tvmt.setHideTextLbls(tf) ;
    end
    function setAllShowHide(obj, tfHideOverall, tfHideTxtMT, tfShowCurrTgtOnly, tfShowSkel)
      obj.tvmt.setAllShowHide(tfHideOverall, tfHideTxtMT, tfShowCurrTgtOnly, tfShowSkel) ;
      obj.tvtrx.setAllShowHide(tfHideOverall, tfShowCurrTgtOnly) ;
    end
    function initAndUpdateSkeletonEdges(obj, sedges)
      obj.tvmt.initAndUpdateSkeletonEdges(sedges) ;
    end
    function updateLandmarkColors(obj, ptsClrs)
      obj.tvmt.updateLandmarkColors(ptsClrs) ;
    end
    function updateTrajColors(obj)
      obj.tvtrx.updateColors() ;
    end
    function setMarkerCosmetics(obj, pvargs)
      obj.tvmt.setMarkerCosmetics(pvargs) ;
    end
    function setTextCosmetics(obj, pvargs)
      obj.tvmt.setTextCosmetics(pvargs) ;
    end
    function setTextOffset(obj, offsetPx)
      obj.tvmt.setTextOffset(offsetPx) ;
    end
    function updateSkeletonCosmetics(obj)
      obj.tvmt.updateSkeletonCosmetics() ;
    end
    function delete(obj)
      obj.tvmt.delete() ;
      obj.tvtrx.delete() ;
    end
    function deleteGfxHandles(obj) %#ok<MANU>
      % no-op; sub-TVs handle their own handles
    end
  end

end
