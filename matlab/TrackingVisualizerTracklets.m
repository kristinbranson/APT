classdef TrackingVisualizerTracklets < handle
  % Tracket visualization
  % - landmarks via TVMT
  % - trx/target label via tvtrx
  
  properties
    tvmt % scalar TrackingVisualizerMT
    tvtrx % scalar TrackingVisualizerTrx
    ptrx % ptrx structure: has landmarks in addition to .x, .y
    frm2trx % [nfrmmax] cell with frm2trx{f} giving iTgts (indices into 
      %.ptrx) that are live 
      % TEMPORARY using frm2trx logical array. cell will be more compact
      % but possibly slower
    
    npts    
    ntrxmax
    
    idTrxLive % [ntgtslive] currently live trx
  end
  
  methods
    function obj = TrackingVisualizerTracklets(lObj,ntrxmax,handleTagPfix)
      obj.tvmt = TrackingVisualizerMT(handleTagPfix);
      obj.tvtrx = TrackingVisualizerTrx(lObj);
      %obj.ptrx = ptrxs;
      obj.npts = lObj.nLabelPoints;
      obj.ntrxmax = ntrxmax;
      
      obj.idTrxLive = [];
    end
    function vizInit(obj,nfrmmax,ptrxs,varargin)
      ntgt = obj.ntrxmax;
      obj.tvmt.vizInit('ntgts',ntgt);
      obj.tvtrx.init(false,ntgt);
      
      obj.ptrx = ptrxs;
      obj.frm2trx = Labeler.trxHlpComputeF2t(nfrmmax,ptrxs);
    end
    function newFrame(obj,frm)
      % find live tracklets
      iTrx = find(obj.frm2trx(frm,:));
      nTrx = numel(iTrx);
      if nTrx>obj.ntrxmax
        warningNoTrace('Too many targets to display (%d); showing first %d targets.',...
          nTrx,obj.ntrxmax);
        nTrx = obj.ntrxmax;
        iTrx = iTrx(1:nTrx);
      end
      npts = obj.npts;
      
      % get landmarks
      ptrx = obj.ptrx;
      p = nan(2*npts,nTrx);
      for j=1:nTrx
        ptrxJ = ptrx(iTrx(j));
        idx = frm + ptrxJ.off;
        p(:,j) = ptrxJ.p(:,idx);
      end
      
      % update tvmt
      xy = reshape(p,npts,2,nTrx);
      tfeo = false(npts,nTrx);
      obj.tvmt.updateTrackRes(xy,tfeo);
      
      % update tvtrx; call setShow
      ids0 = obj.idTrxLive;
      ids1 = [ptrx(iTrx).id]+1; %#ok<*PROPLC>
      tfUpdateIDs = ~isequal(ids0,ids1);
      obj.idTrxLive = ids1;
      nLive = numel(ids1);
      
      tvtrx = obj.tvtrx;
      tfShow = false(tvtrx.nTrx,1);
      tfShow(1:nLive) = true; 
      tvtrx.setShow(tfShow);
      tvtrx.updateTrxCore(ptrx(iTrx),frm,tfShow,0,tfUpdateIDs);
    end
    function updatePrimaryTarget(obj,iTgtPrimary)
      % todo; currently no pred/target selection
    end
    function setShowSkeleton(obj,tf)
      obj.tvmt.setShowSkeleton(tf);
    end
    function setAllShowHide(obj,tfHide,tfHideTxt,tfShowCurrTgtOnly)
      % xxx landmarks only
      obj.tvmt.setAllShowHide(tfHide,tfHideTxt,tfShowCurrTgtOnly);
    end
    function initAndUpdateSkeletonEdges(obj,sedges)
      obj.tvmt.initAndUpdateSkeletonEdges(sedges);
    end
    function updateLandmarkColors(obj,ptsClrs)
      obj.tvmt.updateLandmarkColors(ptsClrs);
    end
    function setMarkerCosmetics(obj,pvargs)
      % landmarks only
      obj.tvmt.setMarkerCosmetics(pvargs);
    end
    function setTextCosmetics(obj,pvargs)
      % landmark text
      obj.tvmt.setTextCosmetics(pvargs);
    end
    function setTextOffset(obj,offsetPx)
      % xxx currently only landmark text
      obj.tvmt.setTextOffset(offsetPx);
    end
    function setHideTextLbls(obj,tf)
      % xxx currently only landmark text
      obj.tvmt.setHideTextLbls(tf);
    end

  end
  
end