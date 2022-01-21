classdef TrackingVisualizerTracklets < TrackingVisualizerBase
  % Tracket visualization
  % - landmarks via TVMT
  % - trx/target label via tvtrx
  
  properties
    tvmt % scalar TrackingVisualizerMT
    tvtrx % scalar TrackingVisualizerTrxMA
    ptrx % ptrx structure: has landmarks in addition to .x, .y
    %frm2trx % [nfrmmax] cell with frm2trx{f} giving iTgts (indices into 
      %.ptrx) that are live 
      % TEMPORARY using frm2trx logical array. cell will be more compact
      % but possibly slower
    
    npts    
    ntrxmax
    
    currTrklet % scalar int; index into .ptrx
    % Maintain this state in Visualizer for now rather than adding to
    % Labeler. Situation is still slightly unclear, although adding here
    % seems best.
    % * currTrklet clearly differs from currTarget (pre-"reconciliation",
    % there is no known correspondence between targets and tracklets)
    % * currTrklet could differ between tracking res vs imported res, as
    % again in general there is no correspondence between two diff sets of 
    % MA tracking.
    
    iTrxViz2iTrx % [ntrxmax] Mapping from trx in .tvtrx -> ptrx.
                 % iTrxViz2Trx(iTrxTV) gives index into .ptrx for live trx,
                 % and 0 for unused trx.
                 
    tfShowTrxTraj = true;
                 
    hud % AxisHUD
    lObj
  end
  
  % 2022 viz, current tracklet, etc
  %
  % currTrklet. specifies current tracklet; current unknown to Labeler
  % tfShowTrxTraj. show tvtrx viz or not. 
  % obj.lObj.showTrxIDLbl. used when tfShowTrxTraj is true
  
  
  methods
    function obj = TrackingVisualizerTracklets(lObj,ptsPlotInfoFld,handleTagPfix)
      obj.tvmt = TrackingVisualizerMT(lObj,ptsPlotInfoFld,handleTagPfix);
      obj.tvtrx = TrackingVisualizerTrxMA(lObj);
      %obj.ptrx = ptrxs;
      obj.npts = lObj.nLabelPoints;
      obj.ntrxmax = 0;
      
      obj.currTrklet = nan;
      obj.iTrxViz2iTrx = zeros(obj.ntrxmax,1);
      obj.hud = lObj.currImHud;
      obj.lObj = lObj;
    end
    function vizInit(obj,varargin)
      ntgtmax = myparse(varargin,...
        'ntgtmax',20 ...
        );
      
      obj.ntrxmax = ntgtmax;
      obj.iTrxViz2iTrx = zeros(obj.ntrxmax,1);
      obj.tvmt.vizInit('ntgts',ntgtmax);
      obj.tvtrx.init(@(iTrx)obj.trxSelected(iTrx),ntgtmax);
      obj.hud.updateReadoutFields('hasTrklet',true);
    end
    function trkInit(obj,trk)
      assert(isscalar(trk) && isa(trk,'TrkFile'));
      % for tracklets, currently single-view
      
      %trk.initFrm2Tlt(obj.lObj.nframes);
      
      % trk.frm2tlt should already be initted
      assert(trk.nframes==obj.lObj.nframes);
      %assert(size(trk.frm2tlt,1)==obj.lObj.nframes);
      
      ptrxs = load_tracklet(trk);
      ptrxs = TrxUtil.ptrxAddXY(ptrxs);
      obj.ptrx = ptrxs;
      %obj.frm2trx = trk.frm2tlt;
    end
    function iTrx = frm2trx(obj,frm)
      assert(numel(frm)==1);
      iTrx = find([obj.ptrx.firstframe]<=frm & [obj.ptrx.endframe]>=frm);
    end
    function newFrame(obj,frm)
      % find live tracklets
      ptrx = obj.ptrx;
      if isempty(ptrx)
        % eg if no tracklets loaded.
        return;
      end
      
%       if isempty(obj.frm2trx)
%         iTrx = [];
%       else
%         iTrx = find(obj.frm2trx(frm,:));
%       end
      iTrx = obj.frm2trx(frm); % remove refs to frm2trx
      
      nTrx = numel(iTrx);
      if nTrx>obj.ntrxmax
        warningNoTrace('Too many targets to display (%d); showing first %d targets.',...
          nTrx,obj.ntrxmax);
        nTrx = obj.ntrxmax;
        iTrx = iTrx(1:nTrx);
      end
      npts = obj.npts;
      
      % get landmarks
      xy = nan(npts,2,nTrx);
      for j=1:nTrx
        ptrxJ = ptrx(iTrx(j));
        idx = frm + ptrxJ.off;
        xy(:,:,j) = ptrxJ.p(:,:,idx);
      end
      
      % update tvmt
      tfeo = false(npts,nTrx);
      obj.tvmt.updateTrackRes(xy,tfeo);
      
      % update tvtrx; call setShow
      nLive = numel(iTrx);
      iTrx2Viz2iTrxNew = zeros(obj.ntrxmax,1);
      iTrx2Viz2iTrxNew(1:nLive) = iTrx;
      trxMappingChanged = ~isequal(iTrx2Viz2iTrxNew,obj.iTrxViz2iTrx);
      obj.iTrxViz2iTrx = iTrx2Viz2iTrxNew;
      
      tvtrx = obj.tvtrx; %#ok<*PROPLC>
      tfUpdateIDs = trxMappingChanged;
      tvtrx.updateLiveTrx(ptrx(iTrx),frm,tfUpdateIDs);
    end
    function trxSelected(obj,iTrx,tfforce)
      if nargin < 3
        tfforce = false;
      end
      
      iTrklet = obj.iTrxViz2iTrx(iTrx);
      if iTrklet~=obj.currTrklet || tfforce
        trkletID = obj.ptrx(iTrklet).id;
  %       nTrkletLive = nnz(obj.iTrxViz2iTrx>0);
        nTrkletTot = numel(obj.ptrx);
        obj.hud.updateTrklet(trkletID,nTrkletTot);        
        obj.currTrklet = iTrklet;
        obj.lObj.gdata.labelTLInfo.newTarget();
      end
    end
    function updatePrimary(obj,iTgtPrimary) %#ok<INUSD>
      % currently unused. this API is used by Labeler. currently 
      % Labeler/iTgtPrimary does not know about tracklet indices; so any
      % index passed in via iTgtPrimary would not be comparable to
      % .currTrklet etc.
    end
    function setShowOnlyPrimary(obj,tf)
      % none
    end
    function setShowSkeleton(obj,tf)
      obj.tvmt.setShowSkeleton(tf);
    end
    function setHideViz(obj,tf)
      obj.tvmt.setHideViz(tf);
      obj.tvtrx.setHideViz(tf);
    end
    function setAllShowHide(obj,tfHideOverall,tfHideTxtMT,tfShowCurrTgtOnly)
      obj.tvmt.setAllShowHide(tfHideOverall,tfHideTxtMT,tfShowCurrTgtOnly);
      obj.tvtrx.setAllShowHide(tfHideOverall,tfShowCurrTgtOnly);
    end
    function initAndUpdateSkeletonEdges(obj,sedges)
      obj.tvmt.initAndUpdateSkeletonEdges(sedges);
    end
    function updateLandmarkColors(obj,ptsClrs)
      obj.tvmt.updateLandmarkColors(ptsClrs);
    end
    function updateTrajColors(obj)
      obj.tvtrx.updateColors();      
    end
%     function updateShowHideTraj(obj)
%       % relies on lObj.showTrx. Yea, this is confused
%     end
    function setMarkerCosmetics(obj,pvargs)
      % landmarks only
      obj.tvmt.setMarkerCosmetics(pvargs);
    end
    function setTextCosmetics(obj,pvargs)
      % landmark text
      obj.tvmt.setTextCosmetics(pvargs);
    end
    function setTextOffset(obj,offsetPx)
      % MT landmark text only
      obj.tvmt.setTextOffset(offsetPx);
    end
    function setHideTextLbls(obj,tf)
      obj.tvmt.setHideTextLbls(tf);
    end
%     function setHideTrajTextLbls(obj,tf)
%       obj.tvtrx.setHideTextLbls(tf);
%     end
    function delete(obj)
      obj.tvmt.delete();
      obj.tvtrx.delete();
    end
    function deleteGfxHandles(obj)
%       if ~isstruct(obj.hXYPrdRed) % guard against serialized TVs which have PV structs in .hXYPrdRed
%         deleteValidHandles(obj.hXYPrdRed);
%         obj.hXYPrdRed = [];
%       end
%       deleteValidHandles(obj.hXYPrdRedTxt);
%       obj.hXYPrdRedTxt = [];
%       deleteValidHandles(obj.hSkel);
%       obj.hSkel = [];
%       deleteValidHandles(obj.hPch);
%       obj.hPch = [];
%       deleteValidHandles(obj.hPchTxt);
%       obj.hPchTxt = [];
    end
  end
  
end