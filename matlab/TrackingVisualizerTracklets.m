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
                 % Also applies to .tvmt, ie iTrxViz2Trx are trklet (.ptrx)
                 % labels for .tvmt.hXYPrdRed(1,:)
                 
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
      obj.tvmt = TrackingVisualizerMTFast(lObj,ptsPlotInfoFld,handleTagPfix,...
        'skel_linestyle','-','plot_type','pred');
      obj.tvtrx = TrackingVisualizerTrxMAFast(lObj);
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
        'ntgtmax',100 ...
        );
      
      obj.ntrxmax = ntgtmax*2;
      obj.iTrxViz2iTrx = zeros(obj.ntrxmax,1);
      obj.tvmt.vizInit('ntgts',ntgtmax);
      obj.tvtrx.init(@(iTrx)obj.trxSelected(iTrx),obj.ntrxmax);
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
      
      iTrx = obj.frm2trx(frm);
      
      nTrx = numel(iTrx);
      % This shouldn't be required with fast visualizers
      % if nTrx>obj.ntrxmax
      %   isalive = false(1,nTrx);
      %   for n=1:nTrx
      %     trxn = iTrx(n);
      %     isalive(n) = ~isnan(ptrx(trxn).x(frm+ptrx(trxn).off));
      %   end
      %   isalive = find(isalive);
      %   if numel(isalive)>nTrx
      %     warningNoTrace('Number of targets to display (%d) is much more than max number of animals (%d). Showing first %d targets.',...
      %       nTrx,obj.ntrxmax,obj.ntrxmax);
      % 
      %     nTrx = obj.ntrxmax;
      %     iTrx = iTrx(isalive(1:nTrx));
      %   else
      %     iTrx = iTrx(isalive);
      %     nTrx = numel(isalive);
      %   end
      % end
      npts = obj.npts;
      
      % get landmarks
      xy = nan(npts,2,nTrx);
      tfeo = false(npts,nTrx);
      has_occ = isfield(ptrx,'pocc');
      sel_pts = min(npts,size(ptrx(1).p,1));
      for j=1:nTrx
        ptrxJ = ptrx(iTrx(j));
        if ~isempty(ptrxJ.p)
          idx = frm + ptrxJ.off;
          xy(1:sel_pts,:,j) = ptrxJ.p(1:sel_pts,:,idx);
          if has_occ
            tfeo(1:sel_pts,j) = ptrxJ.pocc(1:sel_pts,idx);
          end
        end
      end

      nLive = numel(iTrx);
      iTrx2Viz2iTrxNew = zeros(obj.ntrxmax,1);
      iTrx2Viz2iTrxNew(1:nLive) = iTrx;
      trxMappingChanged = ~isequal(iTrx2Viz2iTrxNew,obj.iTrxViz2iTrx);
      obj.iTrxViz2iTrx = iTrx2Viz2iTrxNew;
      
      tvtrx = obj.tvtrx; %#ok<*PROPLC>
      tvtrx_primary = find(iTrx2Viz2iTrxNew==obj.currTrklet);
      tvmt_primary = tvtrx_primary; % could be empty
      if isempty(tvtrx_primary)
        tvtrx_primary = 0;
      end

      % update tvmt
      obj.tvmt.updateTrackRes(xy,tfeo);
      obj.tvmt.updatePrimary(tvmt_primary);
      
      % update tvtrx; call setShow
      % tvtrx.updatePrimaryTrx(tvtrx_primary);
      tvtrx.updateLiveTrx(ptrx(iTrx),frm,trxMappingChanged);
    end
    function iTrxViz = iTrx2iTrxViz(obj,iTrx)
       [~,iTrxViz] = ismember(iTrx,obj.iTrxViz2iTrx);
    end
    function trxSelected(obj,iTrxViz,tfforce)
      % This method is passed to .tvtrx and used as a callback
      % iTrxViz: scalar (1..obj.tvtrx.nTrx) index of the active trx in .tvtrx. 
      %          will not be 0, will not be empty
      % If selecting from an external client with a tracklet ID (index into
      % .ptrx), use trxSelectedTrxID below
      if nargin < 3
        tfforce = false;
      end
      
      iTrklet = obj.iTrxViz2iTrx(iTrxViz);
      if iTrklet~=obj.currTrklet || tfforce
        trkletID = obj.ptrx(iTrklet).id;
  %       nTrkletLive = nnz(obj.iTrxViz2iTrx>0);
        nTrkletTot = numel(obj.ptrx);
        obj.hud.updateTrklet(trkletID,nTrkletTot);        
        obj.currTrklet = iTrklet;
        iTrx = obj.frm2trx(obj.lObj.currFrame);
        obj.tvtrx.updateLiveTrx(obj.ptrx(iTrx),obj.lObj.currFrame,true) %already called
        obj.tvmt.updatePrimary(iTrxViz);
        obj.lObj.gdata.labelTLInfo.updateLabels('doRecompute',true);
      end
    end
    function trxSelectedTrxID(obj,iTrklet,tfforce)
      % This uses actual trx id. compare with the above fn
      if nargin < 3
        tfforce = false;
      end      
      
      if iTrklet~=obj.currTrklet || tfforce
        trkletID = obj.ptrx(iTrklet).id;
  %       nTrkletLive = nnz(obj.iTrxViz2iTrx>0);
        nTrkletTot = numel(obj.ptrx);
        obj.hud.updateTrklet(trkletID,nTrkletTot);        
        obj.currTrklet = iTrklet;
        obj.lObj.gdata.labelTLInfo.updateLabels();
        iviz = find(obj.iTrxViz2iTrx==iTrklet);
        if isempty(iviz)
          warning('This should not happen. Not setting primary trx');
        else
          obj.tvtrx.updatePrimaryTrx(iviz);
          obj.tvmt.updatePrimary(iviz);
        end
      end
    end    
    function centerPrimary(obj)
      %use whatever the current zoom to center on the primary target if it
      %is outside current video
      lobj = obj.lObj;
      currframe = lobj.currFrame;
      trx_curr = obj.ptrx(obj.currTrklet);
      if (currframe<trx_curr.firstframe)|| (currframe>trx_curr.endframe)
        warning('Frame is outside current tracklets range. Not centering on the animal');
        return;
      end
      
      ndx_fr = currframe + trx_curr.off;
      pts_curr = trx_curr.p(:,:,ndx_fr);      
      if all(isnan(pts_curr(:)))
        warning('No data for primary animal for current frame. Not centering on the animal');
        return; 
      end
      minx=nanmin(pts_curr(:,1)); maxx=nanmax(pts_curr(:,1));
      miny=nanmin(pts_curr(:,2)); maxy=nanmax(pts_curr(:,2));      

      v = lobj.controller_.videoCurrentAxis();
      x_ok = (minx>= (v(1)-1))&(maxx<=(v(2)+1));
      y_ok = (miny>= (v(3)-1))&(maxy<=(v(4)+1));
      if x_ok && y_ok
        return;
      end
      lobj.controller_.videoCenterOn(trx_curr.x(ndx_fr),trx_curr.y(ndx_fr));      
    end
    function updatePrimary(obj,iTgtPrimary) %#ok<INUSD>
      % currently unused. this API is used by Labeler. currently 
      % Labeler/iTgtPrimary does not know about tracklet indices; so any
      % index passed in via iTgtPrimary would not be comparable to
      % .currTrklet etc.
    end
    function setShowOnlyPrimary(obj,tf)
      % * "primary" <-> currTrklet; maintained internally (eg selected by
      % .tvtrx click) and not by Labeler. primary index is concurrently
      % maintained in .tvmt, .tvtrx

      obj.tvmt.setShowOnlyPrimary(tf); 
      obj.tvtrx.setShowOnlyPrimary(tf);
      % Note, this hides trx as well; so to change tracklets, must use
      % Switch Targets UI
    end
    function setShowSkeleton(obj,tf)
      obj.tvmt.setShowSkeleton(tf);
    end
    function setHideViz(obj,tf)
      obj.tvmt.setHideViz(tf);
      obj.tvtrx.setHideViz(tf);
    end
    function setAllShowHide(obj,tfHideOverall,tfHideTxtMT,tfShowCurrTgtOnly,tfShowSkel)
      obj.tvmt.setAllShowHide(tfHideOverall,tfHideTxtMT,tfShowCurrTgtOnly,tfShowSkel);
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
    function skeletonCosmeticsUpdated(obj)
      obj.tvmt.skeletonCosmeticsUpdated();
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
%         deleteValidGraphicsHandles(obj.hXYPrdRed);
%         obj.hXYPrdRed = [];
%       end
%       deleteValidGraphicsHandles(obj.hXYPrdRedTxt);
%       obj.hXYPrdRedTxt = [];
%       deleteValidGraphicsHandles(obj.hSkel);
%       obj.hSkel = [];
%       deleteValidGraphicsHandles(obj.hPch);
%       obj.hPch = [];
%       deleteValidGraphicsHandles(obj.hPchTxt);
%       obj.hPchTxt = [];
    end
  end
  
end
