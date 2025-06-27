classdef TrackingVisualizerMTFast < TrackingVisualizerBase

  % XXX TODO: occ mrkrs
  % xxx todo: primary skel/hPred?
  
  properties 
    lObj 

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler
    
    trk % scalar trkfile, views merged. See TrackingVisualizerBase, Frame 
        % updates, loaded tracking results
    xyCurr % [npts x 2 x nTgts] current pred coords (posns assigned to 
           % .XData, .YData of hPred, hSkel). nTgts can be anything
    occCurr % [npts x nTgts] logical, current occludedness
    xyCurrITgts % [nTgts] target indices/labels for 3rd dim of xyCurr. Used 
                % for cross-referencing with .iTgtPrimary.

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptsPlotInfoFld % eg 'labelPointsPlotInfo'
    

    % "Convenience" lObj props
    % Besides these props, other cosmetics are currently maintained on 
    % lObj.(ptsPlotInfoFld). Note this is a simplification where this
    % TrackingVisualizer can't have its own custom cosmetics as in
    % "Auxiliary Tracking Results".
    mrkrReg % char, regular marker. convenience prop used when tracking pred is occ
    mrkrOcc % char, marker for est-occ. etc.    
    txtOffPx % scalar, px offset for landmark text labels 
    skelEdges % like lObj.skeletonEdges,[[nEdges x 2]. applies to all tgts
    skelIDataPt1 % pre-computed indexing prop for fast skeleton update
    skelIDataPt2 % "

    tfHideViz % scalar, true if tracking res hidden
    tfHideTxt % scalar, if true then hide text even if tfHideViz is false
    tfShowOnlyPrimary % logical scalar    
    tfShowPch % scalar, if true then show pches    
    tfShowSkel % etc
            
    handleTagPfix % char, prefix for handle tags
    
    %%% GFX handles %%%
    
    hPred; % [npts] plot handles for tracking results, current 
               % frame. hPred(ipt) shows landmark ipt across all 
               % targets, primary or otherwise (so these cannot have 
               % different cosmetics)
    hPredTxt; % [nPts] handle vec, text labels for hPred
              % currently showing only for primary
    
    % Could have hSkelPrimary and hSkelOther
    hSkel   % [1xnview] skeleton line handle (all edges/tgts)
            % format of .XData, .YData: see setSkelCoords
    
    %hPch  % [ntgt] handle vec
    
    %hPchTxt % [ntgt] text/lbl for pch
    
    doPch = false % if false, don't draw pches at all
    %pch
    %pchColor = [0.3 0.3 0.3];
    %pchFaceAlpha = 0.15;
    
    iTgtPrimary % [nprimary] tgt indices for 'primary' targets. 
                % Primariness might typically be eg 'current' but it 
                % doesn't have to correspond. 
                %
                % any use of iTgtPrimary is done by looking up against
                % .xyCurrITgts.
                % 
                % * for SA-trx, iTgtPrimary operates in trx-space; this
                % is the same as the "tracklet space" of .trk
                % * for MA, TrackingVisualizerTracklets should be in use,
                % not this class (for now).
    
    %iTgtHide % [nhide] tgt indices for hidden targets. 
                
    %skelEdgeColor = [.7,.7,.7];
  end
%   properties (Constant)
%     SAVEPROPS = {'ipt2vw' 'ptClrs' 'txtOffPx' 'tfHideViz' 'tfHideTxt' ...
%       'handleTagPfix' 'ptsPlotInfoFld'};
%     LINE_PROPS_COSMETIC_SAVE = {'Color' 'LineWidth' 'Marker' ...
%       'MarkerEdgeColor' 'MarkerFaceColor' 'MarkerSize'};
%     TEXT_PROPS_COSMETIC_SAVE = {'FontSize' 'FontName' 'FontWeight' 'FontAngle'};
%     
%     CMAP_DARKEN_BETA = -0.5;
%     MRKR_SIZE_FAC = 0.6;
%     
%   end
  properties (Dependent)
    nPts
    nTgts
  end
  methods
    function v = get.nPts(obj)
      v = numel(obj.ipt2vw);
    end
    function v = get.nTgts(obj)
      v = size(obj.xyCurr,3);
    end
  end  
  
  methods
    function obj = TrackingVisualizerMTFast(lObj,ptsPlotInfoField,handleTagPfix)

      if nargin==0
        return;
      end
      
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
      %obj.trk = []; % initted in trkInit
      obj.ipt2vw = lObj.labeledposIPt2View;      
      obj.ptsPlotInfoFld = ptsPlotInfoField;      
      %obj.mrkrReg , mrkrOcc, txtOffPx; etc
      
      obj.tfHideTxt = false;
      obj.tfHideViz = false;
      obj.tfShowOnlyPrimary = false;
      obj.tfShowPch = false;
      obj.tfShowSkel = false;

      obj.handleTagPfix = handleTagPfix;    
    end
    
    function deleteGfxHandles(obj)
      if ~isstruct(obj.hPred) % guard against serialized TVs which have PV structs in .hPred
        deleteValidGraphicsHandles(obj.hPred);
        obj.hPred = [];
      end
      deleteValidGraphicsHandles(obj.hPredTxt);
      obj.hPredTxt = [];
      deleteValidGraphicsHandles(obj.hSkel);
      obj.hSkel = [];
    end
      
    function delete(obj)
      obj.deleteGfxHandles();
    end
  
    function vizInit(obj,varargin)
      % plot gfx handles
      %
      % cosmetics handling:
      % 1. gfx handles cosmetics initted from lObj.(ptsPlotInfoFld).
      % 2. Some convenience props for cosmetics similarly initted.
      % 3. Subsequent adjustments to cosmetics must update gfx handles and
      % (if needed) convenience props as well
      % 4. Currently show/hide viz state is NOT set here, for no particular
      % reason. Clients who call this should almost definitely call
      % setShowHideAll(), etc.
      
      obj.deleteGfxHandles();
      
      pppiFld = obj.ptsPlotInfoFld;
      pppi = obj.lObj.(pppiFld);
      
      obj.mrkrReg = pppi.MarkerProps.Marker;
      obj.mrkrOcc = pppi.OccludedMarker;
      obj.txtOffPx = pppi.TextOffset;
      obj.skelEdges = obj.lObj.skeletonEdges;
      
      npts = numel(obj.ipt2vw);
      ptclrs = obj.lObj.Set2PointColors(pppi.Colors);
      szassert(ptclrs,[npts 3]);      

      % init .xyVizPlotArgs*
      [markerPVs,textPVs,pchTextPVs,skelPVs] = obj.convertLabelerCosmeticPVs(pppi);
      markerPVscell = struct2paramscell(markerPVs);
      textPVscell = struct2paramscell(textPVs);
      skelPVs = struct2paramscell(skelPVs);
            
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      ipt2set = obj.lObj.labeledposIPt2Set;
      hTmp = gobjects(npts,1);
      hTxt = gobjects(npts,1);
      pfix = obj.handleTagPfix;
      for ipt = 1:npts
        clr = ptclrs(ipt,:);
        ivw = ipt2View(ipt);
        ptset = ipt2set(ipt);
        hTmp(ipt) = plot(ax(ivw),nan,nan,markerPVscell{:},...
          'Color',clr,...
          'LineStyle','none',...
          'Tag',sprintf('%s_pred_%d',pfix,ipt));
        hTxt(ipt) = text(nan,nan,num2str(ptset),...
          'Parent',ax(ivw),...
          'Color',clr,textPVscell{:},...
          'Tag',sprintf('%s_PrdRedTxt_%d',pfix,ipt));
      end
      obj.hPred = hTmp;
      obj.hPredTxt = hTxt;

      nvw = obj.lObj.nview;
      obj.hSkel = gobjects(1,nvw);
      
      %skelClr = obj.skelEdgeColor;
      for ivw=1:nvw
        ax = ax(ivw);
        % cf LabelCore.initSkeletonEdge
        obj.hSkel(ivw) = plot(ax,nan,nan,'-',...
          'PickableParts','none',...
          'Tag',sprintf('TrackingVisualizerMTFast_Skel'),...
          skelPVs{:});
      end

      assert(~obj.doPch);
      if ~obj.lObj.maIsMA,
        obj.iTgtPrimary = obj.lObj.currTarget;
      else
        obj.iTgtPrimary = zeros(1,0);
      end

      % tf* props are NOT updated here. See comments above      
      %obj.vizInitHook();
    end
%     function vizInitHook(obj)
%       % overload me
%     end
    function trkInit(obj,trk)
      assert(isscalar(trk) && isa(trk,'TrkFile'));
      % trk.frm2tlt should already be initted
      assert(trk.nframes==obj.lObj.nframes);
      %assert(size(trk.frm2tlt,1)==obj.lObj.nframes);
      obj.trk = trk;
    end
    function updateSkel(obj)
      % update obj.hSkel .XData, .YData appropriately per
      %   .sedges, .xyCurr, .tfShowOnlyPrimary
      %
      % effect of .tfShowSkel (and tfHideViz) is controlled via 
      % updateShowHideAll() and 'Visible' prop of .hSkel.
      %
      % Recall .sedges is [e1pt1 e1pt2; e2pt1 e2pt2; ...]
      % format of .XData, .YData: (k edges)
      % .XData = [xy(e1pt1,1,itgt=1) xy(e1pt2,1,itgt=1) nan 
      %           xy(e2pt1,1,itgt=1) xy(e2pt2,1,itgt=1) nan 
      %           ...
      %           xy(ekpt1,1,itgt=1) xy(ekpt2,1,itgt=1) nan
      %           xy(e1pt1,1,itgt=2) xy(e1pt2,1,itgt=2) nan
      %           ...
      %           ]
      % .YData = etc
      
      if obj.tfHideViz || ~obj.tfShowSkel 
        return
      end
      
      assert(isscalar(obj.hSkel),'Multiview support todo.');
      
      xy = obj.xyCurr; % [npt x 2 x ntgt]
      if obj.tfShowOnlyPrimary
        tf = obj.iTgtPrimary==obj.xyCurrITgts;
        xy = xy(:,:,tf);
      end
      
      TrackingVisualizerMTFast.updateSkelStc(...
                          obj.hSkel,obj.skelEdges,obj.nPts,xy);
    end
  end
  methods (Static)
    function updateSkelStc(hSkel,skelEdges,npt,xy,varargin)
      % Set hSkel.XData/.YData per xy
      %
      % hSkel: [nview] graphics handles
      % skelEdges: kx2 [e1pt1 e1pt2; e2pt1 e2pt2; ...; ekpt1 ekpt2]. See
      %   below
      % xy: [npt x 2 x ntgtshow]
      %
      % It is assumed that skelEdges is wrt view 1, and that those edges 
      % apply to all views. Recall currently pts in all views correspond to 
      % the same physical pts.
      
      [linestyle,alpha,linewidth] = myparse(varargin,'linestyle','','alpha',0.5, 'linewidth',0.5);

      se = skelEdges;
      k = size(se,1);
      ntgtshow = size(xy,3);
      totlen = k*ntgtshow*3;
     
      nview = numel(hSkel);
      nptphys = npt/nview;

      % se(:,1) are edge pt1s. when we index into xyview we need to skip
      % nptphys*2 for each successive tgt.
      ixskip = 0:nptphys*2:nptphys*2*ntgtshow-1;
      isept1 = repmat(se(:,1),1,ntgtshow); % index into xy for edge pt1's
      isept2 = repmat(se(:,2),1,ntgtshow); % index into xy for edge pt2's
      ixpt1 = isept1 + ixskip; % auto singleton expansion
      ixpt2 = isept2 + ixskip; % etc
      % se(:,2) are edge pt2s. when we index into xy we need to skip npt
      % to get past x's, then npt*2 for each successive tgt.
      iyskip = nptphys:nptphys*2:nptphys*2*ntgtshow-1;
      iypt1 = isept1 + iyskip;
      iypt2 = isept2 + iyskip;

      idatapt1 = 1:3:totlen;
      idatapt2 = 2:3:totlen;

      for iview=1:nview
        iptview = (1:nptphys) + (iview-1)*nptphys;
        xyview = xy(iptview,:,:);
        xdata = nan(1,totlen);
        ydata = nan(1,totlen);
        xdata(idatapt1) = xyview(ixpt1);
        xdata(idatapt2) = xyview(ixpt2);
        ydata(idatapt1) = xyview(iypt1);
        ydata(idatapt2) = xyview(iypt2);        
        cc = get(hSkel(iview),'Color');
        if numel(cc) == 3
          cc(end+1) = alpha;
        else
          cc(end) = alpha;
        end
        set(hSkel(iview),'XData',xdata,'YData',ydata,'LineStyle',linestyle,...
          'linewidth',linewidth,'Color',cc);

      end
    end
  end
  methods
    function initAndUpdateSkeletonEdges(obj,sedges)
      % In our case we dont need to init the gfx handles.
      obj.skelEdges = sedges;
      obj.updateSkel();
    end
    function setShowSkeleton(obj,tf)
      obj.tfShowSkel = tf;
      obj.updateShowHideAll();
    end
    function setHideViz(obj,tf)
      obj.tfHideViz = tf;
      obj.updateShowHideAll();
    end
    function setHideTextLbls(obj,tf)
      obj.tfHideTxt = tf;
      obj.updateShowHideAll();
    end
    function setShowPches(obj,tf)
      obj.tfShowPch = tf;
      obj.updateShowHideAll();
    end
    function hideOtherTargets(obj)
      obj.setShowOnlyPrimary(true);
    end
    function setShowOnlyPrimary(obj,tf)
      obj.tfShowOnlyPrimary = tf;
      obj.updateShowHideAll();      
    end
    function setAllShowHide(obj,tfHide,tfHideTxt,tfShowCurrTgtOnly,tfShowSkel)
      obj.tfHideViz = tfHide;
      obj.tfHideTxt = tfHideTxt;
      obj.tfShowSkel = tfShowSkel;
      obj.tfShowOnlyPrimary = tfShowCurrTgtOnly;
      obj.updateShowHideAll();      
    end
    function updateShowHideAll(obj)
      
      if ~isempty(obj.hPred) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        onoffViz = onIff(~obj.tfHideViz);
        set(obj.hPred,'Visible',onoffViz);        
        onoffTxt = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
        set(obj.hPredTxt,'Visible',onoffTxt);
        obj.updatePreds();
        obj.updatePredsTxt();
      end
      
      if ~isempty(obj.hSkel)
        onoffSkel = onIff(~obj.tfHideViz && obj.tfShowSkel);
        set(obj.hSkel,'Visible',onoffSkel);
        % because updateSkel() early returns if visible is off
        obj.updateSkel();
      end
    end
    function updatePreds(obj)
      % set obj.hPred .XData, .YData appropriately per .xyCurr, .occCurr, 
      % .xyCurrITgts, and .tfShowOnlyPrimary
      %
      % effect of .tfHideViz is controlled via updateShowHideAll() and
      % 'Visible' prop of .hPred.
      %
      % format of .XData, .YData: (q visible tgts):
      % .XData = [xy(ipt,1,1) xy(ipt,1,2) .. xy(ipt,1,q)]
      % .YData = etc
      
      if obj.tfHideViz 
        return;
      end
      
      xy = obj.xyCurr; % [npt x 2 x ntgt]
      if obj.tfShowOnlyPrimary
        tf = obj.iTgtPrimary==obj.xyCurrITgts;
        xy = xy(:,:,tf);
      end
      
      h = obj.hPred;
      
      if isempty(xy)
        % no data; hide x/y for all pts
        set(h,'XData',nan,'YData',nan);
      else
        npt = obj.nPts;
        for ipt=1:npt
          xdata = xy(ipt,1,:); % [1 1 ntgtshow]
          ydata = xy(ipt,2,:);
          set(h(ipt),'XData',xdata(:),'YData',ydata(:));
        end
      end
    end
    function updatePredsTxt(obj)
      % set obj.hPredTxt .XData, .YData appropriately per .xyCurr,
      % .xyCurrITgts, and .tfShowOnlyPrimary
      %
      % effect of .tfHideViz/.tfHideTxt is controlled via 
      % updateShowHideAll() and 'Visible' prop of .hPredTxt.
      %
      % Currently only shows txt for primary target. If there isn't a
      % primary target nothing is shown!!
      
      if obj.tfHideViz || obj.tfHideTxt
        return;
      end
      
      itgtP = obj.iTgtPrimary;
      itgtXY = obj.xyCurrITgts;
      if isempty(itgtP) || isempty(itgtXY)
        set(obj.hPredTxt,'Position',[nan nan]);
        return;
      end

      tf = itgtP==itgtXY; % includes isnan(itgtP)
      if ~any(tf)
        set(obj.hPredTxt,'Position',[nan nan]);
        return;
      end

      xy = obj.xyCurr; % [npt x 2 x ntgt]
      xy = xy(:,:,tf);
      xypos = xy + obj.txtOffPx;
      h = obj.hPredTxt;
      npt = obj.nPts;
      for ipt=1:npt
        pos = xypos(ipt,:,1); % 3rd dim should be 1 anyway; to be safe take first
        set(h(ipt),'Position',pos(:)');
      end
    end

    function updateTrackRes(obj,xy,tfeo,xyITgts)
      % update .xyCurr, .occCur, .xyCurrITgts, then call updates for gfx
      % handles.
      %
      % xy: [npts x 2 x ntgts] 
      % tfeo: [npts x nTgts] logical for est-occ
      % xyITgts: [nTgts] indices/labels
      %

      if nargin < 3
        [npts,~,ntgts] = size(xy);
        tfeo = false(npts,ntgts);
        xyITgts = (1:ntgts)';
      end
      
      obj.xyCurr = xy;
      obj.occCurr = tfeo;
      obj.xyCurrITgts = xyITgts;      
      obj.updatePreds();
      obj.updatePredsTxt();
      obj.updateSkel();      
    end
    function newFrame(obj,frm)
      [tfhaspred,xy,tfocc] = obj.trk.getPTrkFrame(frm,'collapse',true);
      itgts = find(tfhaspred);
      obj.updateTrackRes(xy(:,:,tfhaspred),tfocc(:,tfhaspred),itgts);
    end
    function updatePrimary(obj,iTgtPrimary)
      iTgtPrimary0 = obj.iTgtPrimary;
      iTgtChanged = ~isequal(iTgtPrimary,iTgtPrimary0);
      obj.iTgtPrimary = iTgtPrimary;
      
      if iTgtChanged
        obj.updateShowHideAll();
      end
    end
%     function updatePches(obj)
%       if obj.doPch
%         ntgts = obj.nTgts;
%         hP = obj.hPch;
%         hPT = obj.hPchTxt;
%         hXY = obj.hPred;        
%         for iTgt=1:ntgts
%           xy = cell2mat(get(hXY(:,iTgt),{'XData' 'YData'}));
%           roi = obj.lObj.maGetLossMask(xy);
%           set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2));  
%           set(hPT(iTgt),'Position',[roi(1,:) 0]);          
%         end
%       end
%     end
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      szassert(ptsClrs,[npts 3]);
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hPred(iPt),'Color',clr);
        set(obj.hPredTxt(iPt),'Color',clr);
      end
      %obj.ptClrs = ptsClrs;
    end
    function setMarkerCosmetics(obj,pvargs)
      if isstruct(pvargs)
        pvargs = obj.convertLabelerMarkerPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hPred);
        obj.mrkrReg = pvargs.Marker;
      else
        assert(false);
        %arrayfun(@(x)set(x,pvargs{:}),obj.hPred);
      end
    end
    function setTextCosmetics(obj,pvargs)
      if isstruct(pvargs)
        pvargs = obj.convertLabelerTextPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hPredTxt);        
      else
        assert(false);
        %arrayfun(@(x)set(x,pvargs{:}),obj.hPredTxt);
      end
    end
    function setTextOffset(obj,offsetPx)
      obj.txtOffPx = offsetPx; 
      obj.updatePredsTxt();
    end    
    function skeletonCosmeticsUpdated(obj)
      ppiFld = obj.ptsPlotInfoFld;
      ppi = obj.lObj.(ppiFld);
      set(obj.hSkel,ppi.SkeletonProps);
    end
%     function cbkPchTextBDF(obj,s,e)
%       iTgt = s.UserData;
%       % lObj was supposed to be used as minimally as possible to access
%       % image data; oops
%       obj.lObj.setTarget(iTgt);
%     end
  end
  
  methods (Static)
    function [markerPVs,textPVs,pchTextPVs,skelPVs] = ...
                                      convertLabelerCosmeticPVs(pppi)
      % convert .ptsPlotInfo from labeler to that used by this obj

      markerPVs = TrackingVisualizerMTFast.convertLabelerMarkerPVs(pppi.MarkerProps);
      textPVs = TrackingVisualizerMTFast.convertLabelerTextPVs(pppi.TextProps);
      pchTextPVs = struct('FontSize',round(textPVs.FontSize*2.0));
      skelPVs = pppi.SkeletonProps;
    end
    function markerPVs = convertLabelerMarkerPVs(markerPVs)
      %sizefac = TrackingVisualizerMT.MRKR_SIZE_FAC;
      %markerPVs.MarkerSize = round(markerPVs.MarkerSize*sizefac);
      markerPVs.PickableParts = 'none';
    end      
    function textPVs = convertLabelerTextPVs(textPVs)
      %sizefac = TrackingVisualizerMT.MRKR_SIZE_FAC;
      %textPVs.FontSize = round(textPVs.FontSize*sizefac);
      textPVs.PickableParts = 'none'; 
    end
  end
  
end