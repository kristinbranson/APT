classdef TrackingVisualizerMTFast < TrackingVisualizerBase

  % XXX TODO: txt lbls
  % XXX TODO: occ mrkrs
  
  properties 
    lObj 

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler
    
    trk % scalar trkfile, views merged. See TrackingVisualizerBase, Frame 
        % updates, loaded trakcing results
    xyCurr % [npts x 2 x nTgts] current pred coords (posns assigned to 
           % .XData, .YData of hPred, hSkel). nTgts can be anything
    occCurr % [npts x nTgts] logical, current occludedness

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
    
    %hPredTxt; % [nPts] handle vec, text labels for hPred
    
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
    function deleteGfxHandles(obj)
      if ~isstruct(obj.hPred) % guard against serialized TVs which have PV structs in .hPred
        deleteValidHandles(obj.hPred);
        obj.hPred = [];
      end
%       deleteValidHandles(obj.hPredTxt);
%       obj.hPredTxt = [];
      deleteValidHandles(obj.hSkel);
      obj.hSkel = [];
      deleteValidHandles(obj.hPch);
      obj.hPch = [];
      deleteValidHandles(obj.hPchTxt);
      obj.hPchTxt = [];
    end
  end
  methods (Static)
    function [markerPVs,textPVs,pchTextPVs] = convertLabelerCosmeticPVs(pppi)
      % convert .ptsPlotInfo from labeler to that used by this obj

      markerPVs = TrackingVisualizerMTFast.convertLabelerMarkerPVs(pppi.MarkerProps);
      textPVs = TrackingVisualizerMTFast.convertLabelerTextPVs(pppi.TextProps);
      pchTextPVs = struct('FontSize',round(textPVs.FontSize*2.0));
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
      [markerPVs,textPVs,pchTextPVs] = obj.convertLabelerCosmeticPVs(pppi);
      markerPVscell = struct2paramscell(markerPVs);
      textPVscell = struct2paramscell(textPVs);
            
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      %ipt2set = obj.lObj.labeledposIPt2Set;
      hTmp = gobjects(npts,1);
      %hTxt = gobjects(npts,ntgts);
      pfix = obj.handleTagPfix;
      for ipt = 1:npts
        clr = ptclrs(ipt,:);
        ivw = ipt2View(ipt);
        %ptset = ipt2set(ipt);
        hTmp(ipt) = plot(ax(ivw),nan,nan,markerPVscell{:},...
          'Color',clr,...
          'Tag',sprintf('%s_pred_%d',pfix,ipt));
%         hTxt(iPt,iTgt) = text(nan,nan,num2str(ptset),...
%           'Parent',ax(iVw),...
%           'Color',clr,textPVscell{:},...
%           'Tag',sprintf('%s_PrdRedTxt_%d_%d',pfix,iPt,iTgt));
      end
      obj.hPred = hTmp;
      %obj.hPredTxt = hTxt;

      nvw = obj.lObj.nview;
      obj.hSkel = gobjects(1,nvw);
      
      skelClr = obj.skelEdgeColor;
      for ivw=1:nvw
          ax = ax(ivw);
          % cf LabelCore.initSkeletonEdge
          obj.hSkel(ivw) = plot(ax,nan,nan,'-',...
            'Color',skelClr,...
            'PickableParts','none',...
            'Tag',sprintf('TrackingVisualizerMTFast_Skel'),...
            'LineWidth',pppi.SkeletonProps.LineWidth);           
      end

      assert(~obj.doPch);
      obj.iTgtPrimary = zeros(1,0);

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
    function setSkelCoords(obj)
      % set obj.hSkel .XData, .YData appropriately per .sedges and .xyCurr
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
      
      se = obj.sedges; % kx2  [e1pt1 e1pt2; e2pt1 e2pt2; ...; ekpt1 ekpt2]
      k = size(se,2);
      npt = obj.nPts;
      ntgt = obj.nTgts;
      xy = obj.xyCurr; % [npt x 2 x ntgt]
      
      totlen = k*ntgt*3;
      xdata = nan(1,totlen);
      ydata = nan(1,totlen);
      
      idatapt1 = 1:3:totlen;
      idatapt2 = 2:3:totlen;
      % se(:,1) are edge pt1s. when we index into xy we need to skip npt*2
      % for each successive tgt.
      ixskip = 0:npt*2:npt*2*ntgt-1;
      isept1 = repmat(se(:,1),1,ntgt); % index into xy for edge pt1's
      isept2 = repmat(se(:,2),1,ntgt); % index into xy for edge pt2's
      ixpt1 = isept1 + ixskip; % auto singleton expansion
      ixpt2 = isept2 + ixskip; % etc
      % se(:,2) are edge pt2s. when we index into xy we need to skip npt
      % to get past x's, then npt*2 for each successive tgt.      
      iyskip = npt:npt*2:npt*2*ntgt-1;
      iypt1 = isept1 + iyskip;
      iypt2 = isept2 + iyskip;
      
      xdata(idatapt1) = xy(ixpt1);
      xdata(idatapt2) = xy(ixpt2);
      ydata(idatapt1) = xy(iypt1);
      ydata(idatapt2) = xy(iypt2);
      
      set(obj.hSkel,'XData',xdata,'YData',ydata);
    end
    function initAndUpdateSkeletonEdges(obj,sedges)
      % In our case we dont need to init the gfx handles.
      obj.skelEdges = sedges;
      obj.setSkelCoords();
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
      obj.showOnlyPrimary = tf;
      obj.updateShowHideAll();      
    end
    function setAllShowHide(obj,tfHide,tfHideTxt,tfShowCurrTgtOnly)
      % xxx stopped here. add setshowskel to this sig; and TVMT; and callsites
      obj.tfHideViz = tfHide;
      obj.tfHideTxt = tfHideTxt;
      obj.showOnlyPrimary = tfShowCurrTgtOnly;
      obj.updateShowHideAll();      
    end
    function updateShowHideAll(obj)
      % update .Visible for 
      % * .hXYPrd* [npts x ntgt]
      % * .hSkel [nedge x ntgt]
      % * .hPch [ntgt]
      %
      % iTgtHide does not apply to skel or Pch. The only client atm is 
      % LabelCoreSeqMA
      
      % 'overall' on/offness      % trk: TrkFile
      %
      % See TrackingVisualizerBase
      % See "Construction/Init notes" below

      onoffViz = onIff(~obj.tfHideViz);
      onoffTxt = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
      
      if obj.showOnlyPrimary
        tfTgtOn = false(1,obj.nTgts);
        tfTgtOn(obj.iTgtPrimary) = true;
      else
        tfTgtOn = true(1,obj.nTgts);
      end
      tfTgtOnHideAffected = tfTgtOn;
      tfTgtOnHideAffected(obj.iTgtHide) = false;
           
      if ~isempty(obj.hPred) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hPred(:,tfTgtOnHideAffected).Visible] = deal(onoffViz);
        [obj.hPred(:,~tfTgtOnHideAffected).Visible] = deal('off');
        [obj.hPredTxt(:,tfTgtOnHideAffected).Visible] = deal(onoffTxt);
        [obj.hPredTxt(:,~tfTgtOnHideAffected).Visible] = deal('off');
      end
      
      % skel, pch: not affected by hide
      if ~isempty(obj.hSkel)
        onoffSkel = onIff(~isempty(obj.hSkel) && ~obj.tfHideViz && obj.tfShowSkel);
        [obj.hSkel(:,tfTgtOn).Visible] = deal(onoffSkel);
        [obj.hSkel(:,~tfTgtOn).Visible] = deal('off');
      end      
      if obj.doPch
        onoffPch = onIff(obj.tfShowPch);
        [obj.hPch(tfTgtOn).Visible] = deal(onoffPch);
        [obj.hPchTxt(tfTgtOn).Visible] = deal(onoffPch);
        [obj.hPch(~tfTgtOn).Visible] = deal('off');
        [obj.hPchTxt(~tfTgtOn).Visible] = deal('off');
      end
    end
    function updateTrackResI(obj,xy,tfeo,iTgt)
      % xy: [npts x 2]
      % tfeo: [npts] logical for est-occ; can be [] to skip
      % iTgt: target index to update
      
      npts = obj.nPts;
      skelEdges = obj.lObj.skeletonEdges;
      h = obj.hPred;
      hTxt = obj.hPredTxt;
      hSkl = obj.hSkel;
      dx = obj.txtOffPx;
      xyoff = xy+dx;

      for iPt=1:npts
        set(h(iPt,iTgt),'XData',xy(iPt,1),'YData',xy(iPt,2));
        set(hTxt(iPt,iTgt),'Position',[xyoff(iPt,:) 0]);
      end
      %pppi = obj.ptsPlotInfo;
      if ~isempty(tfeo)
        tfeo = logical(tfeo);
        set(h(tfeo,iTgt),'Marker',obj.mrkrOcc);
        set(h(~tfeo,iTgt),'Marker',obj.mrkrReg);
      end
      
      tfOccld = any(isinf(xy),2);
      LabelCore.setSkelCoords(xy,tfOccld,hSkl(:,iTgt),skelEdges);
      
      if obj.doPch
        hP = obj.hPch;
        hPT = obj.hPchTxt;
        roi = obj.lObj.maGetLossMask(xy);
        set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2));
        set(hPT(iTgt),'Position',[roi(1,:) 0]);        
      end
    end
    function updateTrackRes(obj,xy,tfeo)
      %
      % xy: [npts x 2 x ntgtsgiven] 
      % tfeo: [npts x ntgtsgiven] logical for est-occ
      %
      % ntgtsgiven must be <= .nTgts. Targets > ntgtsgiven are set to nan
      % locs.
      
      if nargin<3
        tfeo = [];postLoadI
      end
      
      xy = xy(:,:,1);
      tfeo = tfeo(:,1);
      
      ntgtsgiven = size(xy,3);
      npts = obj.nPts;
      ntgts = obj.nTgts;
      assert(ntgtsgiven<=ntgts);
      assert(isempty(tfeo)||ntgtsgiven==size(tfeo,2));
      
      skelEdges = obj.lObj.skeletonEdges;
      h = obj.hPred;
      hTxt = obj.hPredTxt;
      hSkl = obj.hSkel;
      hP = obj.hPch;
      hPT = obj.hPchTxt;
      dx = obj.txtOffPx;
      xyoff = xy+dx;
      for iTgt=1:1
        if iTgt>ntgtsgiven
          set(h(:,iTgt),'XData',nan,'YData',nan);
          set(hTxt(:,iTgt),'Position',[nan nan 0]);          
          LabelCore.setSkelCoords(nan(npts,2),false(npts,1),hSkl(:,iTgt),skelEdges);
          if obj.doPch
            set(hP(iTgt),'XData',nan,'YData',nan);
            set(hPT(iTgt),'Position',[nan nan 0]);
          end
        else
          xytgt = xy(:,:,iTgt);
          
          for iPt=1:npts
            set(h(iPt,iTgt),'XData',xytgt(iPt,1),'YData',xytgt(iPt,2));
            set(hTxt(iPt,iTgt),'Position',[xyoff(iPt,:,iTgt) 0]);
          end        
          if ~isempty(tfeo)
            %pppi = obj.ptsPlotInfo;
            set(h(tfeo(:,iTgt),iTgt),'Marker',obj.mrkrOcc);
            set(h(~tfeo(:,iTgt),iTgt),'Marker',obj.mrkrReg);
          end

          tfOccld = any(isinf(xytgt),2);
          LabelCore.setSkelCoords(xytgt,tfOccld,hSkl(:,iTgt),skelEdges);

          if obj.doPch
            roi = obj.lObj.maGetLossMask(xytgt);
            set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2)); 
            set(hPT(iTgt),'Position',[roi(1,:) 0]);
          end
        end
      end
    end
    function newFrame(obj,frm)
      [tfhaspred,xy,tfocc] = obj.trk.getPTrkFrame(frm);
      obj.updateTrackRes(xy,tfocc);
    end
    function updatePrimary(obj,iTgtPrimary)
      iTgtPrimary0 = obj.iTgtPrimary;
      iTgtChanged = ~isequal(iTgtPrimary,iTgtPrimary0);
      obj.iTgtPrimary = iTgtPrimary;
      
      if iTgtChanged
        trajClrCurr = obj.lObj.projPrefs.Trx.TrajColorCurrent;
        hSkl = obj.hSkel;
        if ~isempty(hSkl)
            if iTgtPrimary0>0
              set(hSkl(:,iTgtPrimary0),'Color',obj.skelEdgeColor);
            end
            if iTgtPrimary>0
              set(hSkl(:,iTgtPrimary),'Color',trajClrCurr);
            end
        end
        if obj.showOnlyPrimary
          obj.updateShowHideAll();
        end
      end
    end
    function updateHideTarget(obj,iTgtHide)
      % unhide/show iTgtHide0, hide iTgtHide
      
      iTgtHide0 = obj.iTgtHide;
      tfnochange = isequal(iTgtHide0,iTgtHide);
      if tfnochange
        return;
      end

      if obj.showOnlyPrimary
        tfTgtHide0on = iTgtHide0==obj.iTgtPrimary;
      else
        tfTgtHide0on = true;
      end
      if tfTgtHide0on
        onoffVizH0 = onIff(~obj.tfHideViz);
        onoffTxtH0 = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
        onoffSkelH0 = onIff(~isempty(obj.hSkel) && ~obj.tfHideViz && obj.tfShowSkel);
      else
        onoffVizH0 = 'off';
        onoffTxtH0 = 'off';
        onoffSkelH0 = 'off';
      end
      
      [obj.hPred(:,iTgtHide0).Visible] = deal(onoffVizH0);
      [obj.hPredTxt(:,iTgtHide0).Visible] = deal(onoffTxtH0);
      [obj.hPred(:,iTgtHide).Visible] = deal('off');
      [obj.hPredTxt(:,iTgtHide).Visible] = deal('off');
      if ~isempty(obj.hSkel)
        [obj.hSkel(:,iTgtHide0).Visible] = deal(onoffSkelH0);
        [obj.hSkel(:,iTgtHide).Visible] = deal('off');
      end
      
      obj.iTgtHide = iTgtHide;
    end
    function updatePches(obj)
      if obj.doPch
        ntgts = obj.nTgts;
        hP = obj.hPch;
        hPT = obj.hPchTxt;
        hXY = obj.hPred;        
        for iTgt=1:ntgts
          xy = cell2mat(get(hXY(:,iTgt),{'XData' 'YData'}));
          roi = obj.lObj.maGetLossMask(xy);
          set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2));  
          set(hPT(iTgt),'Position',[roi(1,:) 0]);          
        end
      end
    end
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      szassert(ptsClrs,[npts 3]);
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hPred(iPt,:),'Color',clr);
        set(obj.hPredTxt(iPt,:),'Color',clr);
      end
      obj.ptClrs = ptsClrs;
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
      
      npts = obj.nPts;
      ntgts = obj.nTgts;
      
      h = obj.hPred;
      x = get(h,'XData');
      y = get(h,'YData');
      x = reshape(cell2mat(x(:)),[npts 1 ntgts]);
      y = reshape(cell2mat(y(:)),[npts 1 ntgts]);
      xy = cat(2,x,y);
      szassert(xy,[npts 2 ntgts]);
      
      obj.updateTrackRes(xy,[]);
    end
    function cbkPchTextBDF(obj,s,e)
      iTgt = s.UserData;
      % lObj was supposed to be used as minimally as possible to access
      % image data; oops
      obj.lObj.setTarget(iTgt);
    end
  end
  
  methods 
    % Construction/Init notes 
    %
    % 1. Call the constructor normally, then vizInit();
    %   - This initializes cosmetics from labeler.predPointsPlotInfo
    %   - This is the codepath used for LabelTrackers. LabelTracker TVs
    %   are not serialized. New/fresh ones are created and cosmetics are
    %   initted from labeler.predPointsPlotInfo.
    % 2. From serialized. Call constructor with no args, then postLoadInit()
    %   - SaveObj restores various cosmetic state, including PV props in
    %   .hPred and .hPredTxt
    %   - PostLoadInit->vizInit sets up cosmetic state on handles
    %
    % Save/load strategy. (This is for the Labeler auxiliary trkRes)
    %
    % In saveobj we record the cosmetics used for a TrackingVisualizer for 
    % the .hPred line handles by doing a get and saving the resulting 
    % PVs in .hPred; similarly for .hPredTxt.
    %
    % Loadobj keeps these PVs in .hPred and .hxYPrdRedTxt. At 
    % postLoadInit->vizInit('postload',true) time, the PVs are re-set on 
    % the .hPred line handles. In this way, serialized TVs can keep
    % arbitrary customized cosmetics.
    
%     function postLoadInit(obj,lObj)
%       obj.lObj = lObj;
%       gd = lObj.gdata;
%       obj.hAxs = gd.axes_all;
%       obj.hIms = gd.images_all;
% 
%       assert(isequal(obj.ipt2vw,lObj.labeledposIPt2View));
%       
%       obj.vizInit('postload',true);
%     end
    function delete(obj)
      obj.deleteGfxHandles();
    end
%     function s = saveobj(obj)
%       s = struct();
%       for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
%         s.(p) = obj.(p);
%       end
%       
%       lineprops = obj.LINE_PROPS_COSMETIC_SAVE;
%       vals = get(obj.hPred,lineprops); % [nhandle x nprops]
%       s.hPred = cell2struct(vals,lineprops,2);
%       
%       textprops = obj.TEXT_PROPS_COSMETIC_SAVE;
%       vals = get(obj.hPredTxt,textprops); % [nhandle x nprops]
%       s.hPredTxt = cell2struct(vals,textprops,2);
%     end
  end
%   methods (Static)
%     function b = loadobj(a)
%       if isstruct(a)
%         b = TrackingVisualizer();
%         for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
%           b.(p) = a.(p);
%         end
%         b.hPred = a.hPred;
%         if isfield(a,'hPredTxt')
%           b.hPredTxt = a.hPredTxt;
%         end
%       else
%         b = a;
%       end
%     end
%   end
end