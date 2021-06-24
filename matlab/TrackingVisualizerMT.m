classdef TrackingVisualizerMT < TrackingVisualizerBase
  
  % TrackingVisualizerMT
  % Like TrackingVisualizer, but can handles/display results for many 
  % targets at once

  properties 
    lObj % Included only to access the current raw image. Ideally used as little as possible

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler
    
    trk % scalar trkfile, views merged. See TrackingVisualizerBase, Frame 
        % updates, loaded trakcing results

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    %ptsPlotInfo % lObj.labelPointsPlotInfo
    mrkrReg % char, regular marker 
    mrkrOcc % char, marker for est-occ
    ptClrs % [nptsx3].
    
    txtOffPx % scalar, px offset for landmark text labels 

    tfHideViz % scalar, true if tracking res hidden
    tfHideTxt % scalar, if true then hide text even if tfHideViz is false
    
    tfShowPch % scalar, if true then show pches    
    tfShowSkel % etc
    
    % besides colors, txtOffPx, the the show/hide state, other cosmetic 
    % state is stored just in the various graphics handles.
    %
    % Note that at least one visibility flag must be stored outside the
    % handles themselves, since text and markers+text can be independently 
    % shown/hidden.
        
    handleTagPfix % char, prefix for handle tags
    
    hXYPrdRed; % [npts x ntgt] plot handles for tracking results, current 
            % frame. This includes 'primary' target as well as all others.
            % 
            % Theoretically, ntgt here is 'number of displayed targets' and
            % this needs not match lObj.nTargets.
    hXYPrdRedTxt; % [nPts x ntgt] handle vec, text labels for hXYPrdRed
    hSkel   % [nEdge x ntgt] handle vec, skeleton line handles
    
    hPch  % [ntgt] handle vec
    hPchTxt % [ntgt] text/lbl for pch
    doPch % if false, don't draw pches at all
    pchColor = [0.3 0.3 0.3];
    pchFaceAlpha = 0.15;
    
    iTgtPrimary % [nprimary] tgt indices for 'primary' targets. 
                % Primariness might typically be eg 'current' but it 
                % doesn't have to correspond.
    showOnlyPrimary=false % logical scalar
    
    iTgtHide % [nhide] tgt indices for hidden targets. 
                
    skelEdgeColor = [.7,.7,.7];
  end
  properties (Constant)
    SAVEPROPS = {'ipt2vw' 'ptClrs' 'txtOffPx' 'tfHideViz' 'tfHideTxt' ...
      'handleTagPfix'};
    LINE_PROPS_COSMETIC_SAVE = {'Color' 'LineWidth' 'Marker' ...
      'MarkerEdgeColor' 'MarkerFaceColor' 'MarkerSize'};
    TEXT_PROPS_COSMETIC_SAVE = {'FontSize' 'FontName' 'FontWeight' 'FontAngle'};
    
    CMAP_DARKEN_BETA = -0.5;
    MRKR_SIZE_FAC = 0.6;
    
  end
  properties (Dependent)
    nPts
    nTgts
  end
  methods
    function v = get.nPts(obj)
      v = numel(obj.ipt2vw);
    end
    function v = get.nTgts(obj)
      v = size(obj.hXYPrdRed,2);
    end
  end
  
  methods
    function deleteGfxHandles(obj)
      if ~isstruct(obj.hXYPrdRed) % guard against serialized TVs which have PV structs in .hXYPrdRed
        deleteValidHandles(obj.hXYPrdRed);
        obj.hXYPrdRed = [];
      end
      deleteValidHandles(obj.hXYPrdRedTxt);
      obj.hXYPrdRedTxt = [];
      deleteValidHandles(obj.hSkel);
      obj.hSkel = [];
      deleteValidHandles(obj.hPch);
      obj.hPch = [];
      deleteValidHandles(obj.hPchTxt);
      obj.hPchTxt = [];
    end
    function [markerPVs,textPVs,pchTextPVs] = ...
                                  convertLabelerCosmeticPVs(obj,pppi)
      % convert .ptsPlotInfo from labeler to that used by this obj
      %
      % The point being that for eg MA labeling, we want smaller markers
      % etc on other targets.

      markerPVs = obj.convertLabelerMarkerPVs(pppi.MarkerProps);
      textPVs = obj.convertLabelerTextPVs(pppi.TextProps);
      pchTextPVs = struct('FontSize',round(textPVs.FontSize*2.0));
    end
    function markerPVs = convertLabelerMarkerPVs(obj,markerPVs)
      sizefac = TrackingVisualizerMT.MRKR_SIZE_FAC;
      markerPVs.MarkerSize = round(markerPVs.MarkerSize*sizefac);
      markerPVs.PickableParts = 'none';
    end      
    function textPVs = convertLabelerTextPVs(obj,textPVs)
      sizefac = TrackingVisualizerMT.MRKR_SIZE_FAC;
      textPVs.FontSize = round(textPVs.FontSize*sizefac);
      textPVs.PickableParts = 'none'; 
    end

    function vizInit(obj,varargin)
      % trk: TrkFile
      %
      % See TrackingVisualizerBase
      % See "Construction/Init notes" below

      [postload,ntgts] = myparse(varargin,...
        'postload',false, ... % see Construction/Init notes
        'ntgts',[] ... % optionally provide known number/max of targets
        );      
      
      obj.deleteGfxHandles();
      
      pppi = obj.lObj.labelPointsPlotInfo; %predPointsPlotInfo;
      %obj.ptsPlotInfo = pppi;
      obj.mrkrReg = pppi.MarkerProps.Marker;
      obj.mrkrOcc = pppi.OccludedMarker;
      
      npts = numel(obj.ipt2vw);
      if isempty(ntgts)
        ntgts = obj.lObj.nTargets;
      end
      if postload
        ptclrspppi = obj.ptClrs;
      else
        ptclrs = obj.lObj.LabelPointColors;
        %ptclrs = brighten(ptclrs,TrackingVisualizerMT.CMAP_DARKEN_BETA);
        obj.ptClrs = ptclrs;
        obj.txtOffPx = pppi.TextOffset;
      end
      szassert(ptclrs,[npts 3]);      

      % init .xyVizPlotArgs*
      [markerPVs,textPVs,pchTextPVs] = obj.convertLabelerCosmeticPVs(pppi);
      markerPVscell = struct2paramscell(markerPVs);
      textPVscell = struct2paramscell(textPVs);
      
      if postload
        % We init first with markerPVs/textPVs, then set saved custom PVs
        hXYPrdRed0 = obj.hXYPrdRed;
        hXYPrdRedTxt0 = obj.hXYPrdRedTxt;
      end
      
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      ipt2set = obj.lObj.labeledposIPt2Set;
      hTmp = gobjects(npts,ntgts);
      hTxt = gobjects(npts,ntgts);
      pfix = obj.handleTagPfix;
      for iTgt = 1:ntgts
      for iPt = 1:npts
        clr = ptclrs(iPt,:);
        iVw = ipt2View(iPt);
        ptset = ipt2set(iPt);
        hTmp(iPt,iTgt) = plot(ax(iVw),nan,nan,markerPVscell{:},...
          'Color',clr,...
          'Tag',sprintf('%s_XYPrdRed_%d_%d',pfix,iPt,iTgt));
        hTxt(iPt,iTgt) = text(nan,nan,num2str(ptset),...
          'Parent',ax(iVw),...
          'Color',clr,textPVscell{:},...
          'Tag',sprintf('%s_PrdRedTxt_%d_%d',pfix,iPt,iTgt));
      end
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedTxt = hTxt;
      
      obj.initSkeletonEdges(obj.lObj.skeletonEdges);
      
      if obj.doPch
        hPc = gobjects(1,ntgts);
        hPchT = gobjects(1,ntgts);
        clr = obj.pchColor;
        alp = obj.pchFaceAlpha;
        for iTgt = 1:ntgts
          hPc(iTgt) = patch(ax,nan,nan,clr,...
            'FaceAlpha',alp,...
            'PickableParts','none',...
            'Tag',sprintf('%s_Pch_%d',pfix,iTgt));
          hPchT(iTgt) = text(nan,nan,num2str(iTgt),...
            'Parent',ax(iVw),...
            'Color',[0 0 0],...
            'fontsize',pchTextPVs.FontSize,...
            'fontweight','bold',...
            'Tag',sprintf('%s_PchTxt_%d',pfix,iTgt),...
            'userdata',iTgt,...
            'ButtonDownFcn',@(s,e)obj.cbkPchTextBDF(s,e));
        end
        obj.hPch = hPc;
        obj.hPchTxt = hPchT;
      end
      
      if postload
        if isstruct(hXYPrdRed0)
          if numel(hXYPrdRed0)==numel(hTmp)
            arrayfun(@(x,y)set(x,y),hTmp,hXYPrdRed0);
          else
            warningNoTrace('.hXYPrdRed: Number of saved prop-val structs does not match number of line handles.');
          end
        end
        if isstruct(hXYPrdRedTxt0)
          if numel(hXYPrdRedTxt0)==numel(hTxt)
            arrayfun(@(x,y)set(x,y),hTxt,hXYPrdRedTxt0);
          else
            warningNoTrace('.hXYPrdRedTxt: Number of saved prop-val structs does not match number of line handles.');
          end
        end
      end
      
      obj.tfShowPch = false;
      obj.tfShowSkel = obj.lObj.showSkeleton;
      
      % default textPVs do not respect .tfHideViz/.tfHideTxt
      obj.updateShowHideAll(); 
      
      obj.iTgtPrimary = zeros(1,0);
      obj.iTgtHide = zeros(1,0);
      
      obj.vizInitHook();
    end
    function vizInitHook(obj)
      % overload me
    end
    function trkInit(obj,trk)
      assert(isscalar(trk) && isa(trk,'TrkFile'));
      % trk.frm2tlt should already be initted
      assert(trk.nframes==obj.lObj.nframes);
      %assert(size(trk.frm2tlt,1)==obj.lObj.nframes);
      obj.trk = trk;
    end
    function initSkeletonEdges(obj,sedges)
      % Creates/inits .hSkel graphics handles appropriately for edge-set
      % sedges
      
      nEdge = size(sedges,1);
      ntgts = obj.nTgts;
      ipt2View = obj.ipt2vw;
      ax = obj.hAxs;
      pppi = obj.lObj.predPointsPlotInfo;

      deleteValidHandles(obj.hSkel);
      obj.hSkel = gobjects(nEdge,ntgts);
      
      skelClr = obj.skelEdgeColor;
      for iTgt = 1:ntgts
        for ie = 1:nEdge
          edge = sedges(ie,:);
          ivws = ipt2View(edge);
          assert(all(ivws==ivws(1)),'Skeleton edge crosses multiple views.');
          ax = ax(ivws(1));
          % cf LabelCore.initSkeletonEdge
          obj.hSkel(ie,iTgt) = plot(ax,nan(2,1),nan(2,1),'-',...
            'Color',skelClr,...
            'PickableParts','none',...
            'Tag',sprintf('TrackingVisualizerMT_Skel_%d_%d',ie,iTgt),...
            'LineWidth',pppi.MarkerProps.LineWidth);           
        end
      end
    end
    function initAndUpdateSkeletonEdges(obj,sedges)
      % Inits skel edges and sets their posns based on current hXYPrdRed.
      
      obj.initSkeletonEdges(sedges);
      
      h = obj.hXYPrdRed;
      hSkl = obj.hSkel;
      ntgts = obj.nTgts;
      for iTgt=1:ntgts
        x = get(h(:,iTgt),{'XData'});
        y = get(h(:,iTgt),{'YData'});
        xytgt = [cell2mat(x) cell2mat(y)];
        tfOccld = any(isnan(xytgt),2);
        LabelCore.setSkelCoords(xytgt,tfOccld,hSkl(:,iTgt),sedges);
      end
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
      
      % 'overall' on/offness
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
           
      if ~isempty(obj.hXYPrdRed) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hXYPrdRed(:,tfTgtOnHideAffected).Visible] = deal(onoffViz);
        [obj.hXYPrdRed(:,~tfTgtOnHideAffected).Visible] = deal('off');
        [obj.hXYPrdRedTxt(:,tfTgtOnHideAffected).Visible] = deal(onoffTxt);
        [obj.hXYPrdRedTxt(:,~tfTgtOnHideAffected).Visible] = deal('off');
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
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
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
        roi = obj.lObj.maGetRoi(xy);
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
        tfeo = [];
      end
      
      ntgtsgiven = size(xy,3);
      npts = obj.nPts;
      ntgts = obj.nTgts;
      assert(ntgtsgiven<=ntgts);
      assert(isempty(tfeo)||ntgtsgiven==size(tfeo,2));
      
      skelEdges = obj.lObj.skeletonEdges;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      hSkl = obj.hSkel;
      hP = obj.hPch;
      hPT = obj.hPchTxt;
      dx = obj.txtOffPx;
      xyoff = xy+dx;
      for iTgt=1:ntgts
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
            roi = obj.lObj.maGetRoi(xytgt);
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
      
      [obj.hXYPrdRed(:,iTgtHide0).Visible] = deal(onoffVizH0);
      [obj.hXYPrdRedTxt(:,iTgtHide0).Visible] = deal(onoffTxtH0);
      [obj.hXYPrdRed(:,iTgtHide).Visible] = deal('off');
      [obj.hXYPrdRedTxt(:,iTgtHide).Visible] = deal('off');
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
        hXY = obj.hXYPrdRed;        
        for iTgt=1:ntgts
          xy = cell2mat(get(hXY(:,iTgt),{'XData' 'YData'}));
          roi = obj.lObj.maGetRoi(xy);
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
        set(obj.hXYPrdRed(iPt,:),'Color',clr);
        set(obj.hXYPrdRedTxt(iPt,:),'Color',clr);
      end
      obj.ptClrs = ptsClrs;
    end
    function setMarkerCosmetics(obj,pvargs)
      if isstruct(pvargs)
        pvargs = obj.convertLabelerMarkerPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRed);
        obj.mrkrReg = pvargs.Marker;
      else
        assert(false);
        %arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRed);
      end
    end
    function setTextCosmetics(obj,pvargs)
      if isstruct(pvargs)
        pvargs = obj.convertLabelerTextPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRedTxt);        
      else
        assert(false);
        %arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRedTxt);
      end
    end
    function setTextOffset(obj,offsetPx)
      obj.txtOffPx = offsetPx; 
      
      npts = obj.nPts;
      ntgts = obj.nTgts;
      
      h = obj.hXYPrdRed;
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
    %   .hXYPrdRed and .hXYPrdRedTxt
    %   - PostLoadInit->vizInit sets up cosmetic state on handles
    %
    % Save/load strategy. (This is for the Labeler auxiliary trkRes)
    %
    % In saveobj we record the cosmetics used for a TrackingVisualizer for 
    % the .hXYPrdRed line handles by doing a get and saving the resulting 
    % PVs in .hXYPrdRed; similarly for .hXYPrdRedTxt.
    %
    % Loadobj keeps these PVs in .hXYPrdRed and .hxYPrdRedTxt. At 
    % postLoadInit->vizInit('postload',true) time, the PVs are re-set on 
    % the .hXYPrdRed line handles. In this way, serialized TVs can keep
    % arbitrary customized cosmetics.
    
    function obj = TrackingVisualizerMT(lObj,handleTagPfix)
      obj.tfHideTxt = false;
      obj.tfHideViz = false;         

      if nargin==0
        return;
      end
      
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
      obj.ipt2vw = lObj.labeledposIPt2View;    
      
      obj.handleTagPfix = handleTagPfix;
    end
    function postLoadInit(obj,lObj)
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;

      assert(isequal(obj.ipt2vw,lObj.labeledposIPt2View));
      
      obj.vizInit('postload',true);
    end
    function delete(obj)
      obj.deleteGfxHandles();
    end
    function s = saveobj(obj)
      s = struct();
      for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
        s.(p) = obj.(p);
      end
      
      lineprops = obj.LINE_PROPS_COSMETIC_SAVE;
      vals = get(obj.hXYPrdRed,lineprops); % [nhandle x nprops]
      s.hXYPrdRed = cell2struct(vals,lineprops,2);
      
      textprops = obj.TEXT_PROPS_COSMETIC_SAVE;
      vals = get(obj.hXYPrdRedTxt,textprops); % [nhandle x nprops]
      s.hXYPrdRedTxt = cell2struct(vals,textprops,2);
    end
  end
  methods (Static)
    function b = loadobj(a)
      if isstruct(a)
        b = TrackingVisualizer();
        for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
          b.(p) = a.(p);
        end
        b.hXYPrdRed = a.hXYPrdRed;
        if isfield(a,'hXYPrdRedTxt')
          b.hXYPrdRedTxt = a.hXYPrdRedTxt;
        end
      else
        b = a;
      end
    end
  end
end