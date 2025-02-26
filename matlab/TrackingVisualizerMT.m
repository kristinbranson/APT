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
    ptsPlotInfoFld % eg 'labelPointsPlotInfo'
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
    
    % Targets (ntgt)
    % TrackingVisualizerMT contains gfx handles for a fixed number of
    % targets (ntgt). If the number of displayed targets is actually less,
    % then the extra handles have .X/YData set to nan etc.
    % 
    % ntgt can be dynamically increased/decreased via setNumTargets().
    %
    % Note that currently each (pt,tgt) gets its own handle as i) marker
    % type changes based on occludedness; ii) the primary target may have
    % different cosmetics based on labeling state, primary-ness etc; iii)
    % text labels cannot be 'vectorized' across targets.
    %
    % That said, we use a single skeleton handle for optimization as
    % empirically drawing the skeleton appears to be the costliest gfx op.    
    hXYPrdRed; % [npts x ntgt] plot handles for tracking results, current 
            % frame. This includes 'primary' target as well as all others.
            % 
            % Theoretically, ntgt here is 'number of displayed targets' and
            % this needs not match lObj.nTargets.
    hXYPrdRedTxt; % [nPts x ntgt] handle vec, text labels for hXYPrdRed
    hSkel   % [1 x nview]  skeleton line handle (all edges/tgts)
            % format of .XData, .YData: see setSkelCoords
    skel_linestyle = '-'
    hPch  % [1 x ntgt] handle vec
    hPchTxt % [1 x ntgt] text/lbl for pch
    doPch % if false, don't draw pches at all
    pchColor = [0.3 0.3 0.3];
    pchFaceAlpha = 0.15;
    
    iTgtPrimary % [nprimary] tgt indices for 'primary' targets. 
                % Primariness might typically be eg 'current' but it 
                % doesn't have to correspond.
    showOnlyPrimary=false % logical scalar
    
    iTgtHide % [nhide currently must equal 1] tgt indices for hidden targets. 
  end
  properties (Constant)
    SAVEPROPS = {'ipt2vw' 'ptClrs' 'txtOffPx' 'tfHideViz' 'tfHideTxt' ...
      'handleTagPfix' 'ptsPlotInfoFld'};
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
        deleteValidGraphicsHandles(obj.hXYPrdRed);
        obj.hXYPrdRed = [];
      end
      deleteValidGraphicsHandles(obj.hXYPrdRedTxt);
      obj.hXYPrdRedTxt = [];
      deleteValidGraphicsHandles(obj.hSkel);
      obj.hSkel = [];
      deleteValidGraphicsHandles(obj.hPch);
      obj.hPch = [];
      deleteValidGraphicsHandles(obj.hPchTxt);
      obj.hPchTxt = [];
    end  

    function addTgts(obj,ntgtsadd)
      % plots/adds new gfx handles without touching existing
      % Impacted gfx handles: .hXY*, .hPch*
      
      pppiFld = obj.ptsPlotInfoFld;
      pppi = obj.lObj.(pppiFld); 
       [markerPVs,textPVs,pchTextPVs,~] = ...
          TrackingVisualizerMT.convertLabelerCosmeticPVs(pppi);
      markerPVscell = struct2paramscell(markerPVs);
      textPVscell = struct2paramscell(textPVs);

      iTgtOffset = obj.nTgts;
      [hPred,hTxt] = obj.hlpPlotTgts(ntgtsadd,iTgtOffset,markerPVscell,textPVscell);
      [hPch1,hPchT] = obj.hlpPlotPches(ntgtsadd,iTgtOffset,pchTextPVs);
      
      obj.hXYPrdRed = [obj.hXYPrdRed hPred];
      obj.hXYPrdRedTxt = [obj.hXYPrdRedTxt hTxt];
      obj.hPch = [obj.hPch hPch1];
      obj.hPchTxt = [obj.hPchTxt hPchT];
    end
    function ensureNTgts(obj,ntgtsreqd)
      % Ensure that obj can handle ntgtsreqd 
      % Relevant gfx handles: .hXY*, .hPch*
      
      ntgts0 = obj.nTgts;
      if isempty(ntgtsreqd) || isnan(ntgtsreqd) || ntgts0>=ntgtsreqd
        return;
      end
      
      ntgts0 = max(ntgts0,ntgtsreqd);
      
      % increase
      NTGTS_GROWTH_FAC = 2;
      ntgtsadd = ntgts0*(NTGTS_GROWTH_FAC-1);
      ntgtsadd = round(ntgtsadd);
      obj.addTgts(ntgtsadd);
    end
    function [hPred,hTxt] = hlpPlotTgts(obj,ntgtsplot,itgtoffset,...
        markerPVscell,textPVscell)
      % create/plot gfx handles for ntgtsplot targets
      % 
      % itgtoffset: graphics Tags range over itgtoffset+(1:ntgtsplot)
      
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      ipt2set = obj.lObj.labeledposIPt2Set;
      npts = numel(ipt2View);
      
      ptclrs = obj.ptClrs;
      
      hPred = gobjects(npts,ntgtsplot);
      hTxt = gobjects(npts,ntgtsplot);
      pfix = obj.handleTagPfix;
      for iTgt = 1:ntgtsplot
        iTgtAbs = itgtoffset+iTgt;
        for iPt = 1:npts
          clr = ptclrs(iPt,:);
          iVw = ipt2View(iPt);
          ptset = ipt2set(iPt);
          hPred(iPt,iTgt) = plot(ax(iVw),nan,nan,markerPVscell{:},...
            'Color',clr,...
            'Tag',sprintf('%s_XYPrdRed_%d_%d',pfix,iPt,iTgtAbs));
          hTxt(iPt,iTgt) = text(nan,nan,num2str(ptset),...
            'Parent',ax(iVw),...
            'Color',clr,textPVscell{:},...
            'Tag',sprintf('%s_PrdRedTxt_%d_%d',pfix,iPt,iTgtAbs));
        end
      end
    end
    function [hPc,hPchT] = hlpPlotPches(obj,ntgtsplot,itgtoffset,pchTextPVs)
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);

      assert(isscalar(ax),'Unsupported for multiview.');

      clr = obj.pchColor;
      alp = obj.pchFaceAlpha;
      pfix = obj.handleTagPfix;

      hPc = gobjects(1,ntgtsplot);
      hPchT = gobjects(1,ntgtsplot);
      for iTgt = 1:ntgtsplot
        iTgtAbs = itgtoffset+iTgt;
        hPc(iTgt) = patch(ax,nan,nan,clr,...
          'FaceAlpha',alp,...
          'PickableParts','none',...
          'Tag',sprintf('%s_Pch_%d',pfix,iTgtAbs));
        hPchT(iTgt) = text(nan,nan,num2str(iTgtAbs),...
          'Parent',ax,...
          'Color',[0 0 0],...
          'fontsize',pchTextPVs.FontSize,...
          'fontweight','bold',...
          'Tag',sprintf('%s_PchTxt_%d',pfix,iTgtAbs),...
          'userdata',iTgtAbs,...
          'ButtonDownFcn',@(s,e)obj.cbkPchTextBDF(s,e));
      end
    end    
    function vizInit(obj,varargin)
      % trk: TrkFile
      %
      % See TrackingVisualizerBase
      % See "Construction/Init notes" below

      [postload,ntgtsinitial,~] = myparse(varargin,...
        'postload',false, ... % see Construction/Init notes
        'ntgts',[], ... % optionally provide known initial number of targets
        'ntgtmax',[] ... % unused, just eliminates warning
        );      
      
      obj.deleteGfxHandles();
      
      pppiFld = obj.ptsPlotInfoFld;
      pppi = obj.lObj.(pppiFld); 
      obj.mrkrReg = pppi.MarkerProps.Marker;
      obj.mrkrOcc = pppi.OccludedMarker;
      
      npts = numel(obj.ipt2vw);
      if isempty(ntgtsinitial)
        ntgtsinitial = obj.lObj.nTargets;
      end
      if postload
        ptclrspppi = obj.ptClrs;
      else
        ptclrs = obj.lObj.Set2PointColors(pppi.Colors);
        obj.ptClrs = ptclrs; % .ptClrs field now prob unnec
        obj.txtOffPx = pppi.TextOffset; % .txtOffPx now prob unnec
      end
      szassert(ptclrs,[npts 3]);      

      % init .xyVizPlotArgs*
      [markerPVs,textPVs,pchTextPVs,skelPVs] = ...
          TrackingVisualizerMT.convertLabelerCosmeticPVs(pppi);
      markerPVscell = struct2paramscell(markerPVs);
      textPVscell = struct2paramscell(textPVs);
      skelPVscell = struct2paramscell(skelPVs);
      
      if postload
        % We init first with markerPVs/textPVs, then set saved custom PVs
        hXYPrdRed0 = obj.hXYPrdRed;
        hXYPrdRedTxt0 = obj.hXYPrdRedTxt;
      end
      
      axs = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),axs);

      [obj.hXYPrdRed,obj.hXYPrdRedTxt] = ...
            obj.hlpPlotTgts(ntgtsinitial,0,markerPVscell,textPVscell);      
      
      nvw = obj.lObj.nview;
      obj.hSkel = gobjects(1,nvw);      
      for ivw=1:nvw
        ax = axs(ivw);
        % cf LabelCore.initSkeletonEdge
        obj.hSkel(ivw) = plot(ax,nan,nan,'-',...
          'PickableParts','none',...
          'Tag',sprintf('TrackingVisualizerMT_Skel'),...
          skelPVscell{:});
      end
      
      if obj.doPch
        [obj.hPch,obj.hPchTxt] = obj.hlpPlotPches(ntgtsinitial,0,pchTextPVs);
      end
      
      if postload
        if isstruct(hXYPrdRed0)
          if numel(hXYPrdRed0)==numel(obj.hXYPrdRed)
            arrayfun(@(x,y)set(x,y),obj.hXYPrdRed,hXYPrdRed0);
          else
            warningNoTrace('.hXYPrdRed: Number of saved prop-val structs does not match number of line handles.');
          end
        end
        if isstruct(hXYPrdRedTxt0)
          if numel(hXYPrdRedTxt0)==numel(obj.hXYPrdRedTxt)
            arrayfun(@(x,y)set(x,y),obj.hXYPrdRedTxt,hXYPrdRedTxt0);
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
    
    function initAndUpdateSkeletonEdges(obj,sedges)
      % Inits skel edges and sets their posns based on current hXYPrdRed.
      %obj.skelEdges = sedges;
      obj.updateSkel();
    end
    function updateSkel(obj,xy)
      % xy (opt): if provided, must be [npts 2 ntgts].
      
      if obj.tfHideViz || ~obj.tfShowSkel || isempty(obj.hSkel)
        return;
      end
      
      se = obj.lObj.skeletonEdges;
      if isempty(se)
        return;
      end
      
      npts = obj.nPts;
      ntgts = obj.nTgts;
      
      % compile itgtshow, those tgts which have a visible skeleton
      if obj.showOnlyPrimary
        itgtshow = obj.iTgtPrimary;
        if isequal(itgtshow,obj.iTgtHide)
          itgtshow = zeros(1,0);
        end
      else
        itgtshow = 1:ntgts;
        itgtshow(:,obj.iTgtHide) = [];
      end      
      
      if nargin<2
        % get xy from current .hXYPrdRed .XData, .YData
        
        h = obj.hXYPrdRed;
        ntgtshow = numel(itgtshow);
        xy = nan(npts,2,ntgtshow);
        c = 1;
        for itgt=itgtshow
          if isempty(itgt) continue; end
          x = get(h(:,itgt),{'XData'});
          y = get(h(:,itgt),{'YData'});
          xytgt = [cell2mat(x) cell2mat(y)];
          % xytgt should be nan for both estocc and fullocc.
          xy(:,:,c) = xytgt;
          c = c+1;
          
          %tfOccld = any(isnan(xytgt),2);
          %LabelCore.setSkelCoords(xytgt,tfOccld,hSkl(:,iTgt),sedges);
        end
        
      else
        szassert(xy,[npts 2 ntgts]);
        xy = xy(:,:,itgtshow);
      end
      
      TrackingVisualizerMTFast.updateSkelStc(obj.hSkel,se,npts,xy,obj.skel_linestyle);
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
    function setAllShowHide(obj,tfHide,tfHideTxt,tfShowCurrTgtOnly,tfShowSkel)
      obj.tfHideViz = tfHide;
      obj.tfHideTxt = tfHideTxt;
      obj.tfShowSkel = tfShowSkel;
      obj.showOnlyPrimary = tfShowCurrTgtOnly;
      obj.updateShowHideAll();      
    end
    function updateShowHideAll(obj)
      % update .Visible for 
      % * .hXYPrd* [npts x ntgt]
      % * .hSkel [nedge x ntgt]
      % * .hPch [ntgt]
      
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
        set(obj.hSkel,'Visible',onoffSkel);
        % because updateSkel() early returns if visible is off
        obj.updateSkel(); 
      end      
      if obj.doPch
        onoffPch = onIff(obj.tfShowPch);
        [obj.hPch(tfTgtOn).Visible] = deal(onoffPch);
        [obj.hPchTxt(tfTgtOn).Visible] = deal(onoffPch);
        [obj.hPch(~tfTgtOn).Visible] = deal('off');
        [obj.hPchTxt(~tfTgtOn).Visible] = deal('off');
      end
    end
    function set_hittest(obj,onoff)
      if ~isempty(obj.hXYPrdRed) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hXYPrdRed.HitTest] = deal(onoff);
        [obj.hXYPrdRedTxt.HitTest] = deal(onoff);
      end
      
      % skel, pch: not affected by hide
      if ~isempty(obj.hSkel)
        set(obj.hSkel,'HitTest',onoff);
        % because updateSkel() early returns if visible is off
        obj.updateSkel(); 
      end      
      if obj.doPch
        [obj.hPch.HitTest] = deal(onoff);
        [obj.hPchTxt.HitTest] = deal(onoff);
      end            
    end
    function hittest_off_all(obj)
      obj.set_hittest('off');
    end
    function hittest_on_all(obj)
      obj.set_hittest('on');
    end
    
    function updateTrackResI(obj,xy,tfeo,iTgt)
      % xy: [npts x 2]
      % tfeo: [npts] logical for est-occ; can be [] to skip
      % iTgt: target index to update
      
      obj.ensureNTgts(iTgt);
      
      npts = obj.nPts;
      %skelEdges = obj.lObj.skeletonEdges;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      %hSkl = obj.hSkel;
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
      
      %tfOccld = any(isinf(xy),2);      
      %LabelCore.setSkelCoords(xy,tfOccld,hSkl(:,iTgt),skelEdges);
      obj.updateSkel();
      
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
      % Targets > ntgtsgiven are set to nan locs.
      
      if nargin<3
        tfeo = [];
      end
      
      ntgtsgiven = size(xy,3);
      obj.ensureNTgts(ntgtsgiven);
      npts = obj.nPts;
      ntgts = obj.nTgts;      
      %assert(ntgtsgiven<=ntgts);
      assert(isempty(tfeo)||ntgtsgiven==size(tfeo,2));
      
      %skelEdges = obj.lObj.skeletonEdges;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      %hSkl = obj.hSkel;
      hP = obj.hPch;
      hPT = obj.hPchTxt;
      dx = obj.txtOffPx;
      xyoff = xy+dx;
      for iTgt=1:ntgts
        if iTgt>ntgtsgiven
          set(h(:,iTgt),'XData',nan,'YData',nan);
          set(hTxt(:,iTgt),'Position',[nan nan 0]);
          %LabelCore.setSkelCoords(nan(npts,2),false(npts,1),hSkl(:,iTgt),skelEdges);
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

          %tfOccld = any(isinf(xytgt),2);
          %LabelCore.setSkelCoords(xytgt,tfOccld,hSkl(:,iTgt),skelEdges);

          if obj.doPch
            roi = obj.lObj.maGetLossMask(xytgt);
            set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2)); 
            set(hPT(iTgt),'Position',[roi(1,:) 0]);
          end
        end
      end
      
      if ntgts>ntgtsgiven
        xy = cat(3,xy,nan(npts,2,ntgts-ntgtsgiven));
      end
      obj.updateSkel(xy);      
    end
    function newFrame(obj,frm)
      [tfhaspred,xy,tfocc] = obj.trk.getPTrkFrame(frm,'collapse',true);
      obj.updateTrackRes(xy,tfocc);
    end
    function updatePrimary(obj,iTgtPrimary)
      iTgtPrimary0 = obj.iTgtPrimary;
      iTgtChanged = ~isequal(iTgtPrimary,iTgtPrimary0);
      
      if iTgtChanged
        obj.ensureNTgts(iTgtPrimary);
        obj.iTgtPrimary = iTgtPrimary;

      % TODO 20220209: skel color primary?
%         trajClrCurr = obj.lObj.projPrefs.Trx.TrajColorCurrent;
%         hSkl = obj.hSkel;
%         if ~isempty(hSkl)
%             if iTgtPrimary0>0
%               set(hSkl(:,iTgtPrimary0),'Color',obj.skelEdgeColor);
%             end
%             if iTgtPrimary>0
%               set(hSkl(:,iTgtPrimary),'Color',trajClrCurr);
%             end
%         end
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
      
      obj.ensureNTgts(iTgtHide);

      if obj.showOnlyPrimary
        tfTgtHide0on = iTgtHide0==obj.iTgtPrimary;
      else
        tfTgtHide0on = true;
      end
      if tfTgtHide0on
        onoffVizH0 = onIff(~obj.tfHideViz);
        onoffTxtH0 = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
        %onoffSkelH0 = onIff(~isempty(obj.hSkel) && ~obj.tfHideViz && obj.tfShowSkel);
      else
        onoffVizH0 = 'off';
        onoffTxtH0 = 'off';
        %onoffSkelH0 = 'off';
      end
      
      [obj.hXYPrdRed(:,iTgtHide0).Visible] = deal(onoffVizH0);
      [obj.hXYPrdRedTxt(:,iTgtHide0).Visible] = deal(onoffTxtH0);
      [obj.hXYPrdRed(:,iTgtHide).Visible] = deal('off');
      [obj.hXYPrdRedTxt(:,iTgtHide).Visible] = deal('off');
      
      obj.iTgtHide = iTgtHide;
      
      if ~isempty(obj.hSkel)
        % Needs to occur after .iTgtHide is set
        obj.updateSkel();
      end
    end
    function updatePches(obj)
      if obj.doPch
        ntgts = obj.nTgts;
        hP = obj.hPch;
        hPT = obj.hPchTxt;
        hXY = obj.hXYPrdRed;        
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
    function skeletonCosmeticsUpdated(obj)
      ppiFld = obj.ptsPlotInfoFld;
      ppi = obj.lObj.(ppiFld);
      set(obj.hSkel,ppi.SkeletonProps);
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
    
    function obj = TrackingVisualizerMT(lObj,ptsPlotInfoField,handleTagPfix,varargin)
      obj.tfHideTxt = false;
      obj.tfHideViz = false;         

      if nargin==0
        return;
      end
 
      [skel_linestyle] = myparse(varargin,'skel_linestyle','-');
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
      obj.ipt2vw = lObj.labeledposIPt2View;    
      
      obj.ptsPlotInfoFld = ptsPlotInfoField;
      obj.handleTagPfix = handleTagPfix;
      obj.skel_linestyle = skel_linestyle;
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
    function [markerPVs,textPVs,pchTextPVs,skelPVs] = ...
                                  convertLabelerCosmeticPVs(pppi)
      % convert .ptsPlotInfo from labeler to that used by this obj
      %
      % The point being that for eg MA labeling, we want smaller markers
      % etc on other targets.

      markerPVs = TrackingVisualizerMT.convertLabelerMarkerPVs(pppi.MarkerProps);
      textPVs = TrackingVisualizerMT.convertLabelerTextPVs(pppi.TextProps);
      pchTextPVs = struct('FontSize',round(textPVs.FontSize*2.0));
      skelPVs = pppi.SkeletonProps;
    end
    function markerPVs = convertLabelerMarkerPVs(markerPVs)
      sizefac = TrackingVisualizerMT.MRKR_SIZE_FAC;
      markerPVs.MarkerSize = round(markerPVs.MarkerSize*sizefac);
      markerPVs.PickableParts = 'none';
    end      
    function textPVs = convertLabelerTextPVs(textPVs)
      sizefac = TrackingVisualizerMT.MRKR_SIZE_FAC;
      textPVs.FontSize = round(textPVs.FontSize*sizefac);
      textPVs.PickableParts = 'none'; 
    end
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