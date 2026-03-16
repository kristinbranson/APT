classdef TrackingVisualizerMT < TrackingVisualizerBase

  % TrackingVisualizerMT
  % Like TrackingVisualizer, but can handles/display results for many
  % targets at once.
  %
  % Non-gobject model state lives on the associated TrackingVisualizerMTModel
  % (accessed via obj.tvm_).  This class only owns graphics handles and
  % rendering methods.

  properties
    parent_ % LabelerController reference
    tvm_ % TrackingVisualizerMTModel reference, set by creator

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler

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
    hPch  % [1 x ntgt] handle vec
    hPchTxt % [1 x ntgt] text/lbl for pch
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
      v = obj.tvm_.nPts ;
    end
    function v = get.nTgts(obj)
      v = size(obj.hXYPrdRed, 2) ;
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

    function addTgts(obj, ntgtsadd)
      % plots/adds new gfx handles without touching existing
      % Impacted gfx handles: .hXY*, .hPch*

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      pppiFld = tvm.ptsPlotInfoFld ;
      pppi = lObj.(pppiFld) ;
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
    function ensureNTgts(obj, ntgtsreqd)
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
    function [hPred,hTxt] = hlpPlotTgts(obj, ntgtsplot, itgtoffset, ...
        markerPVscell, textPVscell)
      % create/plot gfx handles for ntgtsplot targets
      %
      % itgtoffset: graphics Tags range over itgtoffset+(1:ntgtsplot)

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = tvm.ipt2vw ;
      ipt2set = lObj.labeledposIPt2Set;
      npts = numel(ipt2View);

      ptclrs = tvm.ptClrs ;

      hPred = gobjects(npts,ntgtsplot);
      hTxt = gobjects(npts,ntgtsplot);
      pfix = tvm.handleTagPfix ;
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
    function [hPc,hPchT] = hlpPlotPches(obj, ntgtsplot, itgtoffset, pchTextPVs)
      tvm = obj.tvm_ ;
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);

      assert(isscalar(ax),'Unsupported for multiview.');

      clr = tvm.pchColor ;
      alp = tvm.pchFaceAlpha ;
      pfix = tvm.handleTagPfix ;

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
    function vizInit(obj, varargin)
      % Initialize graphics handles and cosmetics.
      %
      % See TrackingVisualizerBase
      % See "Construction/Init notes" below

      [postload,ntgtsinitial,~] = myparse(varargin,...
        'postload',false, ... % see Construction/Init notes
        'ntgts',[], ... % optionally provide known initial number of targets
        'ntgtmax',[] ... % unused, just eliminates warning
        );

      obj.deleteGfxHandles();

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      pppiFld = tvm.ptsPlotInfoFld ;
      pppi = lObj.(pppiFld) ;
      tvm.mrkrReg = pppi.MarkerProps.Marker ;
      tvm.mrkrOcc = pppi.OccludedMarker ;

      npts = tvm.nPts ;
      if isempty(ntgtsinitial)
        ntgtsinitial = lObj.nTargets;
      end
      if postload
        ptclrs = tvm.ptClrs ;
      else
        ptclrs = lObj.mapSetColorsToPointColors(pppi.Colors);
        tvm.ptClrs = ptclrs ;
        tvm.txtOffPx = pppi.TextOffset ;
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

      nvw = lObj.nview;
      obj.hSkel = gobjects(1,nvw);
      for ivw=1:nvw
        ax = axs(ivw);
        % cf LabelCore.initSkeletonEdge
        obj.hSkel(ivw) = plot(ax,nan,nan,'-',...
          'PickableParts','none',...
          'Tag',sprintf('TrackingVisualizerMT_Skel'),...
          skelPVscell{:});
      end

      if tvm.doPch
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

      tvm.tfShowPch = false ;
      tvm.tfShowSkel = lObj.showSkeleton ;

      % default textPVs do not respect .tfHideViz/.tfHideTxt
      obj.updateShowHideAll();

      tvm.iTgtPrimary = zeros(1,0) ;
      tvm.iTgtHide = zeros(1,0) ;

      obj.vizInitHook();
    end
    function vizInitHook(obj) %#ok<MANU>
      % overload me
    end
    function initAndUpdateSkeletonEdges(obj, sedges) %#ok<INUSD>
      % Inits skel edges and sets their posns based on current hXYPrdRed.
      obj.updateSkel();
    end
    function updateSkel(obj, xy)
      % xy (opt): if provided, must be [npts 2 ntgts].

      tvm = obj.tvm_ ;
      if tvm.tfHideViz || ~tvm.tfShowSkel || isempty(obj.hSkel)
        return;
      end

      lObj = obj.parent_.labeler_ ;
      se = lObj.skeletonEdges;
      if isempty(se)
        return;
      end

      npts = obj.nPts;
      ntgts = obj.nTgts;

      % compile itgtshow, those tgts which have a visible skeleton
      if tvm.showOnlyPrimary
        itgtshow = tvm.iTgtPrimary ;
        if isequal(itgtshow, tvm.iTgtHide)
          itgtshow = zeros(1,0);
        end
      else
        itgtshow = 1:ntgts;
        itgtshow(:, tvm.iTgtHide) = [];
      end

      if nargin<2
        % get xy from current .hXYPrdRed .XData, .YData

        h = obj.hXYPrdRed;
        ntgtshow = numel(itgtshow);
        xy = nan(npts,2,ntgtshow);
        c = 1;
        for itgt=itgtshow
          if isempty(itgt), continue; end
          x = get(h(:,itgt),{'XData'});
          y = get(h(:,itgt),{'YData'});
          xytgt = [cell2mat(x) cell2mat(y)];
          xy(:,:,c) = xytgt;
          c = c+1;
        end

      else
        szassert(xy,[npts 2 ntgts]);
        xy = xy(:,:,itgtshow);
      end

      TrackingVisualizerMTFast.updateSkelStc(obj.hSkel, se, npts, xy, ...
        'linestyle', tvm.skel_linestyle) ;
    end
    function setShowSkeleton(obj, tf)
      obj.tvm_.tfShowSkel = tf ;
      obj.updateShowHideAll();
    end
    function setHideViz(obj, tf)
      obj.tvm_.tfHideViz = tf ;
      obj.updateShowHideAll();
    end
    function setHideTextLbls(obj, tf)
      obj.tvm_.tfHideTxt = tf ;
      obj.updateShowHideAll();
    end
    function setShowPches(obj, tf)
      obj.tvm_.tfShowPch = tf ;
      obj.updateShowHideAll();
    end
    function hideOtherTargets(obj)
      obj.setShowOnlyPrimary(true);
    end
    function setShowOnlyPrimary(obj, tf)
      obj.tvm_.showOnlyPrimary = tf ;
      obj.updateShowHideAll();
    end
    function setAllShowHide(obj, tfHide, tfHideTxt, tfShowCurrTgtOnly, tfShowSkel)
      tvm = obj.tvm_ ;
      tvm.tfHideViz = tfHide ;
      tvm.tfHideTxt = tfHideTxt ;
      tvm.tfShowSkel = tfShowSkel ;
      tvm.showOnlyPrimary = tfShowCurrTgtOnly ;
      obj.updateShowHideAll();
    end
    function updateShowHideAll(obj)
      % update .Visible for
      % * .hXYPrd* [npts x ntgt]
      % * .hSkel [nedge x ntgt]
      % * .hPch [ntgt]

      tvm = obj.tvm_ ;

      % 'overall' on/offness
      onoffViz = onIff(~tvm.tfHideViz);
      onoffTxt = onIff(~tvm.tfHideViz && ~tvm.tfHideTxt);

      if tvm.showOnlyPrimary
        tfTgtOn = false(1,obj.nTgts);
        tfTgtOn(tvm.iTgtPrimary) = true;
      else
        tfTgtOn = true(1,obj.nTgts);
      end
      tfTgtOnHideAffected = tfTgtOn;
      tfTgtOnHideAffected(tvm.iTgtHide) = false;

      if ~isempty(obj.hXYPrdRed) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hXYPrdRed(:,tfTgtOnHideAffected).Visible] = deal(onoffViz);
        [obj.hXYPrdRed(:,~tfTgtOnHideAffected).Visible] = deal('off');
        [obj.hXYPrdRedTxt(:,tfTgtOnHideAffected).Visible] = deal(onoffTxt);
        [obj.hXYPrdRedTxt(:,~tfTgtOnHideAffected).Visible] = deal('off');
      end

      % skel, pch: not affected by hide
      if ~isempty(obj.hSkel)
        onoffSkel = onIff(~isempty(obj.hSkel) && ~tvm.tfHideViz && tvm.tfShowSkel);
        set(obj.hSkel,'Visible',onoffSkel);
        obj.updateSkel();
      end
      if tvm.doPch
        onoffPch = onIff(tvm.tfShowPch);
        [obj.hPch(tfTgtOn).Visible] = deal(onoffPch);
        [obj.hPchTxt(tfTgtOn).Visible] = deal(onoffPch);
        [obj.hPch(~tfTgtOn).Visible] = deal('off');
        [obj.hPchTxt(~tfTgtOn).Visible] = deal('off');
      end
    end
    function set_hittest(obj, onoff)
      if ~isempty(obj.hXYPrdRed) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hXYPrdRed.HitTest] = deal(onoff);
        [obj.hXYPrdRedTxt.HitTest] = deal(onoff);
      end

      if ~isempty(obj.hSkel)
        set(obj.hSkel,'HitTest',onoff);
        obj.updateSkel();
      end
      if obj.tvm_.doPch
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

    function updateTrackResI(obj, xy, tfeo, iTgt)
      % xy: [npts x 2]
      % tfeo: [npts] logical for est-occ; can be [] to skip
      % iTgt: target index to update

      tvm = obj.tvm_ ;
      obj.ensureNTgts(iTgt);

      npts = obj.nPts;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      dx = tvm.txtOffPx ;
      xyoff = xy+dx;

      for iPt=1:npts
        set(h(iPt,iTgt),'XData',xy(iPt,1),'YData',xy(iPt,2));
        set(hTxt(iPt,iTgt),'Position',[xyoff(iPt,:) 0]);
      end
      if ~isempty(tfeo)
        tfeo = logical(tfeo);
        set(h(tfeo,iTgt),'Marker',tvm.mrkrOcc);
        set(h(~tfeo,iTgt),'Marker',tvm.mrkrReg);
      end

      obj.updateSkel();

      if tvm.doPch
        lObj = obj.parent_.labeler_ ;
        hP = obj.hPch;
        hPT = obj.hPchTxt;
        roi = lObj.maGetLossMask(xy);
        set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2));
        set(hPT(iTgt),'Position',[roi(1,:) 0]);
      end
    end
    function updateTrackRes(obj, xy, tfeo)
      % xy: [npts x 2 x ntgtsgiven]
      % tfeo: [npts x ntgtsgiven] logical for est-occ
      %
      % Targets > ntgtsgiven are set to nan locs.

      if nargin<3
        tfeo = [];
      end

      tvm = obj.tvm_ ;
      ntgtsgiven = size(xy,3);
      obj.ensureNTgts(ntgtsgiven);
      npts = obj.nPts;
      ntgts = obj.nTgts;
      assert(isempty(tfeo)||ntgtsgiven==size(tfeo,2));

      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      hP = obj.hPch;
      hPT = obj.hPchTxt;
      dx = tvm.txtOffPx ;
      xyoff = xy+dx;

      lObj = obj.parent_.labeler_ ;
      for iTgt=1:ntgtsgiven,
        xytgt = xy(:,:,iTgt);

        for iPt=1:npts
          set(h(iPt,iTgt),'XData',xytgt(iPt,1),'YData',xytgt(iPt,2));
          set(hTxt(iPt,iTgt),'Position',[xyoff(iPt,:,iTgt) 0]);
        end
        if ~isempty(tfeo)
          set(h(tfeo(:,iTgt),iTgt),'Marker',tvm.mrkrOcc);
          set(h(~tfeo(:,iTgt),iTgt),'Marker',tvm.mrkrReg);
        end

        if tvm.doPch
          roi = lObj.maGetLossMask(xytgt);
          set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2));
          set(hPT(iTgt),'Position',[roi(1,:) 0]);
        end
      end


      if ntgts > ntgtsgiven,
        needupdate = false(npts,ntgts);
        needupdate(:,ntgtsgiven+1:end) = reshape(cellfun(@(x) any(~isnan(x(:))), {h(:,ntgtsgiven+1:end).XData}),npts,[]);
        if any(needupdate(:)),
          set(h(needupdate),'XData',nan,'YData',nan);
          set(hTxt(needupdate),'Position',[nan nan 0]);
        end
        if tvm.doPch
          needupdate = false(ntgts,1);
          needupdate(ntgtsgiven+1:end) = cellfun(@(x) any(~isnan(x(:))), {hP(ntgtsgiven+1:end).XData});
          if any(needupdate(:)),
            set(hP(needupdate),'XData',nan,'YData',nan);
            set(hPT(needupdate),'Position',[nan nan 0]);
          end
        end
      end


      if ntgts>ntgtsgiven
        xy = cat(3,xy,nan(npts,2,ntgts-ntgtsgiven));
      end
      obj.updateSkel(xy);
    end
    function newFrame(obj, frm)
      % Display tracking results for given/new frame.
      [~, xy, tfocc] = obj.tvm_.newFrame(frm) ;
      obj.updateTrackRes(xy, tfocc) ;
    end
    function updatePrimary(obj, iTgtPrimary)
      tvm = obj.tvm_ ;
      iTgtPrimary0 = tvm.iTgtPrimary ;
      iTgtChanged = ~isequal(iTgtPrimary, iTgtPrimary0) ;

      if iTgtChanged
        obj.ensureNTgts(iTgtPrimary);
        tvm.iTgtPrimary = iTgtPrimary ;

        if tvm.showOnlyPrimary
          obj.updateShowHideAll();
        end
      end
    end
    function updateHideTarget(obj, iTgtHide)
      % unhide/show iTgtHide0, hide iTgtHide

      tvm = obj.tvm_ ;
      iTgtHide0 = tvm.iTgtHide ;
      tfnochange = isequal(iTgtHide0, iTgtHide) ;
      if tfnochange
        return;
      end

      obj.ensureNTgts(iTgtHide);

      if tvm.showOnlyPrimary
        tfTgtHide0on = iTgtHide0 == tvm.iTgtPrimary ;
      else
        tfTgtHide0on = true;
      end
      if tfTgtHide0on
        onoffVizH0 = onIff(~tvm.tfHideViz);
        onoffTxtH0 = onIff(~tvm.tfHideViz && ~tvm.tfHideTxt);
      else
        onoffVizH0 = 'off';
        onoffTxtH0 = 'off';
      end

      [obj.hXYPrdRed(:,iTgtHide0).Visible] = deal(onoffVizH0);
      [obj.hXYPrdRedTxt(:,iTgtHide0).Visible] = deal(onoffTxtH0);
      [obj.hXYPrdRed(:,iTgtHide).Visible] = deal('off');
      [obj.hXYPrdRedTxt(:,iTgtHide).Visible] = deal('off');

      tvm.iTgtHide = iTgtHide ;

      if ~isempty(obj.hSkel)
        obj.updateSkel();
      end
    end
    function updatePches(obj)
      tvm = obj.tvm_ ;
      if tvm.doPch
        lObj = obj.parent_.labeler_ ;
        ntgts = obj.nTgts;
        hP = obj.hPch;
        hPT = obj.hPchTxt;
        hXY = obj.hXYPrdRed;
        for iTgt=1:ntgts
          xy = cell2mat(get(hXY(:,iTgt),{'XData' 'YData'}));
          roi = lObj.maGetLossMask(xy);
          set(hP(iTgt),'XData',roi(:,1),'YData',roi(:,2));
          set(hPT(iTgt),'Position',[roi(1,:) 0]);
        end
      end
    end
    function updateLandmarkColors(obj, ptsClrs)
      npts = obj.nPts;
      szassert(ptsClrs,[npts 3]);
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hXYPrdRed(iPt,:),'Color',clr);
        set(obj.hXYPrdRedTxt(iPt,:),'Color',clr);
      end
      obj.tvm_.ptClrs = ptsClrs ;
    end
    function setMarkerCosmetics(obj, pvargs)
      if isstruct(pvargs)
        pvargs = obj.convertLabelerMarkerPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRed);
        obj.tvm_.mrkrReg = pvargs.Marker ;
      else
        assert(false);
      end
    end
    function setTextCosmetics(obj, pvargs)
      if isstruct(pvargs)
        pvargs = obj.convertLabelerTextPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRedTxt);
      else
        assert(false);
      end
    end
    function setTextOffset(obj, offsetPx)
      obj.tvm_.txtOffPx = offsetPx ;

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
    function updateSkeletonCosmetics(obj)
      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      ppiFld = tvm.ptsPlotInfoFld ;
      ppi = lObj.(ppiFld);
      set(obj.hSkel,ppi.SkeletonProps);
    end
    function cbkPchTextBDF(obj, s, e) %#ok<INUSD>
      iTgt = s.UserData;
      obj.parent_.labeler_.setTarget(iTgt);
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

    function obj = TrackingVisualizerMT(parent, tvm)
      % Construct a TrackingVisualizerMT.
      %
      % parent: LabelerController
      % tvm: TrackingVisualizerMTModel

      if nargin == 0
        return;
      end

      obj.parent_ = parent ;
      obj.tvm_ = tvm ;
      gd = parent;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
    end
    function delete(obj)
      obj.deleteGfxHandles();
    end
    function s = saveobj(obj)
      s = struct();
      tvm = obj.tvm_ ;
      for p = TrackingVisualizerMT.SAVEPROPS, p = p{1}; %#ok<FXSET>
        s.(p) = tvm.(p);
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
        b = TrackingVisualizerMT();
        % Legacy load: stash SAVEPROPS for postLoadInit
        for p = TrackingVisualizerMT.SAVEPROPS, p = p{1}; %#ok<FXSET>
          if isfield(a, p)
            % Can't set on tvm_ since it doesn't exist yet; stash in UserData-style
            % This path is for legacy serialized TVs and may need updating.
          end
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
