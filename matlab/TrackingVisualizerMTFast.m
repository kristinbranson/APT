classdef TrackingVisualizerMTFast < TrackingVisualizerBase

  % TrackingVisualizerMTFast
  % Like TrackingVisualizerMT, but uses fewer graphics handles for speed.
  %
  % Non-gobject model state lives on the associated
  % TrackingVisualizerMTFastModel (accessed via obj.tvm_).

  properties
    parent_ % LabelerController reference
    tvm_ % TrackingVisualizerMTFastModel reference, set by creator

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler

    %%% GFX handles %%%

    hPred; % [npts] plot handles for tracking results, current
               % frame. hPred(ipt) shows landmark ipt across all
               % targets, primary or otherwise
    hPredTxt; % [nPts] handle vec, text labels for hPred
              % currently showing only for primary

    hSkel   % [1xnview] skeleton line handle (all edges/tgts)
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
      v = obj.tvm_.nTgts ;
    end
  end

  methods
    function obj = TrackingVisualizerMTFast(parent, tvm)
      % Construct a TrackingVisualizerMTFast.
      %
      % parent: LabelerController
      % tvm: TrackingVisualizerMTFastModel

      if nargin == 0
        return;
      end

      obj.parent_ = parent ;
      obj.tvm_ = tvm ;
      gd = parent ;
      obj.hAxs = gd.axes_all ;
      obj.hIms = gd.images_all ;
    end

    function deleteGfxHandles(obj)
      if ~isstruct(obj.hPred)
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

    function vizInit(obj, varargin)
      % Initialize graphics handles and cosmetics.

      obj.deleteGfxHandles();

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      pppiFld = tvm.ptsPlotInfoFld ;
      pppi = lObj.(pppiFld);

      tvm.mrkrReg = pppi.MarkerProps.Marker ;
      tvm.mrkrOcc = pppi.OccludedMarker ;
      tvm.txtOffPx = pppi.TextOffset ;
      tvm.skelEdges = lObj.skeletonEdges ;

      npts = tvm.nPts ;
      ptclrs = lObj.mapSetColorsToPointColors(pppi.Colors);
      szassert(ptclrs,[npts 3]);

      [markerPVs,~,~,skelPVs] = TrackingVisualizerMTFast.convertLabelerCosmeticPVs(pppi);
      markerPVscell = struct2paramscell(markerPVs);
      [~,textPVs,~,~] = TrackingVisualizerMTFast.convertLabelerCosmeticPVs(pppi);
      textPVscell = struct2paramscell(textPVs);
      skelPVs = struct2paramscell(skelPVs);

      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = tvm.ipt2vw ;
      ipt2set = lObj.labeledposIPt2Set ;
      hTmp = gobjects(npts,1);
      hTxt = gobjects(npts,1);
      pfix = tvm.handleTagPfix ;
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

      nvw = lObj.nview ;
      obj.hSkel = gobjects(1,nvw);

      for ivw=1:nvw
        axvw = ax(ivw);
        obj.hSkel(ivw) = plot(axvw,nan,nan,'-',...
          'PickableParts','none',...
          'Tag',sprintf('TrackingVisualizerMTFast_Skel'),...
          skelPVs{:});
      end

      assert(~tvm.doPch);
      if ~lObj.maIsMA,
        tvm.iTgtPrimary = lObj.currTarget ;
      else
        tvm.iTgtPrimary = zeros(1,0) ;
      end
    end

    function updateSkel(obj)
      % update obj.hSkel .XData, .YData appropriately

      tvm = obj.tvm_ ;
      if tvm.tfHideViz || ~tvm.tfShowSkel
        return
      end

      assert(isscalar(obj.hSkel),'Multiview support todo.');

      xy = tvm.xyCurr ;
      if tvm.tfShowOnlyPrimary
        tf = tvm.iTgtPrimary == tvm.xyCurrITgts ;
        xy = xy(:,:,tf);
      end

      TrackingVisualizerMTFast.updateSkelStc(...
                          obj.hSkel, tvm.skelEdges, tvm.nPts, xy);
    end
  end
  methods (Static)
    function updateSkelStc(hSkel, skelEdges, npt, xy, varargin)
      % Set hSkel.XData/.YData per xy

      [linestyle,alpha,linewidth] = myparse(varargin,'linestyle','','alpha',0.5, 'linewidth',0.5);

      se = skelEdges;
      k = size(se,1);
      ntgtshow = size(xy,3);
      totlen = k*ntgtshow*3;

      nview = numel(hSkel);
      nptphys = npt/nview;

      ixskip = 0:nptphys*2:nptphys*2*ntgtshow-1;
      isept1 = repmat(se(:,1),1,ntgtshow);
      isept2 = repmat(se(:,2),1,ntgtshow);
      ixpt1 = isept1 + ixskip;
      ixpt2 = isept2 + ixskip;
      iyskip = nptphys:nptphys*2:nptphys*2*ntgtshow-1;
      iypt1 = isept1 + iyskip;
      iypt2 = isept2 + iyskip;

      idatapt1 = 1:3:totlen;
      idatapt2 = 2:3:totlen;

      for iview=1:nview
        iptview = (1:nptphys) + (iview-1)*nptphys;
        xyview = xy(iptview,:,:);
        if all(isnan(xyview(:))),
          xdata = hSkel(iview).XData;
          if all(isnan(xdata(:))),
            continue;
          end
        end
        xdata = nan(1,totlen);
        ydata = nan(1,totlen);
        xdata(idatapt1) = xyview(ixpt1);
        xdata(idatapt2) = xyview(ixpt2);
        ydata(idatapt1) = xyview(iypt1);
        ydata(idatapt2) = xyview(iypt2);
        cc = get(hSkel(iview),'Color');
        if numel(cc) == 3
          cc(end+1) = alpha; %#ok<AGROW>
        else
          cc(end) = alpha;
        end
        set(hSkel(iview),'XData',xdata,'YData',ydata,'LineStyle',linestyle,...
          'linewidth',linewidth,'Color',cc);
      end
    end
  end
  methods
    function initAndUpdateSkeletonEdges(obj, sedges)
      obj.tvm_.skelEdges = sedges ;
      obj.updateSkel();
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
      obj.tvm_.tfShowOnlyPrimary = tf ;
      obj.updateShowHideAll();
    end
    function setAllShowHide(obj, tfHide, tfHideTxt, tfShowCurrTgtOnly, tfShowSkel)
      tvm = obj.tvm_ ;
      tvm.tfHideViz = tfHide ;
      tvm.tfHideTxt = tfHideTxt ;
      tvm.tfShowSkel = tfShowSkel ;
      tvm.tfShowOnlyPrimary = tfShowCurrTgtOnly ;
      obj.updateShowHideAll();
    end
    function updateShowHideAll(obj)
      tvm = obj.tvm_ ;

      if ~isempty(obj.hPred)
        onoffViz = onIff(~tvm.tfHideViz);
        set(obj.hPred,'Visible',onoffViz);
        onoffTxt = onIff(~tvm.tfHideViz && ~tvm.tfHideTxt);
        set(obj.hPredTxt,'Visible',onoffTxt);
        obj.updatePreds();
        obj.updatePredsTxt();
      end

      if ~isempty(obj.hSkel)
        onoffSkel = onIff(~tvm.tfHideViz && tvm.tfShowSkel);
        set(obj.hSkel,'Visible',onoffSkel);
        obj.updateSkel();
      end
    end
    function updatePreds(obj)
      % set obj.hPred .XData, .YData appropriately

      tvm = obj.tvm_ ;
      if tvm.tfHideViz
        return;
      end

      xy = tvm.xyCurr ;
      if tvm.tfShowOnlyPrimary
        tf = tvm.iTgtPrimary == tvm.xyCurrITgts ;
        xy = xy(:,:,tf);
      end

      h = obj.hPred;

      if isempty(xy)
        set(h,'XData',nan,'YData',nan);
      else
        npt = obj.nPts;
        for ipt=1:npt
          xdata = xy(ipt,1,:);
          ydata = xy(ipt,2,:);
          set(h(ipt),'XData',xdata(:),'YData',ydata(:));
        end
      end
    end
    function updatePredsTxt(obj)
      % set obj.hPredTxt positions for primary target

      tvm = obj.tvm_ ;
      if tvm.tfHideViz || tvm.tfHideTxt
        return;
      end

      itgtP = tvm.iTgtPrimary ;
      itgtXY = tvm.xyCurrITgts ;
      if isempty(itgtP) || isempty(itgtXY)
        set(obj.hPredTxt,'Position',[nan nan]);
        return;
      end

      tf = itgtP == itgtXY ;
      if ~any(tf)
        set(obj.hPredTxt,'Position',[nan nan]);
        return;
      end

      xy = tvm.xyCurr ;
      xy = xy(:,:,tf);
      xypos = xy + tvm.txtOffPx ;
      h = obj.hPredTxt;
      npt = obj.nPts;
      for ipt=1:npt
        pos = xypos(ipt,:,1);
        set(h(ipt),'Position',pos(:)');
      end
    end

    function updateTrackRes(obj, xy, tfeo, xyITgts)
      % Update current predictions and render.

      tvm = obj.tvm_ ;
      if nargin < 3
        [npts,~,ntgts] = size(xy);
        tfeo = false(npts,ntgts);
        xyITgts = (1:ntgts)';
      end

      tvm.xyCurr = xy ;
      tvm.occCurr = tfeo ;
      tvm.xyCurrITgts = xyITgts ;
      obj.updatePreds();
      obj.updatePredsTxt();
      obj.updateSkel();
    end
    function newFrame(obj, frm)
      % Display tracking results for given/new frame.
      [~, xy, tfocc] = obj.tvm_.newFrame(frm) ;
      tvm = obj.tvm_ ;
      itgts = tvm.xyCurrITgts ;
      obj.updateTrackRes(xy(:,:,1:numel(itgts)), tfocc(:,1:numel(itgts)), itgts) ;
    end
    function updatePrimary(obj, iTgtPrimary)
      tvm = obj.tvm_ ;
      iTgtPrimary0 = tvm.iTgtPrimary ;
      iTgtChanged = ~isequal(iTgtPrimary, iTgtPrimary0) ;
      tvm.iTgtPrimary = iTgtPrimary ;

      if iTgtChanged
        obj.updateShowHideAll();
      end
    end
    function updateLandmarkColors(obj, ptsClrs)
      npts = obj.nPts;
      szassert(ptsClrs,[npts 3]);
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hPred(iPt),'Color',clr);
        set(obj.hPredTxt(iPt),'Color',clr);
      end
    end
    function setMarkerCosmetics(obj, pvargs)
      if isstruct(pvargs)
        pvargs = TrackingVisualizerMTFast.convertLabelerMarkerPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hPred);
        obj.tvm_.mrkrReg = pvargs.Marker ;
      else
        assert(false);
      end
    end
    function setTextCosmetics(obj, pvargs)
      if isstruct(pvargs)
        pvargs = TrackingVisualizerMTFast.convertLabelerTextPVs(pvargs);
        arrayfun(@(x)set(x,pvargs),obj.hPredTxt);
      else
        assert(false);
      end
    end
    function setTextOffset(obj, offsetPx)
      obj.tvm_.txtOffPx = offsetPx ;
      obj.updatePredsTxt();
    end
    function skeletonCosmeticsUpdated(obj)
      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      ppiFld = tvm.ptsPlotInfoFld ;
      ppi = lObj.(ppiFld);
      set(obj.hSkel,ppi.SkeletonProps);
    end
  end

  methods (Static)
    function [markerPVs,textPVs,pchTextPVs,skelPVs] = ...
                                      convertLabelerCosmeticPVs(pppi)
      markerPVs = TrackingVisualizerMTFast.convertLabelerMarkerPVs(pppi.MarkerProps);
      textPVs = TrackingVisualizerMTFast.convertLabelerTextPVs(pppi.TextProps);
      pchTextPVs = struct('FontSize',round(textPVs.FontSize*2.0));
      skelPVs = pppi.SkeletonProps;
    end
    function markerPVs = convertLabelerMarkerPVs(markerPVs)
      markerPVs.PickableParts = 'none';
    end
    function textPVs = convertLabelerTextPVs(textPVs)
      textPVs.PickableParts = 'none';
    end
  end

end
