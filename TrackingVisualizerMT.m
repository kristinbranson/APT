classdef TrackingVisualizerMT < handle
  
  % TrackingVisualizerMT
  % Like TrackingVisualizer, but can handles/display results for many 
  % targets at once

  properties 
    lObj % Included only to access the current raw image. Ideally used as little as possible

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3], like Labeler/labeledposIPt2View.
    
    txtOffPx % scalar, px offset for landmark text labels 

    tfHideViz % scalar, true if tracking res hidden
    tfHideTxt % scalar, if true then hide text even if tfHideViz is false
    %tfHideSkel currently interrogate lObj.showSkeleton
    
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
    iTgtPrimary % [1 x nprimary] tgt indices for 'primary' targets. 
                % Primariness might typically be eg 'current' but it 
                % doesn't have to correspond.
                
    skelEdgeColor = [.7,.7,.7];
  end
  properties (Constant)
    SAVEPROPS = {'ipt2vw' 'ptClrs' 'txtOffPx' 'tfHideViz' 'tfHideTxt' ...
      'handleTagPfix'};
    LINE_PROPS_COSMETIC_SAVE = {'Color' 'LineWidth' 'Marker' ...
      'MarkerEdgeColor' 'MarkerFaceColor' 'MarkerSize'};
    TEXT_PROPS_COSMETIC_SAVE = {'FontSize' 'FontName' 'FontWeight' 'FontAngle'};
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
    end
    function vizInit(obj,varargin)
      % Inits .hXYPrdRed, .hXYPrdRedTxt, .hSkel, .iTgtPrimary
      % 
      % See "Construction/Init notes" below      

      postload = myparse(varargin,...
        'postload',false... % see Construction/Init notes
        );      
      
      obj.deleteGfxHandles();
      
      pppi = obj.lObj.predPointsPlotInfo;

      npts = numel(obj.ipt2vw);
      ntgts = obj.lObj.nTargets;
      if postload
        ptclrs = obj.ptClrs;
      else
        ptclrs = obj.lObj.PredictPointColors;
        obj.ptClrs = ptclrs;
        obj.txtOffPx = pppi.TextOffset;
      end
      szassert(ptclrs,[npts 3]);      

      % init .xyVizPlotArgs*
      markerPVs = pppi.MarkerProps;
      textPVs = pppi.TextProps;
      markerPVs.PickableParts = 'none';
      textPVs.PickableParts = 'none';
      markerPVs = struct2paramscell(markerPVs);
      textPVs = struct2paramscell(textPVs);
      %markerPVsNonTarget = markerPVs; % TODO: customize
      
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
        hTmp(iPt,iTgt) = plot(ax(iVw),nan,nan,markerPVs{:},...
          'Color',clr,...
          'Tag',sprintf('%s_XYPrdRed_%d_%d',pfix,iPt,iTgt));
        hTxt(iPt,iTgt) = text(nan,nan,num2str(ptset),'Parent',ax(iVw),...
          'Color',clr,textPVs{:},...
          'Tag',sprintf('%s_PrdRedTxt_%d_%d',pfix,iPt,iTgt));
      end
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedTxt = hTxt;
      
      obj.initSkeletonEdges(obj.lObj.skeletonEdges);
      
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
      
      % default textPVs do not respect .tfHideViz/.tfHideTxt
      obj.updateHideVizHideText(); 
      
      obj.iTgtPrimary = zeros(1,0);
      
      obj.vizInitHook();
    end
    function vizInitHook(obj)
      % overload me
    end
    function initSkeletonEdges(obj,sedges)
      % Creates/inits .hSkel graphics handles appropriately for edge-set
      % sedges
      
      nEdge = size(sedges,1);
      ntgts = obj.lObj.nTargets;
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
      ntgts = obj.lObj.nTargets;
      for iTgt=1:ntgts
        x = get(h(:,iTgt),{'XData'});
        y = get(h(:,iTgt),{'YData'});
        xytgt = [cell2mat(x) cell2mat(y)];
        tfOccld = any(isnan(xytgt),2);
        LabelCore.setSkelCoords(xytgt,tfOccld,hSkl(:,iTgt),sedges);
      end
    end
    function setHideViz(obj,tf)
      obj.tfHideViz = tf;
      obj.updateHideVizHideText();
    end
    function setHideTextLbls(obj,tf)
      obj.tfHideTxt = tf;
      obj.updateHideVizHideText();
    end
    function updateHideVizHideText(obj)
      onoffViz = onIff(~obj.tfHideViz);
      [obj.hXYPrdRed.Visible] = deal(onoffViz);
      onoffTxt = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
      [obj.hXYPrdRedTxt.Visible] = deal(onoffTxt);
      if ~isempty(obj.hSkel)
        onoffSkel = onIff(~obj.tfHideViz && obj.lObj.showSkeleton);
        [obj.hSkel.Visible] = deal(onoffSkel);
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
    function updateTrackRes(obj,xy,iTgtPrimary)
      %
      % xy: [npts x 2 x ntgts]
      % iTgtPrimary: [nprimary] target indices for 'primary', could be
      %   empty. unused atm
            
      npts = obj.nPts;
      ntgts = obj.nTgts;
      skelEdges = obj.lObj.skeletonEdges;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      hSkl = obj.hSkel;
      dx = obj.txtOffPx;
      xyoff = xy+dx;
      for iTgt=1:ntgts
        for iPt=1:npts
          set(h(iPt,iTgt),'XData',xy(iPt,1,iTgt),'YData',xy(iPt,2,iTgt));
          set(hTxt(iPt,iTgt),'Position',[xyoff(iPt,:,iTgt) 0]);
        end
        
        xytgt = xy(:,:,iTgt);
        tfOccld = any(isinf(xytgt),2);
        LabelCore.setSkelCoords(xytgt,tfOccld,hSkl(:,iTgt),skelEdges);        
      end
      
      trajClrCurr = obj.lObj.projPrefs.Trx.TrajColorCurrent;
      set(hSkl(:,obj.iTgtPrimary),'Color',obj.skelEdgeColor);
      set(hSkl(:,iTgtPrimary),'Color',trajClrCurr);
      
      obj.iTgtPrimary = iTgtPrimary;
    end
    function setMarkerCosmetics(obj,pvargs)
      if isstruct(pvargs)
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRed);
      else
        arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRed);
      end
    end
    function setTextCosmetics(obj,pvargs)
      if isstruct(pvargs)
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRedTxt);
      else        
        arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRedTxt);
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
      
      obj.updateTrackRes(xy,obj.iTgtPrimary);
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
    % Save/load strategy. 
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