classdef TrackingVisualizer < handle
  
  % TrackingVisualizers know how to plot/show tracking results on an axes
  % (not owned by itself). They know how to show things and they own the 
  % relevant lines/graphics handles but that's it. Theoretically you can 
  % create/delete them at will to add/rm tracking overlays on top of your
  % images/movies.

  % LabelTracker Property forwarding notes 20181211 LabelTracker contains 
  % stateful props (.hideViz, .hideVizTxt, .showVizReplicates) that are 
  % conceptually forwarding props to TrackingVisualizer. They are there 
  % because i) LT currently presents a single public interface to clients 
  % for tracking (including SetObservability); and ii) LT handles 
  % serialization/reload of these props. This seems fine for now.
  %
  % The pattern followed here for these props is
  % - stateful, SetObservable prop in LT
  % - stateful prop in TrkVizer. Note a lObj has multiple TVs.
  % - set methods set stateful prop and forward to trkVizer, which performs
  % action and sets their stateful props
  % - no getters, just get the prop
  % - get/loadSaveToken set the stateful prop and forward to trkVizer

  properties 
    lObj % Included only to access the current raw image. Ideally used as little as possible

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3], like Labeler/labeledposIPt2View.
    
    txtOffPx % scalar, px offset for landmark text labels 

    tfHideViz % scalar, true if tracking res hidden
    tfHideTxt % scalar, if true then hide text even if tfHideViz is false
    
    handleTagPfix % char, prefix for handle tags

    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
    hXYPrdRedTxt; % [nPts] handle vec, text labels for hXYPrdRed
  end
  properties (Constant)
    SAVEPROPS = {'ipt2vw' 'ptClrs' 'txtOffPx' 'tfHideViz' 'tfHideTxt' ...
      'handleTagPfix'};
    LINE_PROPS_COSMETIC_SAVE = {'Color' 'LineWidth' 'Marker' ...
      'MarkerEdgeColor' 'MarkerFaceColor' 'MarkerSize'};
  end
  properties (Dependent)
    nPts
  end
  methods
    function v = get.nPts(obj)
      v = numel(obj.ipt2vw);
    end    
  end
  
  methods
    function deleteGfxHandles(obj)
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      deleteValidHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
      deleteValidHandles(obj.hXYPrdRedTxt);
      obj.hXYPrdRedTxt = [];
    end
    function vizInit(obj,varargin)
      % Sets .hXYPrdRed, .hXYPrdRedOther

      postload = myparse(varargin,...
        'postload',false... % set to true for post-load init
        );
      
      obj.deleteGfxHandles();
      
      npts = numel(obj.ipt2vw);
      if postload
        ptclrs = obj.ptClrs;
      else
        ptclrs = obj.lObj.PredictPointColors;
        obj.ptClrs = ptclrs;
        obj.txtOffPx = obj.lObj.labelPointsPlotInfo.LblOffset;
      end
      szassert(ptclrs,[npts 3]);      

      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.projPrefs.Track;
      ptsPlotInfo = obj.lObj.labelPointsPlotInfo;
      plotPrefs = trackPrefs.PredictPointsPlot;
      plotPrefs.PickableParts = 'none';
      xyVizPlotArgs = struct2paramscell(plotPrefs);
      xyVizPlotArgsNonTarget = xyVizPlotArgs; % TODO: customize
      
      hXYPrdRed0 = obj.hXYPrdRed;
      
      npts = obj.nPts;
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      ipt2set = obj.lObj.labeledposIPt2Set;
      hTmp = gobjects(npts,1);
      hTmpOther = gobjects(npts,1);
      hTxt = gobjects(npts,1);
      pfix = obj.handleTagPfix;
      for iPt = 1:npts
        clr = ptclrs(iPt,:);
        iVw = ipt2View(iPt);
        set = ipt2set(iPt);
        hTmp(iPt) = plot(ax(iVw),nan,nan,xyVizPlotArgs{:},...
          'Color',clr,...
          'Tag',sprintf('%s_XYPrdRed_%d',pfix,iPt));
        hTmpOther(iPt) = plot(ax(iVw),nan,nan,xyVizPlotArgs{:},...
          'Color',clr,...
          'Tag',sprintf('%s_XYPrdRedOther_%d',pfix,iPt));
        hTxt(iPt) = text(nan,nan,num2str(set),'Parent',ax(iVw),...
          'Color',clr,...
          'FontSize',ptsPlotInfo.FontSize,...
          'PickableParts','none',...
          'Tag',sprintf('%s_PrdRedTxt_%d',pfix,iPt));
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedOther = hTmpOther;
      obj.hXYPrdRedTxt = hTxt;
      
      if postload && isstruct(hXYPrdRed0) 
        if numel(hXYPrdRed0)==numel(hTmp)
          arrayfun(@(x,y)set(x,y),hTmp,hXYPrdRed0);
        else
          warningNoTrace('.hXYPrdRed: Number of saved prop-val structs does not match number of line handles.');
        end
      end
      
      obj.vizInitHook();
    end
    function vizInitHook(obj)
      % overload me
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
      [obj.hXYPrdRedOther.Visible] = deal(onoffViz);
      onoffTxt = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
      [obj.hXYPrdRedTxt.Visible] = deal(onoffTxt);
    end
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hXYPrdRed(iPt),'Color',clr);
        set(obj.hXYPrdRedOther(iPt),'Color',clr);
        set(obj.hXYPrdRedTxt(iPt),'Color',clr);
      end
      obj.ptClrs = ptsClrs;
    end
    function updateTrackRes(obj,xy)
      %
      % xy: [npts x 2]
            
      npts = obj.nPts;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      dx = obj.txtOffPx;
      xyoff = xy+dx;
      for iPt=1:npts
        set(h(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2));
        set(hTxt(iPt),'Position',[xyoff(iPt,:) 0]);
      end
    end
    function setMarkerCosmetics(obj,pvargs)
      arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRed);
    end
    function setTextCosmetics(obj,pvargs)
      arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRedTxt);      
    end
  end
  
  methods 
    % Ways to create/init a TrackingVisualizer
    % - Call the constructor normally, then vizInit();
    % - (When loading) Call constructor with no args, then postLoadInit()
    
    function obj = TrackingVisualizer(lObj,handleTagPfix)
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
    end
  end
  methods (Static)
    function b = loadobj(a)
      if isstruct(a)
        b = TrackingVisualizer();
        for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
          b.(p) = a.(p);
        end
      else
        b = a;
      end
    end
  end
end