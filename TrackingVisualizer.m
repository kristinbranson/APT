classdef TrackingVisualizer < handle
  
  % TrackingVisualizers know how to plot/show tracking results on an axes
  % (not owned by itself). They know how to show things and they own the 
  % relevant lines/graphics handles but that's it.

  % LabelTracker Property forwarding notes 20181211 LabelTracker contains 
  % stateful props (.hideViz and .showVizReplicates) that are conceptually 
  % pure forwarding props to TrackingVisualizer. They are there because 
  % i) LT currently presents a single public interface to clients for 
  % tracking (including SetObservability); and ii) LT handle 
  % serialization/reload of these props. This seems fine for now.
  %
  % The pattern followed here for these props is
  % - stateful, SetObservable prop in LT
  % - set methods forward to trkVizer (and set stateful prop)
  % - no getter, just get the prop
  % - get/loadSaveToken set the stateful prop and forward to trkVizer

  properties
    lObj % Included only to access the current raw image. Ideally used as little as possible

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler
    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3] RGB for pts
    
    handleTagPfix % char, prefix for handle tags
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
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
    end
    function vizInit(obj)
      % Sets .hXYPrdRed, .hXYPrdRedOther

      obj.deleteGfxHandles();
      
      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.projPrefs.Track;
      plotPrefs = trackPrefs.PredictPointsPlot;
      plotPrefs.PickableParts = 'none';
      xyVizPlotArgs = struct2paramscell(plotPrefs);
      xyVizPlotArgsNonTarget = xyVizPlotArgs; % TODO: customize
      
      npts = obj.nPts;
      ptsClrs = obj.ptClrs;
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      hTmp = gobjects(npts,1);
      hTmpOther = gobjects(npts,1);
      pfix = obj.handleTagPfix;
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        iVw = ipt2View(iPt);
        hTmp(iPt) = plot(ax(iVw),nan,nan,xyVizPlotArgs{:},...
          'Color',clr,'Tag',sprintf('%s_XYPrdRed_%d',pfix,iPt));
        hTmpOther(iPt) = plot(ax(iVw),nan,nan,xyVizPlotArgs{:},...
          'Color',clr,'Tag',sprintf('%s_XYPrdRedOther_%d',pfix,iPt));
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedOther = hTmpOther;
      
      obj.vizInitHook();
    end
    function vizInitHook(obj)
      % overload me
    end
    function setHideViz(obj,onoff)
      if islogical(onoff)
        onoff = onIff(~tf);
      end
      [obj.hXYPrdRed.Visible] = deal(onoff);
      [obj.hXYPrdRedOther.Visible] = deal(onoff);
    end
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hXYPrdRed(iPt),'Color',clr);
        set(obj.hXYPrdRedOther(iPt),'Color',clr);
      end
      obj.ptClrs = ptsClrs;
    end
  end
  
  methods 
    function obj = TrackingVisualizer(lObj,handleTagPfix)
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
      obj.ipt2vw = lObj.labeledposIPt2View;
      
      obj.ptClrs = lObj.PredictPointColors;      
      npts = numel(obj.ipt2vw);
      szassert(obj.ptClrs,[npts 3]);
      
      obj.handleTagPfix = handleTagPfix;
    end
    function delete(obj)
      obj.deleteGfxHandles();
    end    
  end
  
end