classdef TrackingVisualizerReplicates < handle
  % TrackingVisualizers know how to plot/show tracking results on an axes
  % (not owned by itself). They know how to show things and they own the 
  % relevant lines/graphics handles but that's it.

  % CPRLabelTracker Property forwarding notes 20181211
  % LabelTracker/CPRLT contain stateful props (.hideViz and 
  % .showVizReplicates) that are conceptually pure forwarding props to 
  % TrackingVisualizer. They are there because i) LT/CPRLT currently 
  % present a single public interface to clients for tracking (including 
  % SetObservability); and ii) LT/CPRLT handle serialization/reload of 
  % these props. This seems fine for now.
  %
  % The pattern followed here for these props is
  % - stateful, SetObservable prop in LT/CPRLT
  % - set methods forward to trkVizer (and set stateful prop)
  % - no getter, just get the prop
  % - get/loadSaveToken set the stateful prop and forward to trkVizer
  
  properties
    lObj % Included only to access the current raw image. Ideally used as little as possible
    
    hAxs % [nview] axes handles. Owned by Labeler
    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3] RGB for pts
    
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
    hXYPrdFull; % [npts] scatter handles for replicates, current frame, current target
    xyVizPlotArgs; % cell array of args for regular tracking viz    
    xyVizPlotArgsNonTarget; % " for non current target viz
    xyVizPlotArgsInterp; % " for interpolated tracking viz
    xyVizFullPlotArgs; % " for tracking viz w/replicates. These are PV pairs for scatter() not line()
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
    
    function obj = TrackingVisualizerReplicates(lObj)
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.ipt2vw = lObj.labeledposIPt2View;
      
      npts = numel(obj.ipt2vw);
      obj.ptClrs = lines(npts);
      szassert(obj.ptClrs,[npts 3]);
    end
    
    function delete(obj)
      obj.deleteGfxHandles();
    end
    function deleteGfxHandles(obj)
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      deleteValidHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
      deleteValidHandles(obj.hXYPrdFull);
      obj.hXYPrdFull = [];   
    end
    
  end
  
  methods
    
    function vizInit(obj)
      obj.deleteGfxHandles();
      
      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.projPrefs.Track;
      cprPrefs = obj.lObj.projPrefs.CPRLabelTracker.PredictReplicatesPlot;
      plotPrefs = trackPrefs.PredictPointsPlot;
      plotPrefs.PickableParts = 'none';
      obj.xyVizPlotArgs = struct2paramscell(plotPrefs);
      if isfield(trackPrefs,'PredictInterpolatePointsPlot')
        obj.xyVizPlotArgsInterp = struct2paramscell(trackPrefs.PredictInterpolatePointsPlot);
      else
        obj.xyVizPlotArgsInterp = obj.xyVizPlotArgs;
      end
      obj.xyVizPlotArgsNonTarget = obj.xyVizPlotArgs; % TODO: customize
      if isfield(cprPrefs,'MarkerSize') % AL 201706015: Currently always true
        cprPrefs.SizeData = cprPrefs.MarkerSize^2; % Scatter.SizeData 
        cprPrefs = rmfield(cprPrefs,'MarkerSize');
      end
      obj.xyVizFullPlotArgs = struct2paramscell(cprPrefs);
      
      npts = obj.nPts;
      ptsClrs = obj.lObj.PredictPointColors;
      ax = obj.hAxs;
      %arrayfun(@cla,ax);
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.lObj.labeledposIPt2View;
      hTmp = gobjects(npts,1);
      hTmpOther = gobjects(npts,1);
      hTmp2 = gobjects(npts,1);
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        iVw = ipt2View(iPt);
        hTmp(iPt) = plot(ax(iVw),nan,nan,obj.xyVizPlotArgs{:},'Color',clr,'Tag',sprintf('CPRLabelTracker_XYPrdRed_%d',iPt));
        hTmpOther(iPt) = plot(ax(iVw),nan,nan,obj.xyVizPlotArgs{:},'Color',clr,'Tag',sprintf('CPRLabelTracker_XYPrdRedOther_%d',iPt));
        hTmp2(iPt) = scatter(ax(iVw),nan,nan);
        setIgnoreUnknown(hTmp2(iPt),'MarkerFaceColor',clr,...
          'MarkerEdgeColor',clr,'PickableParts','none',...
          'Tag',sprintf('CPRLabelTracker_XYPrdFull_%d',iPt),...
          obj.xyVizFullPlotArgs{:});
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedOther = hTmpOther;
      obj.hXYPrdFull = hTmp2;
    end

    function setHideViz(obj,tf)
      onoff = onIff(~tf);
      [obj.hXYPrdRed.Visible] = deal(onoff);
      [obj.hXYPrdRedOther.Visible] = deal(onoff);
      [obj.hXYPrdFull.Visible] = deal(onoff);
    end
    
    function setShowReplicates(obj,tf)
      [obj.hXYPrdFull.Visible] = deal(onIff(tf));
    end
    
    function clearReplicates(obj)
      hXY = obj.hXYPrdFull;
      if ~isempty(hXY) % can be empty during initHook
        set(hXY,'XData',nan,'YData',nan);
      end
    end
    
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hXYPrdRed(iPt),'Color',clr);
        set(obj.hXYPrdRedOther(iPt),'Color',clr);
        setIgnoreUnknown(obj.hXYPrdFull(iPt),...
          'MarkerFaceColor',clr,'MarkerEdgeColor',clr);
      end
    end
        
    function updateTrackRes(obj,xy,isinterp,xyfull)
      %
      % xy: [npts x 2]
      % isinterp: scalar logical, no longer used
      % xyfull: either [], or [npts x 2 x nrep]
            
      assert(~isinterp,'Interpolation no longer supported.');
      
      npts = obj.nPts;
      hXY = obj.hXYPrdRed;
%       if isinterp
%         plotargs = obj.xyVizPlotArgsInterp;
%       else
%         plotargs = obj.xyVizPlotArgs;
%       end      
      for iPt=1:npts
        set(hXY(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2));%,plotargs{:});
      end
      
      if ~isequal(xyfull,[])
        hXY = obj.hXYPrdFull;
        %plotargs = obj.xyVizFullPlotArgs;
        for iPt = 1:npts
          set(hXY(iPt),'XData',xyfull(iPt,1,:),'YData',xyfull(iPt,2,:));%,plotargs{:});
        end
      end
    end
        
  end

end