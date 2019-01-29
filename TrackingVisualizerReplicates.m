classdef TrackingVisualizerReplicates < TrackingVisualizer
  
  properties
    hXYPrdFull; % [npts] scatter handles for replicates, current frame, current target
  end
  
  methods
    function obj = TrackingVisualizerReplicates(lObj)
      obj = obj@TrackingVisualizer(lObj,'CPRLabelTracker');
    end    
    function delete(obj)
      obj.deleteGfxHandlesReplicates();
    end
  end
  
  methods
    function deleteGfxHandlesReplicates(obj)
      deleteValidHandles(obj.hXYPrdFull);
      obj.hXYPrdFull = [];   
    end        
    function vizInitHook(obj)      
      cprPrefs = obj.lObj.projPrefs.CPRLabelTracker.PredictReplicatesPlot;
      cprPrefs.SizeData = cprPrefs.MarkerSize^2; % Scatter.SizeData 
      cprPrefs = rmfield(cprPrefs,'MarkerSize');

      xyVizFullPlotArgs = struct2paramscell(cprPrefs);
      
      npts = obj.nPts;
      ptsClrs = obj.ptClrs;
      ax = obj.hAxs;
      ipt2View = obj.ipt2vw;
      hTmp2 = gobjects(npts,1);
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        iVw = ipt2View(iPt);
        hTmp2(iPt) = scatter(ax(iVw),nan,nan);
        setIgnoreUnknown(hTmp2(iPt),'MarkerFaceColor',clr,...
          'MarkerEdgeColor',clr,'PickableParts','none',...
          'Tag',sprintf('%s_XYPrdFull_%d',obj.handleTagPfix,iPt),...
          xyVizFullPlotArgs{:});
      end
      obj.hXYPrdFull = hTmp2;
    end

    function setHideViz(obj,tf)
      onoff = onIff(~tf);
      setHideViz@TrackingVisualizer(obj,onoff);
      [obj.hXYPrdFull.Visible] = deal(onoff);
    end
    
    function updateLandmarkColors(obj,ptsClrs)
      updateLandmarkColors@TrackingVisualizer(obj,ptsClrs);
      npts = obj.nPts;
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        setIgnoreUnknown(obj.hXYPrdFull(iPt),...
          'MarkerFaceColor',clr,'MarkerEdgeColor',clr);
      end
    end
    
    % Hmm maybe just put the replicates .hXYPrdFull into base cls. Wait and
    % see
    
    function setShowReplicates(obj,tf)
      [obj.hXYPrdFull.Visible] = deal(onIff(tf));
    end
    
    function clearReplicates(obj)
      hXY = obj.hXYPrdFull;
      if ~isempty(hXY) % can be empty during initHook
        set(hXY,'XData',nan,'YData',nan);
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
      for iPt=1:npts
        set(hXY(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2));
      end
      
      if ~isequal(xyfull,[])
        hXY = obj.hXYPrdFull;
        for iPt=1:npts
          set(hXY(iPt),'XData',xyfull(iPt,1,:),'YData',xyfull(iPt,2,:));
        end
      end
    end
        
  end

end