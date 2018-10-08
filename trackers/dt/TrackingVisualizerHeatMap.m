classdef TrackingVisualizerHeatMap < handle
  
  properties
    lObj % Included only to access the current raw image. Ideally used as little as possible
    
    hAxs % [nview] axes handles. Owned by Labeler
    hIms % [nview] image handles. Owned by Labeler
    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3] RGB for pts
    
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
    xyVizPlotArgs; % cell array of args for regular tracking viz
    xyVizPlotArgsNonTarget; % " for non current target viz
    
    heatMapEnable % if true, read heatmaps (alongside trkfiles) when changing movies, do heatmap viz
    heatMapReader % scalar HeatMapReader
    heatMapIPtsShow % [nptsShowHM] ipt indices into 1..npts. Show these points in heatmaps    
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
    
    function obj = TrackingVisualizerHeatMap(lObj)
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
      obj.ipt2vw = lObj.labeledposIPt2View;
      
      npts = numel(obj.ipt2vw);
      obj.ptClrs = lines(npts);
      szassert(obj.ptClrs,[npts 3]);

      obj.heatMapEnable = false;
      obj.heatMapReader = HeatmapReader();
      obj.heatMapIPtsShow = 1:npts;
    end
    
    function delete(obj)
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      deleteValidHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
    end
    
  end
  
  methods
    
    function vizInit(obj)
      % Sets .hXYPrdRed, .hXYPrdRedOther, .xyVizPlotArgs, .xyVizPlotArgsNonTarget
    
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      deleteValidHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
       
      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.projPrefs.Track; 
      plotPrefs = trackPrefs.PredictPointsPlot; 
      plotPrefs.PickableParts = 'none'; 
      obj.xyVizPlotArgs = struct2paramscell(plotPrefs); 
      obj.xyVizPlotArgsNonTarget = obj.xyVizPlotArgs; % TODO: customize 
       
      npts = obj.nPts;
      ptsClrs = obj.ptClrs;
      ipt2View = obj.ipt2vw;
      axs = obj.hAxs;

      arrayfun(@(x)hold(x,'on'),axs);
      hTmp = gobjects(npts,1); 
      hTmpOther = gobjects(npts,1); 
      for iPt=1:npts 
        clr = ptsClrs(iPt,:); 
        iVw = ipt2View(iPt); 
        hTmp(iPt) = plot(axs(iVw),nan,nan,obj.xyVizPlotArgs{:},'Color',clr); 
        hTmpOther(iPt) = plot(axs(iVw),nan,nan,obj.xyVizPlotArgs{:},'Color',clr);
%         hTmp2(iPt) = scatter(ax(iVw),nan,nan); 
%         setIgnoreUnknown(hTmp2(iPt),'MarkerFaceColor',clr,... 
%           'MarkerEdgeColor',clr,'PickableParts','none',... 
%           obj.xyVizFullPlotArgs{:}); 
      end 
      obj.hXYPrdRed = hTmp; 
      obj.hXYPrdRedOther = hTmpOther; 
    end
    
    function setHideViz(obj,tf)
      onoff = onIff(~tf);
      [obj.hXYPrdRed.Visible] = deal(onoff);
      [obj.hXYPrdRedOther.Visible] = deal(onoff);
    end
    
    function heatMapInit(obj,hmdir)
      lblrObj = obj.lObj;
      % TODO: multiview
      imnr = lblrObj.movienr;
      imnc = lblrObj.movienc;
      nfrm = lblrObj.nframes;
      ntgt = lblrObj.nTargets;      
      obj.heatMapReader.init(hmdir,imnr,imnc,nfrm,obj.nPts,ntgt);
    end
    
    function updateTrackRes(obj,xy,currFrm,currTgt)
      % Update 'final tracking' markers; if .heatMapEnable, also update
      % images with heatmap data. Avoid using separate axis or transparent
      % layer for heatmap info for perf issues.
      %
      % xy: [npts x 2]
            
      hXY = obj.hXYPrdRed;
      args = obj.xyVizPlotArgs;
      for iPt=1:obj.nPts
        set(hXY(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2),args{:});
      end
      
      if obj.heatMapEnable
        ims = obj.hIms;
        assert(numel(ims)==1,'Multiview projects currently unsupported.');
        currIms = obj.lObj.currIm; % ASSUMED TO BE correct for currFrm. Hmm.
        
        iptsHM = obj.heatMapIPtsShow;
        hm = obj.heatMapReader.read(currFrm,iptsHM,currTgt); % [imnr x imnc x nptsHM]
        
        for ivw=1:numel(ims)
          imHeatmapped = obj.heatMappifyImage(currIms{ivw},hm,iptsHM);
          set(ims(ivw),'CData',imHeatmapped);
          % caxis etc?          
        end
      end
    end
    
    function im1 = heatMappifyImage(obj,im0,hm,iptsHM)
      % im0: [imnr x imnc] raw grayscale image (1 chan only, raw data type)
      % hm: [imnr x imnc x niptsHM] raw heatmaps (normalization/scale unk
      %     for each pt, raw data type)
      % iptsHM: [niptsHM] pt indices labeling 3rd dim of hm
      %
      % im1: [imnr x imnc x 3] RGB image with heatmap coloring
      
      assert(size(im0,3)==1);
      im0 = HistEq.normalizeGrayscaleIm(im0);
      im1 = repmat(im0,1,1,3);
      
      hm = double(hm);
      
      ptclrs = obj.ptClrs;
      for iipt = 1:numel(iptsHM)
        ipt = iptsHM(iipt);
        hmI = hm(:,:,iipt);
        hmI = hmI/max(hmI(:));
        im1 = im1 + hmI.*reshape(ptclrs(ipt,:),[1 1 3]);
      end
    end
    
  end

end