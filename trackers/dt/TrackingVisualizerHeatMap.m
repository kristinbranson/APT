classdef TrackingVisualizerHeatMap < handle
  
  properties
    lObj % Included only to access the current raw image. Ideally used as little as possible
    
    hAxs % [nview] axes handles. Owned by Labeler
    hIms % [nview] image handles. Owned by Labeler
    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3] RGB for pts
    
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
%     xyVizPlotArgs; % cell array of args for regular tracking viz
%     xyVizPlotArgsNonTarget; % " for non current target viz
    
    heatMapEnable % if true, read heatmaps (alongside trkfiles) when changing movies, do heatmap viz
    heatMapRawImType % 'none','reg','invert'
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
      obj.heatMapRawImType = 'reg';
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
      % Sets .hXYPrdRed, .hXYPrdRedOther
    
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      deleteValidHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
       
      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.projPrefs.Track; 
      plotPrefs = trackPrefs.PredictPointsPlot; 
      plotPrefs.PickableParts = 'none'; 
      xyVizPlotArgs = struct2paramscell(plotPrefs); 
%       obj.xyVizPlotArgsNonTarget = obj.xyVizPlotArgs; % TODO: customize 
       
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
        hTmp(iPt) = plot(axs(iVw),nan,nan,xyVizPlotArgs{:},'Color',clr); 
        hTmpOther(iPt) = plot(axs(iVw),nan,nan,xyVizPlotArgs{:},'Color',clr);
%         hTmp2(iPt) = scatter(ax(iVw),nan,nan); 
%         setIgnoreUnknown(hTmp2(iPt),'MarkerFaceColor',clr,... 
%           'MarkerEdgeColor',clr,'PickableParts','none',... 
%           obj.xyVizFullPlotArgs{:}); 
      end 
      obj.hXYPrdRed = hTmp; 
      obj.hXYPrdRedOther = hTmpOther; 
    end
    
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      for iPt=1:npts
        set(obj.hXYPrdRed(iPt),'Color',ptsClrs(iPt,:));
        set(obj.hXYPrdRedOther(iPt),'Color',ptsClrs(iPt,:));
      end
    end
    
    function setHideViz(obj,tf)
      onoff = onIff(~tf);
      [obj.hXYPrdRed.Visible] = deal(onoff);
      [obj.hXYPrdRedOther.Visible] = deal(onoff);
    end
    
    function heatMapInit(obj,hmdir,hmnr,hmnc)
      lblrObj = obj.lObj;
      % TODO: multiview
      nfrm = lblrObj.nframes;
      ntgt = lblrObj.nTargets;      
      obj.heatMapReader.init(hmdir,hmnr,hmnc,nfrm,obj.nPts,ntgt);
    end
    
    function updateTrackRes(obj,xy,currFrm,currTgt,trxXY,trxTh)
      % Update 'final tracking' markers; if .heatMapEnable, also update
      % images with heatmap data. Avoid using separate axis or transparent
      % layer for heatmap info for perf issues.
      %
      % trxXY, trxTh: can be [] if no trx. Used for heatmaps
      %
      % xy: [npts x 2]
            
      hXY = obj.hXYPrdRed;
      %args = obj.xyVizPlotArgs;
      for iPt=1:obj.nPts
        set(hXY(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2)); %,args{:});
      end
      
      if obj.heatMapEnable
        ims = obj.hIms;
        assert(numel(ims)==1,'Multiview projects currently unsupported.');
        currIms = obj.lObj.currIm; % ASSUMED TO BE correct for currFrm. Hmm.
        
        iptsHM = obj.heatMapIPtsShow;
        hm = obj.heatMapReader.read(currFrm,iptsHM,currTgt); % [imnr x imnc x nptsHM]
        
        for ivw=1:numel(ims)
          switch obj.heatMapRawImType
            case 'none'
              imStart = zeros(size(currIms{ivw}));
            case 'reg'
              imStart = im2double(currIms{ivw});
            case 'invert'
              imStart = 1-im2double(currIms{ivw});
          end
          % pi/2 b/c heatmaps oriented so target points towards smaller y
          % ("up" in "axis ij" mode, "down" in "axis xy" mode)
          imHeatmapped = obj.heatMappifyImage(imStart,hm,iptsHM,trxXY,trxTh);
          set(ims(ivw),'CData',imHeatmapped);
          % caxis etc?          
        end
      end
    end
    
    function im1 = heatMappifyImage(obj,im0,hm,iptsHM,trxXY,trxTheta)
      % im0: [imnr x imnc] raw grayscale image (1 chan only, raw data type)
      % hm: [imnr x imnc x niptsHM] raw heatmaps (normalization/scale unk
      %     for each pt, raw data type). The size of hm can differ from im0
      %     (imnr/imnc) if hmCtXYr,trxTheta are supplied.
      % iptsHM: [niptsHM] pt indices labeling 3rd dim of hm
      % trxXY: [], or [2] (opt) trx center in the original movie coords
      % trxTheta: [], or [1] (opt) trx theta in the original movie coords
      %
      % If trxXY, trxTheta are not supplied, hm will have the same row/col
      % size as im0. Otherwise hm can have arbitrary size
      %
      % im1: [imnr x imnc x 3] RGB image with heatmap coloring
      
      xformHM = ~isempty(trxXY); 
      assert(~xor(xformHM,~isempty(trxTheta)));
      
      [hmnr,hmnc,hmnpts] = size(hm);
      assert(hmnpts==numel(iptsHM));
      
      assert(size(im0,3)==1);
      [imnr,imnc] = size(im0);
      %im0 = HistEq.normalizeGrayscaleIm(im0);
      im1 = repmat(im0,1,1,3);
      
      hm = double(hm);
      if xformHM
        % prep for transform
        % MK: if hmnc,hmnr=180,180, then x,y in trx should be at (91,91) of
        % heatmap
        xgv = -floor(hmnc/2):floor((hmnc-1)/2);
        ygv = -floor(hmnr/2):floor((hmnr-1)/2);
        [xg0,yg0] = meshgrid(xgv,ygv);        
        [xg1,yg1] = meshgrid(1:imnc,1:imnr);        
      end
      
      ptclrs = obj.ptClrs;
      for iipt = 1:numel(iptsHM)
        ipt = iptsHM(iipt);
        hmI = hm(:,:,iipt);
        if xformHM
          % don't worry about overall normalization since we are
          % normalizing next
          hmI = readpdf(hmI,xg0,yg0,xg1,yg1,trxXY(1),trxXY(2),trxTheta+pi/2);  
        end
        hmI = hmI/max(hmI(:));
        im1 = (1-hmI).*im1 + hmI.*reshape(ptclrs(ipt,:),[1 1 3]);
      end
    end
    
  end

end