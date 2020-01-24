classdef TrackingVisualizerHeatMap < TrackingVisualizer
  
  properties        
    heatMapEnable % if true, read heatmaps (alongside trkfiles) when changing movies, do heatmap viz
    heatMapRawImType % 'none','reg','invert'
    heatMapReader % scalar HeatMapReader
    heatMapIPtsShow % [nptsShowHM] ipt indices into 1..npts. Show these points in heatmaps    
  end
  
  methods
    
    function obj = TrackingVisualizerHeatMap(lObj)
      obj = obj@TrackingVisualizer(lObj,'TrackingVisualizerHeatMap');
      
      obj.heatMapEnable = false;
      obj.heatMapRawImType = 'reg';
      obj.heatMapReader = HeatmapReader();
      obj.heatMapIPtsShow = 1:obj.nPts;
    end
    
    function delete(obj)
    end
    
  end
  
  methods
    
    function heatMapInit(obj,hmdir,hmnr,hmnc)
      lblrObj = obj.lObj;
      % TODO: multiview
      nfrm = lblrObj.nframes;
      ntgt = lblrObj.nTargets;      
      obj.heatMapReader.init(hmdir,hmnr,hmnc,nfrm,obj.nPts,ntgt);
    end
    
    function updateTrackRes(obj,xy,tfocc,currFrm,currTgt,trxXY,trxTh)
      % Update 'final tracking' markers; if .heatMapEnable, also update
      % images with heatmap data. Avoid using separate axis or transparent
      % layer for heatmap info for perf issues.
      %
      % trxXY, trxTh: can be [] if no trx. Used for heatmaps
      %
      % xy: [npts x 2]
         
      updateTrackRes@TrackingVisualizer(obj,xy,tfocc);
      
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