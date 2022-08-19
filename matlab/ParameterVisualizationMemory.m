classdef ParameterVisualizationMemory < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false; 
      
    hArgs = {'Color','k','LineWidth',2}; 
    imsz = [];
    nettype = '';
    batchsize = nan;
    downsample = nan;
    is_ma = false;
    is2stage = false;
    is_ma_net = false;
    stage = 1;
  end
  
  properties (Constant)
    maxBatchSize = 16;
    %maxBatchSizeFactor = 4;
    maxDownsample = 16;
    nDownsamples = 20;
  end
  
  methods
        
    function propSelected(obj,hAx,lObj,propFullName,sPrm)      
      obj.init(hAx,lObj,propFullName,sPrm);    
    end
    
    function setStage(obj,lObj,prop)
      obj.is_ma = lObj.maIsMA;
      obj.is2stage = lObj.trackerIsTwoStage;
      obj.is_ma_net = false;

      if obj.is_ma && obj.is2stage
        if startsWith(prop,'Deep Learning (pose)')
          obj.stage = 2;
        else
          obj.stage = 1;
          obj.is_ma_net = true;
        end
      else
        obj.stage = 1;
        obj.is_ma_net = true;
      end
    end
    
    function getProjImsz(obj,lObj,sPrm)
      % sets .imsz
      
      if lObj.hasTrx || (obj.is_ma && obj.is2stage && (obj.stage==2))
        prmTgtCrop = sPrm.ROOT.MultiAnimal.TargetCrop;
        cropRad = lObj.maGetTgtCropRad(prmTgtCrop);
        obj.imsz = cropRad*2+[1,1];
      elseif lObj.maIsMA
        if sPrm.ROOT.MultiAnimal.multi_crop_ims
          i_sz = sPrm.ROOT.MultiAnimal.multi_crop_im_sz;
        else
          i_sz = lobj.getMovieRoiMovIdx(MovieIndex(1));
          i_sz = max(i_sz(2)-i_sz(1)+1,i_sz(4)-i_isz(3)+1);
        end
        obj.imsz = [i_sz,i_sz];
      else
        nmov = lObj.nmoviesGTaware;
        rois = nan(nmov,lObj.nview,4);
        for i = 1:nmov
          rois(i,:,:) = lObj.getMovieRoiMovIdx(MovieIndex(i));
        end
        if isempty(rois),
          ParameterVisualization.grayOutAxes(hAx,'No movie loaded in.');
          return;
        end
        
        if lObj.isMultiView
          warningNoTrace('Memory analysis based on first view only.');
        end
        rois = reshape(rois(:,1,:),nmov,4);
        
        hs = rois(:,4)-rois(:,3)+1;
        ws = rois(:,2)-rois(:,1)+1;
        assert(all(hs==hs(1)) && all(ws==ws(1)));
        obj.imsz = [hs(1),ws(1)];
      end
    end
    
    function setOtherProps(obj,lObj,sPrm)
      if obj.is_ma && obj.is2stage && obj.stage == 2
        obj.downsample = sPrm.ROOT.DeepTrack.ImageProcessing.scale;
        obj.nettype = string(lObj.tracker.trnNetType);
        obj.batchsize = sPrm.ROOT.DeepTrack.GradientDescent.batch_size;
      elseif obj.is_ma && obj.is2stage && obj.stage == 1
        obj.downsample = sPrm.ROOT.MultiAnimal.Detect.DeepTrack.ImageProcessing.scale;
        obj.nettype = lObj.tracker.stage1Tracker.algorithmName;
        obj.batchsize = sPrm.ROOT.MultiAnimal.Detect.DeepTrack.GradientDescent.batch_size;        
      elseif obj.is_ma
        obj.downsample = sPrm.ROOT.DeepTrack.ImageProcessing.scale;
        obj.nettype = string(lObj.tracker.trnNetType);
        obj.batchsize = sPrm.ROOT.DeepTrack.GradientDescent.batch_size;        
        
      else
        obj.downsample = sPrm.ROOT.DeepTrack.ImageProcessing.scale;
        obj.nettype = lObj.tracker.algorithmName;
        obj.batchsize = sPrm.ROOT.DeepTrack.GradientDescent.batch_size;        
      end      
      
    end
    
    function init(obj,hAx,lObj,propFullName,sPrm)
      %fprintf('init\n');
      obj.axPos = [.1,.1,.85,.85];
      set(hAx,'Units','normalized','Position',obj.axPos);
      
      obj.initSuccessful = false;
      obj.setStage(lObj,propFullName)      
      obj.getProjImsz(lObj,sPrm);
      obj.setOtherProps(lObj,sPrm);      
  
      if endsWith(propFullName,...
          {'Image Processing.Downsample factor',...
          'Image Processing.scale'})
        
        xstr = 'Downsample factor';
        xs = logspace(log10(0.25),log10(max(obj.downsample,ParameterVisualizationMemory.maxDownsample)),ParameterVisualizationMemory.nDownsamples);
        memuses = nan(size(xs));
        for i = 1:numel(xs),
          imsz1 = max(1,round(obj.imsz/xs(i)));
          memuses(i) = get_network_size(obj.nettype,imsz1,obj.batchsize,obj.is_ma_net);
        end
        xcurr = obj.downsample;
        imsz1 = max(1,round(obj.imsz/xcurr));
        memusecurr = get_network_size(obj.nettype,imsz1,obj.batchsize,obj.is_ma_net);
      elseif endsWith(propFullName,...
          {'Gradient Descent.Training batch size','DeepTrack.GradientDescent.batch_size'})
        
        xstr = 'Batch size';
        xs = 1:max(obj.batchsize,ParameterVisualizationMemory.maxBatchSize);
        imsz1 = max(1,round(obj.imsz/obj.downsample));
        memuses = nan(size(xs));
        for i = 1:numel(xs),
          memuses(i) = get_network_size(obj.nettype,imsz1,xs(i),obj.is_ma_net);
        end
        xcurr = obj.batchsize;
        memusecurr = get_network_size(obj.nettype,imsz1,xcurr,obj.is_ma_net);
      else
        error('Unknown prop %s',propFullName);
      end
      
      cla(hAx);
      hold(hAx,'off');
      h = plot(hAx,xs,memuses/2^10);
      set(h,obj.hArgs{:});
      hold(hAx,'on');
      plot(hAx,xcurr,memusecurr/2^10,'ko','MarkerFaceColor','r');

      axisalmosttight([],hAx);
      xlabel(hAx,xstr);
      ylabel(hAx,'Memory required (GB)');
      
      title(hAx,sprintf('Memory required: %.1f GB',memusecurr/2^10));
      obj.initSuccessful = true;
    end
    
    function propUnselected(obj)
      obj.imsz = [];
      obj.nettype = '';
      obj.batchsize = nan;
      obj.downsample = nan;
      obj.initSuccessful = false;
      obj.stage = 1;
      obj.is_ma = false;
      obj.is_ma_net = false;
    end

    function propUpdated(obj,hAx,lObj,propFullName,sPrm)
      %fprintf('propUpdated: %s.\n',propFullName);
      
      if obj.initSuccessful,
        switch propFullName,
        case {'DeepTrack.ImageProcessing.Downsample factor','DeepTrack.ImageProcessing.scale'},
          if sPrm.ROOT.DeepTrack.ImageProcessing.scale == obj.downsample,
            return;
          end
        case {'DeepTrack.GradientDescent.Training batch size','DeepTrack.GradientDescent.batch_size'},
          if sPrm.ROOT.DeepTrack.GradientDescent.batch_size == obj.batchsize,
            return;
          end
        end
      end
      
      obj.init(hAx,lObj,propFullName,sPrm);
    end
    
    function propUpdatedDynamic(obj,hAx,lObj,propFullName,sPrm,val) 
      %fprintf('propUpdatedDynamic: %s = %f.\n',propFullName,val);
      try
        eval(sprintf('sPrm.ROOT.%s = %f;',propFullName,val));
      catch
        warningNoTrace(sprintf('Unknown property %s',propFullName));
        return;
      end
      obj.propUpdated(hAx,lObj,propFullName,sPrm)
    end
    
  end
  
end