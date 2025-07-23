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
  

  methods(Static)
    function imsz = getProjImsz(lObj,prm,is_ma,is2stage,stage)
      % sets .imsz
      imsz = [];
      if lObj.hasTrx || (is_ma && is2stage && (stage==2))
        prmTgtCrop = ParameterVisualizationMemory.getParamValue(prm,'ROOT.MultiAnimal.TargetCrop');
        cropRad = maGetTgtCropRad(prmTgtCrop);
        imsz = cropRad*2+[1,1];
      elseif lObj.maIsMA
        if ParameterVisualizationMemory.getParamValue(prm,'ROOT.MultiAnimal.multi_crop_ims'),
          i_sz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.MultiAnimal.multi_crop_im_sz');
        else
          i_sz = lObj.getMovieRoiMovIdx(MovieIndex(1));
          i_sz = max(i_sz(2)-i_sz(1)+1,i_sz(4)-i_sz(3)+1);
        end
        imsz = [i_sz,i_sz];
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
        imsz = [hs(1),ws(1)];
      end
    end

    function [ds,nettype,bsz] = getOtherProps(lObj,prm,is_ma,is2stage,stage)

      ds =1; nettype= ''; bsz = 1;
      if is_ma && is2stage && stage == 2
        ds = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.ImageProcessing.scale');
        nettype = string(lObj.tracker.trnNetType);
        bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.GradientDescent.batch_size');
      elseif is_ma && is2stage && stage == 1
        ds = ParameterVisualizationMemory.getParamValue(prm,'ROOT.MultiAnimal.Detect.DeepTrack.ImageProcessing.scale');
        nettype = lObj.tracker.stage1Tracker.algorithmName;
        bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.MultiAnimal.Detect.DeepTrack.GradientDescent.batch_size');        
      elseif is_ma
        ds = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.ImageProcessing.scale');
        nettype = string(lObj.tracker.trnNetType);
        bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.GradientDescent.batch_size');        
        
      else
        ds = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.ImageProcessing.scale');
        nettype = lObj.tracker.algorithmName;
        bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.GradientDescent.batch_size');        
      end      

      % ds = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.ImageProcessing.scale');
      % if is_ma && is2stage && stage == 1
      %   nettype = lObj.tracker.stage1Tracker.algorithmName;
      %   bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.MultiAnimal.Detect.DeepTrack.GradientDescent.batch_size');
      % elseif is_ma
      %   nettype = string(lObj.tracker.trnNetType);        
      %   bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.GradientDescent.batch_size');
      % else
      %   nettype = lObj.tracker.algorithmName;
      %   bsz = ParameterVisualizationMemory.getParamValue(prm,'ROOT.DeepTrack.GradientDescent.batch_size');
      % end
      
    end

    function [is_ma,is2stage,is_ma_net,stage] = getStage(lObj,prop)
      is_ma = lObj.maIsMA;
      is2stage = lObj.trackerIsTwoStage;
      is_ma_net = false;
      stage = 1;

      if is_ma 
        if is2stage
          if startsWith(prop,'Deep Learning (pose)') || startsWith(prop,'DeepTrack'),
            stage = 2;
          else
            stage = 1;
            is_ma_net = true;
          end
        else
          stage = 1;
          is_ma_net = true;
        end
      end
    end



  end

  methods
        
    function propSelected(obj,hAx,lObj,propFullName,prm)      
      obj.init(hAx,lObj,propFullName,prm);    
    end
    
        
    
    function init(obj,hAx,lObj,propFullName,prm)
      %fprintf('init\n');      
      if ~strcmp(hAx.Parent.Type,'tiledlayout'),
        obj.axPos = [.1,.1,.85,.85];
        set(hAx,'Units','normalized','Position',obj.axPos);
      end
      
      obj.initSuccessful = false;
      [is_ma,is2stage,stage,is_ma_net] = ...
        ParameterVisualizationMemory.getStage(...
        lObj,propFullName);
      obj.is_ma = is_ma;
      obj.is2stage = is2stage;
      obj.stage = stage;
      obj.is_ma_net = is_ma_net;
      obj.imsz = ParameterVisualizationMemory.getProjImsz(...
        lObj,prm,obj.is_ma,obj.is2stage,obj.stage);
      [ds,nettype,bsz] = ParameterVisualizationMemory.getOtherProps(...
        lObj,prm,is_ma,is2stage,stage);
      obj.downsample = ds;
      obj.nettype = nettype;
      obj.batchsize = bsz;
  
      if endsWith(propFullName,...
          {'Image Processing.Downsample factor',...
          'ImageProcessing.scale'})
        
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

    function propUpdated(obj,hAx,lObj,propFullName,prm)
      %fprintf('propUpdated: %s.\n',propFullName);      
      obj.init(hAx,lObj,propFullName,prm);
    end
    
    function propUpdatedDynamic(obj,hAx,lObj,propFullName,prm,val) 
      %fprintf('propUpdatedDynamic: %s = %f.\n',propFullName,val);
      propFullName = ParameterVisualizationMemory.modernizePropName(propFullName);
      try
        if isstruct(prm),
          eval(sprintf('prm.%s = %f;',propFullName,val));
        end
      catch
        warningNoTrace(sprintf('Unknown property %s',propFullName));
        return;
      end
      obj.propUpdated(hAx,lObj,propFullName,prm)
    end
    
  end
  
end