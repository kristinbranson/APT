classdef ParameterVisualizationMemory < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false;
      
    hArgs = {'Color','k','LineWidth',2}; 
    propPrefix = '';
    imsz = [];
    nettype = '';
    batchsize = nan;
    downsample = nan;
    is_ma = false;
    is2stage = false;
    is_ma_net = false;
    stage = 1;
    hMem = [];
    hMemCurr = [];
  end
  
  properties (Constant)
    maxBatchSize = 16;
    %maxBatchSizeFactor = 4;
    maxDownsample = 16;
    nDownsamples = 20;
  end
  

  methods

    function setProjImsz(obj)
      % sets .imsz
      obj.imsz = [];
      if obj.lObj.hasTrx || (obj.is_ma && obj.is2stage && (obj.stage==2))
        prmTgtCrop = ParameterVisualizationMemory.getParamValue(obj.prm,'ROOT.MultiAnimal.TargetCrop');
        cropRad = maGetTgtCropRad(prmTgtCrop);
        obj.imsz = cropRad*2+[1,1];
      elseif obj.is_ma,
        if ParameterVisualizationMemory.getParamValue(obj.prm,'ROOT.MultiAnimal.multi_crop_ims'),
          i_sz = ParameterVisualizationMemory.getParamValue(obj.prm,'ROOT.MultiAnimal.multi_crop_im_sz');
        else
          i_sz = obj.lObj.getMovieRoiMovIdx(MovieIndex(1));
          i_sz = max(i_sz(2)-i_sz(1)+1,i_sz(4)-i_sz(3)+1);
        end
        obj.imsz = [i_sz,i_sz];
      else
        nmov = obj.lObj.nmoviesGTaware;
        rois = nan(nmov,obj.lObj.nview,4);
        for i = 1:nmov
          rois(i,:,:) = obj.lObj.getMovieRoiMovIdx(MovieIndex(i));
        end
        if isempty(rois),
          ParameterVisualization.grayOutAxes('No movie loaded in.');
          return;
        end
        
        if obj.lObj.isMultiView
          warningNoTrace('Memory analysis based on first view only.');
        end
        rois = reshape(rois(:,1,:),nmov,4);
        
        hs = rois(:,4)-rois(:,3)+1;
        ws = rois(:,2)-rois(:,1)+1;
        assert(all(hs==hs(1)) && all(ws==ws(1)));
        obj.imsz = [hs(1),ws(1)];
      end
    end

    function setOtherProps(obj)

      obj.downsample = ParameterVisualizationMemory.getParamValue(obj.prm,[obj.propPrefix,'.ImageProcessing.scale']);
      obj.batchsize = ParameterVisualizationMemory.getParamValue(obj.prm,[obj.propPrefix,'.GradientDescent.batch_size']);

      if obj.is_ma,
        if obj.is2stage && obj.stage == 1,
          obj.nettype = obj.lObj.tracker.stage1Tracker.algorithmName;
        else
          obj.nettype = string(obj.lObj.tracker.trnNetType);
        end
      else
        obj.nettype = obj.lObj.tracker.algorithmName;
      end      
      
    end

    function setStage(obj)
      obj.is_ma = obj.lObj.maIsMA;
      obj.is2stage = obj.lObj.trackerIsTwoStage;
      obj.is_ma_net = false;
      obj.stage = 1;

      if obj.is_ma,
        if obj.is2stage && startsWith(obj.propPrefix,'ROOT.DeepTrack'),
          obj.stage = 2;
        else
          obj.is_ma_net = true;
        end
      end
    end

    function [xs,memuses,xcurr,memusecurr,xstr] = getMemUse(obj)
      if endsWith(obj.propFullName,'ImageProcessing.scale')        
        xstr = 'Downsample factor';
        xcurr = obj.downsample;
        xs = logspace(log10(0.25),log10(max(obj.downsample,obj.maxDownsample)),obj.nDownsamples);
        imsz1 = max(1,round(obj.imsz(:)./[xs,xcurr]));
        fprintf('Computing network memory use for batchsize %d, imszs = %s\n',obj.batchsize,mat2str(imsz1));
        memuses = get_network_size(obj.nettype,imsz1,obj.batchsize,obj.is_ma_net);
        fprintf('-> memuses = %s\n',mat2str(memuses));
        memusecurr = memuses(end);
        memuses = memuses(1:end-1);
      elseif endsWith(obj.propFullName,'GradientDescent.batch_size')        
        xstr = 'Batch size';
        xcurr = obj.batchsize;
        xs = 1:max(obj.batchsize,obj.maxBatchSize);
        imsz1 = max(1,round(obj.imsz(:)/obj.downsample));
        memuses = get_network_size(obj.nettype,imsz1,[xs,xcurr],obj.is_ma_net);
        memusecurr = memuses(end);
        memuses = memuses(1:end-1);
      else
        error('Unknown prop %s',obj.propFullName);
      end
    end
            
    function init(obj,hAx,lObj,propFullName,prm)
      %fprintf('init\n');      

      if nargin > 1,
        init@ParameterVisualization(obj,hAx,lObj,propFullName,prm);
      end
      obj.initSuccessful = false;

      [~,idx] = regexp(obj.propFullName,'\.DeepTrack','once');
      assert(numel(idx)==1);
      obj.propPrefix = obj.propFullName(1:idx);
      obj.setStage();
      obj.setProjImsz();
      obj.setOtherProps();

      [xs,memuses,xcurr,memusecurr,xstr] = obj.getMemUse();

      cla(obj.hAx);
      hold(obj.hAx,'off');
      obj.hMem = plot(obj.hAx,xs,memuses/2^10);
      set(obj.hMem,obj.hArgs{:});
      hold(obj.hAx,'on');
      obj.hMemCurr = plot(obj.hAx,xcurr,memusecurr/2^10,'ko','MarkerFaceColor','r');

      xlabel(obj.hAx,xstr);
      ylabel(obj.hAx,'Memory required (GB)');
      
      title(obj.hAx,sprintf('Memory required: %.1f GB',memusecurr/2^10));

      if strcmp(xstr,'Downsample factor'),
        obj.hAx.XScale = 'log';
      else
        obj.hAx.XScale = 'linear';
      end
      axisalmosttight([],obj.hAx);

      obj.initSuccessful = true;

    end
    
    function clear(obj)
      cla(obj.hAx);
      obj.initSuccessful = false;
      obj.hMemCurr = [];
      obj.hMem = [];
    end

    function update(obj)
      if ~obj.initSuccessful,
        obj.init();
        return;
      end
      [xs,memuses,xcurr,memusecurr,xstr] = obj.getMemUse();
      set(obj.hMem,'XData',xs,'YData',memuses/2^10);
      set(obj.hMemCurr,'XData',xcurr,'YData',memusecurr/2^10);
      obj.hAx.XLabel.String = xstr;
      if strcmp(xstr,'Downsample factor'),
        obj.hAx.XScale = 'log';
      else
        obj.hAx.XScale = 'linear';
      end
      axisalmosttight([],obj.hAx);
    end

  end
  
end