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
    
    function getProjImsz(obj,lObj,sPrm)
      % sets .imsz
      
      if lObj.hasTrx,
        obj.imsz = sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius*2+[1,1];
      else
        nmov = lObj.nmoviesGTaware;
        rois = nan(nmov,lObj.nview,4);
        for i = 1:nmov,
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
    
    function init(obj,hAx,lObj,propFullName,sPrm)
      %fprintf('init\n');
      obj.axPos = [.1,.1,.85,.85];
      set(hAx,'Units','normalized','Position',obj.axPos);
      
      obj.initSuccessful = false;
      obj.getProjImsz(lObj,sPrm);
      obj.downsample = sPrm.ROOT.DeepTrack.ImageProcessing.scale;
      
      obj.nettype = lObj.tracker.algorithmName;
      obj.batchsize = sPrm.ROOT.DeepTrack.GradientDescent.batch_size;
  
      if endsWith(propFullName,...
          {'DeepTrack.ImageProcessing.Downsample factor','DeepTrack.ImageProcessing.scale'})
        
        xstr = 'Downsample factor';
        xs = logspace(0,log10(max(obj.downsample,ParameterVisualizationMemory.maxDownsample)),ParameterVisualizationMemory.nDownsamples);
        memuses = nan(size(xs));
        for i = 1:numel(xs),
          imsz1 = max(1,round(obj.imsz/xs(i)));
          memuses(i) = get_network_size(obj.nettype,imsz1,obj.batchsize);
        end
        xcurr = obj.downsample;
        imsz1 = max(1,round(obj.imsz/xcurr));
        memusecurr = get_network_size(obj.nettype,imsz1,obj.batchsize);
      elseif endsWith(propFullName,...
          {'DeepTrack.GradientDescent.Training batch size','DeepTrack.GradientDescent.batch_size'})
        
        xstr = 'Batch size';
        xs = 1:max(obj.batchsize,ParameterVisualizationMemory.maxBatchSize);
        imsz1 = max(1,round(obj.imsz/obj.downsample));
        memuses = nan(size(xs));
        for i = 1:numel(xs),
          memuses(i) = get_network_size(obj.nettype,imsz1,xs(i));
        end
        xcurr = obj.batchsize;
        memusecurr = get_network_size(obj.nettype,imsz1,xcurr);
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