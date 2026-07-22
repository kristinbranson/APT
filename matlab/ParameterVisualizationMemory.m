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

      [obj.imsz,obj.downsample,obj.batchsize] = obj.lObj.trackGetTrainImageSize('stages',obj.stage,'sPrm',obj.prm.structize());

    end

    function setNetType(obj)

      obj.nettype = obj.lObj.trackGetNetType('stages',obj.stage);
      obj.nettype = obj.nettype{1};
      
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
            
    function init(obj,hTile,lObj,propFullName,prm,varargin)
      %fprintf('init\n');      

      if nargin > 1,
        init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
      end
      obj.initSuccessful = false;
      [xs,memuses,xcurr,memusecurr,xstr] = obj.getMemUse();        

      if isempty(obj.hAx),
        obj.hAx = nexttile(obj.hTile);
      end

      [~,idx] = regexp(obj.propFullName,'\.DeepTrack','once');
      assert(numel(idx)==1);
      obj.propPrefix = obj.propFullName(1:idx);
      obj.setStage();
      obj.setProjImsz();
      obj.setNetType();

      cla(obj.hAx);
      hold(obj.hAx,'off');
      if isempty(memuses),
        obj.hMem = text(.5,.5,sprintf('No data on GPU memory for %s',obj.nettype),...
          'HorizontalAlignment','center','VerticalAlignment','middle','Parent',obj.hAx,'Interpreter','none');
        axis(obj.hAx,'off');
      else
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
      end

      obj.initSuccessful = true;

    end

    function clear(obj)
      clear@ParameterVisualization(obj);
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
      if isempty(memuses),
        return;
      end
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