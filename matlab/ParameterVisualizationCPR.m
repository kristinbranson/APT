classdef ParameterVisualizationCPR < ParameterVisualization
  
  properties
    initSuccessful = false;
    initVizInfo = [];% scalar struct with info for updating plot
  end
  
  methods
    
    function propSelected(obj,hAx,lObj,propFullName,prm)
      obj.init(hAx,lObj,prm);
    end
    
    function propUnselected(obj)
      obj.initSuccessful = false;
      obj.initVizInfo = [];
    end

    function propUpdated(obj,hAx,lObj,propFullName,prm)
      %prmFtr = sPrm.ROOT.CPR.Feature;
      obj.init(hAx,lObj,prm);
    end

    function propUpdatedDynamic(obj,hAx,lObj,propFullName,prm,val) %#ok<INUSD>
      

      try
        ParameterVisualizationCPR.getParamValue(prm,propFullName);
      catch 
        warningNoTrace(sprintf('Unknown property %s',propFullName));
        return;
      end
      % don't need to set values if treeNode
      if isstruct(prm),
        eval(sprintf('sPrm.%s = val;',propFullName));
      end
      
%       % to do: store val in sPrm
%       switch propFullName,
%         case 'ImageProcessing.Multiple Targets.Target ROI.Pad background'
%           sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.PadBkgd = val;
%         case 'ImageProcessing.Histogram Equalization.Enable'
%           sPrm.ROOT.ImageProcessing.HistEq.Use = val;
%         case 'ImageProcessing.Histogram Equalization.Num frames sample'
%           sPrm.ROOT.ImageProcessing.HistEq.NSampleH0 = val;
%         case 'ImageProcessing.Background Subtraction.Enable',
%           sPrm.ROOT.ImageProcessing.BackSub.Use = val;
%         case 'ImageProcessing.Background Subtraction.Background Type',
%           sPrm.ROOT.ImageProcessing.BackSub.BGType = val;
%         case 'ImageProcessing.Background Subtraction.Background Read Function',
%           sPrm.ROOT.ImageProcessing.BackSub.BGReadFcn = val;
%         otherwise
%           error('Unknown property changed: %s',propFullName);
%       end
      
      obj.init(hAx,lObj,prm);
      
    end
    
    function init(obj,hAx,lObj,prm)
      % plot sample processed training images
      % Set .initSuccessful, initVizInfo
      % Subsequent changes to can be handled via update(). This avoids
      % recollecting all training labels.
      
      if obj.initSuccessful && isstruct(obj.initVizInfo) && all(ishandle(obj.initVizInfo.hp)),
        return;
      end

      if ~strcmp(hAx.Parent.Type,'tiledlayout'),
        set(hAx,'Units','normalized','Position',obj.axPos);
      end
      
      obj.initSuccessful = false;
      obj.initVizInfo = [];
      success = false;
      for i = 1:numel(lObj.tracker),
        if strcmpi(lObj.tracker(i).algorithmName,'cpr'),
          success = true;
          tracker = lObj.tracker(i);
          break;
        end
      end
      if ~success,
        ParameterVisualization.grayOutAxes(hAx,'No tracker has been trained using CPR.');
        return;
      end
      if isempty(lObj.tracker.lastTrainStats),
        ParameterVisualization.grayOutAxes(hAx,'CPR tracker has not been trained. Please train to view current timing info');
        return;
      end
      
      ParameterVisualization.setBusy(hAx,'Computing visualization. Please wait...');
      
      obj.initVizInfo = struct;
      cla(hAx);
      axis(hAx,'auto');
      axis(hAx,'on');
      box(hAx,'off');
      set(hAx,'DataAspectRatioMode','auto','YDir','normal');
      [obj.initVizInfo.hp,obj.initVizInfo.ht] = tracker.plotTiming('hAx',hAx);
      maxy = max(cellfun(@max,get(obj.initVizInfo.hp,'YData')));
      ylim = get(hAx,'YLim');
      ylim(2) = maxy+.5;
      xlim = get(hAx,'XLim');
      set(hAx,'YLim',ylim);
      text(mean(xlim),maxy+.25,{'Timing information based on tracker trained on',...
        datestr(tracker.lastTrainStats.time.start),...
        '*not current parameters*'},...
        'HorizontalAlignment','center','VerticalAlignment','middle','Color','k');
      hAx.XTickMode = 'auto';
      hAx.YTickMode = 'auto';

      obj.initSuccessful = true;
      
      ParameterVisualization.setReady(hAx);
      obj.initSuccessful = true;
      
    end
        
  end
  
end