classdef XYPlotClickHandler < OlyDat.XYPlotClickSender
  %XYPlotClickHandler Experiment selection and highlighting for XY plots.
  %   XYPlotClickHandler adds selection-response capability to
  %   XYPlotClickSender. The response is to highlight all selected data
  %   points and to list the selected experimentIDs in a textbox.
  %
  %   See OlyDat.XYPlotClickSender.
  
  properties
    infoBackgroundColor = 'k';
    infoTextColor = 'g';
    infoTextFontSize = 11;
    infoTextFontUnits = 'pixels';
    highlightPlotArgs = { ...
      'color',[0 0.5 0],...
      'MarkerSize',10,...
      'Marker','s',...
      'linewidth',3};
  end
  
  properties (Hidden)
    fEid2Idx; % Map from eid->row idx(s) into fData. Values should be row vectors.
    fInfoTextH; % scalar handle to upper-left corner info textbox
    fHilitePointH; % vector of handles to hilite points
  end
  
  methods
    
    %         function delete(obj)
    %             disp('delete xyplotclickhandler');
    %         end
    
    function obj = XYPlotClickHandler(ax,x,y,eids,expCoordinator,tfAddToolbarTool)
      % See OlyDat.XYPlotClickSender/XYPlotClickSender for
      % description of arguments.
      
      if nargin < 6
        tfAddToolbarTool = false;
      end
      obj = obj@OlyDat.XYPlotClickSender(ax,x,y,eids,expCoordinator,tfAddToolbarTool);
      
      if obj.fEidIsCellstr
        m = containers.Map('KeyType','char','ValueType','any');
      else
        m = containers.Map('KeyType',class(eids),'ValueType','any');
      end
      for eidIdx = 1:numel(eids)
        id = eids(eidIdx);
        if obj.fEidIsCellstr
          id = id{1};
        end
        if m.isKey(id)
          oldVal = m(id);
          oldVal(1,end+1) = eidIdx; %#ok<AGROW>
          m(id) = oldVal;
        else
          m(id) = eidIdx;
        end
      end
      obj.fEid2Idx = m;
      
      %%% Configure axis
      
      % Put info text in upper left corner
%       holdstate = ishold(ax);
%       hold(ax,'on');
      
%       xlim = get(ax,'XLim');
%       ylim = get(ax,'YLim');
%       obj.fInfoTextH = text(xlim(1),ylim(2),'Experiment info',...
%         'BackgroundColor',OlyDat.XYPlotClickHandler.infoBackgroundColor,...
%         'Color',OlyDat.XYPlotClickHandler.infoTextColor,...
%         'FontUnits',OlyDat.XYPlotClickHandler.infoTextFontUnits,...
%         'FontSize',OlyDat.XYPlotClickHandler.infoTextFontSize,...
%         'Interpreter','none',...
%         'HorizontalAlignment','left',...
%         'VerticalAlignment','bottom',...
%         'Parent',ax);
      
%       if ~holdstate
%         hold(ax,'off');
%       end
      
      % No hiliter points to start
      obj.fHilitePointH = zeros(0,1);
      
    end
    
  end
  
  methods
    
    function clearHilitePointsAndInfoStr(obj)
      delete(obj.fHilitePointH);
      obj.fHilitePointH = zeros(0,1);
      set(obj.fInfoTextH,'String','No exp selected');
    end
    
    % eids: array of selected eids. If obj.fEidIsCellstr, eids will be
    % a cellstr.
    function respondToEidSelection(obj,eids)
      
      obj.clearHilitePointsAndInfoStr();
      
      % Update hilitePoint
      eidMap = obj.fEid2Idx;
      
      holdState = ishold(obj.fAx);
      hold(obj.fAx,'on');
      
      tfPlotted = false(numel(eids),1);
      for c = 1:numel(eids)
        id = eids(c);
        if obj.fEidIsCellstr
          id = id{1};
        end
        tfPlotted(c) = eidMap.isKey(id);
        if tfPlotted(c)
          idxs = eidMap(id);
          assert(isrow(idxs));
          for idx = idxs
            obj.fHilitePointH(end+1,1) = plot(obj.fAx,...
              obj.fData(idx,1),obj.fData(idx,2),'o',...
              obj.highlightPlotArgs{:},...
              'hittest','off');
          end
        end
      end
      
      if ~holdState
        hold(obj.fAx,'off');
      end
      
      % update infoStr
      plottedEids = eids(tfPlotted);
      tfAnyUnplotted = any(~tfPlotted);
      if numel(plottedEids) > 3
        plottedEids = plottedEids(1:3);
        tfMoreThan3 = true;
      else
        tfMoreThan3 = false;
      end
      
      if obj.fEidIsCellstr
        str = sprintf('%s,',plottedEids{:});
      else
        plottedEids = num2cell(plottedEids);
        str = sprintf('%d,',plottedEids{:});
      end
      if tfMoreThan3
        str = [str '...,'];
      end
      if tfAnyUnplotted
        str = [str 'unplotted,'];
      end
      str = str(1:end-1);
      
      set(obj.fInfoTextH,'String',sprintf('Selected: %s',str));
    end
    
  end
  
end