classdef XYPlotClickSender < handle
  %XYPlotClickSender Experiment/data selection for XY plots.
  %   XYPlotClickSender enables data selection for an XY-scatter plot. Data
  %   selection can occur either via i) single-clicking individual data
  %   points, or ii) dragging a bounding box around multiple points.
  %   Shift-click enables aggregation of selections.
  %
  %   XYPlotClickSender works with OlyDat.ExperimentCoordinator to broadcast
  %   the selected experimentIDs.
  %
  %   To use XYPlotClickSender, simply construct one with your axes and data.
  %   The constructed object will live within the axes' callbacks. When the
  %   axes is deleted, the object will be cleaned up.
  %
  %   XYPlotClickSender only sends signals to the ExperimentCoordinator, it
  %   does not respond to them. You can add this capability by deriving from
  %   XYPlotClickSender and overriding respondToEidSelection().
  %
  %   See OlyDat.ExperimentCoordinator, OlyDat.XYPlotClickHandler,
  %   OlyDat.Plot.
  
  properties (Constant)
    maxDistFrac = .1; % threshold for how close you have to click to a data point to trigger selection
  end
  
  properties (Hidden)
    fAx; % scalar handle to axis.
    fData; % Nx2 array. Each row represents a point (x,y) in the plot.
    fEid; % Nx1 array of experiment ids corresponding to rows in fData; can also be a cellstr
    fEidIsCellstr; % if true, fEid is a cellstr
    fExpCoordinator; % experiment coordinator object
    
    fSelectionMode; % string enum, either 'singleClick' or 'boundingBox'
    
    fBoundingBoxPatchH; % 2x1 handle to patches used in bounding-box selection mode
    fBoundingBoxCoords; % 1x4 vector for bounding-box [x1 y1 x2 y2]
    fBoundingBoxOn; % tf for bounding-box mode
    fBoundingBoxShiftOn; % tf for bounding-box mode
    fBoundingBoxToggleToolH; % handle to bounding-box uitoggletool
  end
  
  properties (SetAccess=private)
    fSelectedEids; % vector of currently selected eids.
  end
  
  methods
    
    %         function delete(obj)
    %             disp('xyplotclicksender dtor');
    %         end
    
    function obj = XYPlotClickSender(ax,x,y,eids,expCoordinator,tfAddToolbarTool)
      % ax: scalar axis handle
      % x: col vec of x coordinates
      % y: col vec of y coordinates
      % eids: col vec of experiment IDs corresponding to rows of x/y
      % expCoordinator: scalar ExperimentCoordinator object
      % tfAddToolbarTool: if true, add a bounding-box uitoggletool to
      % the axis' parent figure.
      
      assert(isscalar(ax) && ishandle(ax) && strcmp(get(ax,'type'),'axes'));
      obj.fAx = ax;
      
      validateattributes(x,{'numeric'},{'vector'});
      validateattributes(y,{'numeric'},{'vector'});
      validateattributes(eids,{'numeric' 'cell'},{'vector'});
      assert(isequal(numel(x),numel(y),numel(eids)));
      obj.fData = [x(:) y(:)];
      obj.fEid = eids(:);
      obj.fEidIsCellstr = iscellstr(obj.fEid);
      if obj.fEidIsCellstr
        warning('OlyDat:XYPlotClickSender:cellEids',...
          'Cellstr eids are not fully supported. Proceed at your own risk.');
      end
      obj.fSelectedEids = zeros(0,1);
      assert(isa(expCoordinator,'OlyDat.ExperimentCoordinator'));
      obj.fExpCoordinator = expCoordinator;
      
      %%% Configure axis
      
      % Nothing else in the axes can be clicked on
      chil = findobj(ax,'HitTest','on');
      chil = setdiff(chil,ax);
      set(chil,'HitTest','off');
      
      % axis deleteFcn needs to clean up this object
      set(ax,'DeleteFcn',@(src,evt)obj.axisDeleted(src));
      
      %%% Configure toolbar
      obj.fBoundingBoxToggleToolH = [];
      if tfAddToolbarTool
        % TODO: For now the XYPlotClickSender directly adds a
        % bounding-box toggle tool to the toolbar of ax's parent
        % figure. This is cleaned up on axis delete. This is a
        % jury-rig for now until the usage is more clear.
        hFig = ancestor(ax,'figure');
        tlbar = findobj(hFig,'Type','uitoolbar');
        if ~isempty(tlbar)
          [X map] = imread(fullfile(matlabroot,'toolbox','matlab','icons','tool_rectangle.gif'));
          icon = ind2rgb(X,map);
          
          % Add bounding box tool to the first toolbar found
          obj.fBoundingBoxToggleToolH = uitoggletool(tlbar(1),...
            'CData',icon,...
            'OnCallback',@(src,evt)obj.setSelectionMode('boundingBox'),...
            'OffCallback',@(src,evt)obj.setSelectionMode('singleClick'));
        end
      end
      
      expCoordinator.registerAxis(ax,@(eids)obj.signalReceivedCallback(eids));
      
      obj.bbReset; % init bounding-box state
      obj.setSelectionMode('singleClick');
    end
    
  end
  
  % Overload this method in derived classes to respond to selections.
  methods
    
    function respondToEidSelection(obj,eids) %#ok<MANU,INUSD>
      % default implementation: none
    end
    
  end
  
  methods
    
    function setSelectionMode(obj,mode)
      obj.bbReset;
      hFig = ancestor(obj.fAx,'figure');
      
      % WindowButtonDownFcn, etc cannot be modified while zoom/pan
      % are on.
      zoom(hFig,'off');
      pan(hFig,'off');
      
      switch mode
        case 'none'
          % This option is really only used for cleanup. By
          % clearing out all axis/figure callbacks, there will no
          % longer be any references to obj from its axis/figure.
          % Under typical usage obj will get cleaned up.
          set(obj.fAx,'ButtonDownFcn',[]);
          
          set(hFig,'WindowButtonDownFcn',[]);
          set(hFig,'WindowButtonUpFcn',[]);
          set(hFig,'WindowButtonMotionFcn',[]);
        case 'singleClick'
          set(obj.fAx,'ButtonDownFcn',@obj.buttonDownFcn);
          
          set(hFig,'WindowButtonDownFcn',[]);
          set(hFig,'WindowButtonUpFcn',[]);
          set(hFig,'WindowButtonMotionFcn',[]);
        case 'boundingBox'
          set(hFig,'WindowButtonDownFcn',@obj.rubberBandBegin);
          set(hFig,'WindowButtonUpFcn',@obj.rubberBandEnd);
          set(hFig,'WindowButtonMotionFcn',@obj.rubberBandUpdate);
          
          set(obj.fAx,'ButtonDownFcn',[]);
          set(obj.fAx,'XLimMode','manual','YLimMode','manual');
        otherwise
          assert(false,'Unrecognized mode');
      end
      obj.fSelectionMode = mode;
    end
    
    function signalReceivedCallback(obj,eids)
      obj.fSelectedEids = eids;
      obj.respondToEidSelection(eids);
    end
    
    function axisDeleted(obj,ax)
      assert(ax==obj.fAx);
      obj.setSelectionMode('none');
      if ~isempty(obj.fBoundingBoxToggleToolH)
        if ishandle(obj.fBoundingBoxToggleToolH)
          delete(obj.fBoundingBoxToggleToolH);
        end
        obj.fBoundingBoxToggleToolH = [];
      end
      obj.fExpCoordinator.unregisterAxis(ax);
    end
    
  end
  
  % callbacks used in single-click selection mode
  methods
    
    function buttonDownFcn(obj,hax,evt)
      
      if evt.Button~=1
        return; % respond only to left-click; save right-click for context menus
      end
      
      % Get clicked point
      currentIdx = obj.getCurrentPointIdx(hax,obj.fData(:,1),obj.fData(:,2));
      if isempty(currentIdx)
        % no point selected (eg out of click range)
        return;
      end
      
      % Send signal
      eid = obj.fEid(currentIdx); % Note: if obj.fEidIsCellstr, this will be a cell array
      hFig = ancestor(hax,'figure');
      switch get(hFig,'SelectionType')
        case 'extend'
          % add new selection to existing
          if ismember(eid,obj.fSelectedEids)
            % point already selected; no-op
          else
            obj.fExpCoordinator.sendSignal(hax,[obj.fSelectedEids;eid]);
          end
        otherwise
          obj.fExpCoordinator.sendSignal(hax,eid);
      end
      
    end
    
  end
  
  % callbacks used in bounding box selection mode
  methods
    
    function bbReset(obj)
      delete(obj.fBoundingBoxPatchH);
      obj.fBoundingBoxPatchH = [];
      obj.fBoundingBoxCoords = nan(1,4);
      obj.fBoundingBoxOn = false;
      obj.fBoundingBoxShiftOn = false;
    end
    
    function [x y z] = bbGetCursorCoordOnAxes(obj)
      crd = get(obj.fAx,'CurrentPoint');
      x = crd(2,1);
      y = crd(2,2);
      z = crd(2,3);
    end
    
    function rubberBandBegin(obj,~,~)
      
      obj.bbReset();
      
      % Create rubber band patches
      obj.fBoundingBoxPatchH = patch('Parent',obj.fAx);
      set(obj.fBoundingBoxPatchH(1), ...
        'EdgeColor', 'k', ...
        'FaceColor', 'none', ...
        'LineWidth', 1.5, ...
        'LineStyle', '-',...
        'XData', [nan nan nan nan],...
        'YData', [nan nan nan nan]);
      %             set(obj.fBoundingBoxPatchH(2), ...
      %                 'EdgeColor', 'k', ...
      %                 'FaceColor', 'k', ...
      %                 'FaceAlpha', .1, ...
      %                 'LineWidth', 0.5, ...
      %                 'LineStyle', '-',...
      %                 'XData', [nan nan nan nan],...
      %                 'YData', [nan nan nan nan]);
      
      [x y] = obj.bbGetCursorCoordOnAxes();
      obj.fBoundingBoxCoords = [x y x y];
      obj.rubberBandSetPos();
      obj.fBoundingBoxOn = true;
      hFig = ancestor(obj.fAx,'figure');
      obj.fBoundingBoxShiftOn = strcmp(get(hFig,'SelectionType'),'extend');
    end
    
    function rubberBandEnd(obj,~,~)
      obj.fBoundingBoxOn = false;
      
      tfRowIdxs = obj.getPointsInBoundingBox(obj.fData(:,1),obj.fData(:,2),obj.fBoundingBoxCoords);
      newlyBoxedEids = obj.fEid(tfRowIdxs);
      newlyBoxedEids = unique(newlyBoxedEids); % for plots with multiple points per eid, could select the same eid twice
      
      if obj.fBoundingBoxShiftOn
        selectedEids = [obj.fSelectedEids;setdiff(newlyBoxedEids,obj.fSelectedEids)];
      else
        selectedEids = newlyBoxedEids;
      end
      obj.fExpCoordinator.sendSignal(obj.fAx,selectedEids);
    end
    
    function rubberBandUpdate(obj,~,~)
      if obj.fBoundingBoxOn
        [x y] = obj.bbGetCursorCoordOnAxes();
        obj.fBoundingBoxCoords(3) = x;
        obj.fBoundingBoxCoords(4) = y;
        obj.rubberBandSetPos();
      end
    end
    
    function rubberBandSetPos(obj,~,~)
      set(obj.fBoundingBoxPatchH, ...
        'XData',obj.fBoundingBoxCoords([1 3 3 1]),...
        'YData',obj.fBoundingBoxCoords([2 2 4 4]));
    end
    
  end
  
  methods (Static)
    
    function rowIdx = getCurrentPointIdx(hax,x,y)
      
      tmp = get(hax,'CurrentPoint');
      xclicked = tmp(1,1);
      yclicked = tmp(1,2);
      
      % scale so that x and y distances are the same
      xlim = get(hax,'XLim');
      ylim = get(hax,'YLim');
      dx = xlim(2)-xlim(1);
      dy = ylim(2)-ylim(1);
      
      % which point is this closest to
      [dist rowIdx] = min( ((x(:)-xclicked)/dx).^2+((y(:)-yclicked)/dy).^2 );
      if dist > OlyDat.XYPlotClickSender.maxDistFrac
        rowIdx = [];
        return;
      end
    end
    
    % x: vector of x coords
    % y: vector of y coords
    % bboxCoords: [x1 y2 x2 y2]
    % rowIdxs: logical row index into x,y
    function rowIdxs = getPointsInBoundingBox(x,y,bboxCoords)
      upperX = max(bboxCoords([1 3]));
      lowerX = min(bboxCoords([1 3]));
      upperY = max(bboxCoords([2 4]));
      lowerY = min(bboxCoords([2 4]));
      rowIdxs = x(:) > lowerX & x(:) < upperX & ...
        y(:) > lowerY & y(:) < upperY;
    end
    
  end
  
end