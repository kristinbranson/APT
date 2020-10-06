classdef keypoints_exported < matlab.apps.AppBase
  
  % Properties that correspond to app components
  properties (Access = public)
    KeypointsUIFigure  matlab.ui.Figure
    GridLayout         matlab.ui.container.GridLayout
    LeftPanel          matlab.ui.container.Panel
    TabGroup           matlab.ui.container.TabGroup
    SkeletonTab        matlab.ui.container.Tab
    UIAxes             matlab.ui.control.UIAxes
    AddEdgeButton      matlab.ui.control.Button
    RemoveEdgeButton   matlab.ui.control.Button
    HeadTailTab        matlab.ui.container.Tab
    UIAxes_ht           matlab.ui.control.UIAxes
    SpecifyHeadButton  matlab.ui.control.Button
    ClearButton        matlab.ui.control.Button
    SwapPairsTab       matlab.ui.container.Tab
    UIAxes_sp           matlab.ui.control.UIAxes
    AddPairButton      matlab.ui.control.Button
    RemovePairButton   matlab.ui.control.Button
    RightPanel         matlab.ui.container.Panel
    UITable            matlab.ui.control.Table
  end
  
  % Properties that correspond to apps with auto-reflow
  properties (Access = private)
    onePanelWidth = 576;
  end
  
  properties (Access = private)
    lObj
    pts
    
    % Skeleton state
    sklEdges
    sklHpts
    sklHEdgeSelected
    sklHEdges
    sklHIm
    sklHTxt
    sklISelected
    
    % Head/Tail state
    htHead
    htHpts
    htHIm
    htHTxt
    htISelected
    
    % swap state
    spEdges
    spHpts
    spHEdgeSelected
    spHEdges
    spHIm
    spHTxt
    spISelected
    
    % Applies to all tabs
    selectedColor
    selectedMarker
    selectedMarkerSize
    selectedLineWidth
    unselectedColor
    unselectedMarker
    unselectedMarkerSize
    unselectedLineWidth
  end
  
  methods (Access = private, Static)  
    function s = parseLabelerState(lObj)
      freezeInfo = lObj.prevAxesModeInfo;
      imagescArgs = {'XData',freezeInfo.xdata,'YData',freezeInfo.ydata};
      im = lObj.prevAxesModeInfo.im;
      axcProps = freezeInfo.axes_curr;
      axesProps = {};
      for prop=fieldnames(axcProps)',
        axesProps(end+1:end+2) = {prop{1},axcProps.(prop{1})};
      end
      if freezeInfo.isrotated,
        axesProps(end+1:end+2) = {'CameraUpVectorMode','auto'};
      end
      isrotated = freezeInfo.isrotated;
      
      iMov = freezeInfo.iMov;
      frm = freezeInfo.frm;
      iTgt = freezeInfo.iTgt;
      
      lpos = lObj.labelsGTaware;
      s = lpos{iMov};
      [~,p,~] = Labels.isLabeledFT(s,frm,iTgt);
      pts = reshape(p,numel(p)/2,2);
      %pts = lpos{iMov}(:,:,frm,iTgt);
      if isrotated,
        pts = [pts,ones(size(pts,1),1)]*freezeInfo.A;
        pts = pts(:,1:2);
      end
      labelCM = lObj.LabelPointColors;            
      if isempty(labelCM)
        labelCM = jet(size(pts,1));
      end
      txtOffset = lObj.labelPointsPlotInfo.TextOffset;
      
      s = struct();
      s.skelEdges = lObj.skeletonEdges;
      if isempty(s.skelEdges)
        s.skelEdges = zeros(0,2);
      end
      s.swaps = lObj.flipLandmarkMatches;
      if isempty(s.swaps)
        s.swaps = zeros(0,2);
      end
      s.head = lObj.skelHead;
      s.skelNames = lObj.skelNames;
      s.im = im;
      s.pts = pts;
      s.imagescArgs = imagescArgs;
      s.labelCM = labelCM;
      s.axesProps = axesProps;
      s.txtOffset = txtOffset;
    end
    function hIm = initImage(hAx,slbl)
      set(hAx,slbl.axesProps{:});
      hold(hAx,'off');
      hIm = imagesc(slbl.im,'Parent',hAx,slbl.imagescArgs{:});
      hold(hAx,'on');
      colormap(hAx,'gray');
      axis(hAx,'off');
    end    
  end
  methods (Access=private) %bdf cbks
    function edgeClicked(app,h,e)
      if h.Parent==app.UIAxes
        app.edgeClickedSkel(h,e);
      elseif h.Parent==app.UIAxes_sp
        app.edgeClickedSwap(h,e);
      end
    end
    function edgeClickedSkel(app,h,e)
      edge = get(h,'UserData');
      
      sklISelected = app.sklISelected;
      sklHpts = app.sklHpts;
      sklHEdgeSelected = app.sklHEdgeSelected;
      
      if ~isempty(sklISelected),
        set(sklHpts(sklISelected),'Marker',app.unselectedMarker,'MarkerSize',app.unselectedMarkerSize);
        set(app.sklHEdgeSelected,'XData',nan(2,1),'YData',nan(2,1));
      end
      if numel(sklISelected==2) && all(edge == sort(sklISelected)),
        sklISelected = [];
      else
        sklISelected = edge;
        set(sklHpts(sklISelected),'Marker',app.selectedMarker,'MarkerSize',app.selectedMarkerSize);
        set(sklHEdgeSelected,'XData',app.pts(sklISelected,1),'YData',app.pts(sklISelected,2));
        isSelected = ismember(sort(sklISelected),app.sklEdges,'rows');
        if isSelected,
          set(sklHEdgeSelected,'Color',app.selectedColor);
        else
          % this should never happen
          set(sklHEdgeSelected,'Color',app.unselectedColor);
        end
      end
      
      app.sklISelected = sklISelected;
    end
    
    function ptClicked(app,h,e)
      if h.Parent==app.UIAxes
        app.ptClickedSkel(h,e,'sklISelected','sklHpts','sklHEdgeSelected','sklEdges');
      elseif h.Parent==app.UIAxes_ht
        app.ptClickedHT(h,e);
      elseif h.Parent==app.UIAxes_sp
        app.ptClickedSkel(h,e,'spISelected','spHpts','spHEdgeSelected','spEdges');
      end
    end
    function ptClickedSkel(app,h,e,fldISel,fldHPts,fldHEdgeSel,fldedges)
      iClicked = get(h,'UserData');
      
      iSeld = app.(fldISel); %#OK
      hpts = app.(fldHPts);
      ptsxy = app.pts;
      hEdgeSeld = app.(fldHEdgeSel);
      
      j = find(iSeld==iClicked);
      if isempty(j),
        iSelect = iClicked;
        iUnselect = [];
        iSeld(end+1) = iClicked;
        if numel(iSeld) > 2,
          iUnselect = iSeld(1);
          iSeld = iSeld(end-1:end);
        end
      else
        iSeld(j) = [];
        iSelect = [];
        iUnselect = iClicked;
      end
      if ~isempty(iSelect),
        set(hpts(iSelect),'Marker',app.selectedMarker,...
          'MarkerSize',app.selectedMarkerSize);
      end
      if ~isempty(iUnselect),
        set(hpts(iUnselect),'Marker',app.unselectedMarker,...
          'MarkerSize',app.unselectedMarkerSize);
      end
      if numel(iSeld) >= 2,
        isSelected = ismember(sort(iSeld),app.(fldedges),'rows');
        set(hEdgeSeld,'XData',ptsxy(iSeld,1),'YData',ptsxy(iSeld,2));
        if isSelected,
          set(hEdgeSeld,'Color',app.selectedColor);
        else
          set(hEdgeSeld,'Color',app.unselectedColor);
        end
      else
        set(hEdgeSeld,'XData',nan(2,1),'YData',nan(2,1));
      end
      
      app.(fldISel) = iSeld;
    end
    function ptClickedHT(app,h,e)
      iClicked = get(h,'UserData');
      
      iSeld = app.htISelected; %#OK
      hpts = app.htHpts;
      
      if isempty(iSeld)
        iSelect = iClicked;
        iUnselect = [];
      elseif iSeld==iClicked
        iSelect = [];
        iUnselect = iSeld;
      else
        iSelect = iClicked;
        iUnselect = iSeld;
      end
      if ~isempty(iSelect),
        set(hpts(iSelect),'Marker',app.selectedMarker,'MarkerSize',app.selectedMarkerSize);
      end
      if ~isempty(iUnselect),
        set(hpts(iUnselect),'Marker',app.unselectedMarker,'MarkerSize',app.unselectedMarkerSize);
      end
      app.htISelected = iSelect;
    end
  end
  methods (Access=private)
    function updateTable(app,head)
      % Updates table state assuming app.SpecfyHeadLandmarkSwitch is set
      % head: either [] if SpecifyHead is off, otherwise head pt
      
      ht = app.UITable;
      
      npts = size(app.pts,1);
      names = arrayfun(@(x)sprintf('pt%d',x),(1:npts)','uni',0);
      
%       tfheadtail = strcmp(app.SpecfyHeadLandmarkSwitch.Value,'On');
%       if tfheadtail
%         hvec = false(npts,1);
%         hvec(head) = true;
%         dat = [names(:) num2cell(hvec)];
%         colhdr = {'Name' 'Head'};
%       else
        dat = names(:);
        colhdr = { 'Name'};
        colw = {'auto'};
        coledit = [true];
%       end
      ncol = size(dat,2);
      
      ht.RowName = 'numbered';
      ht.Data = dat;
      ht.ColumnName = colhdr;
      ht.ColumnEditable = coledit;
      ht.ColumnWidth = colw;
    end
    function [htEnabled,ptNames,ht] = getTableState(app)
      %htEnabled = strcmp(app.SpecfyHeadLandmarkSwitch.Value,'On');
      htEnabled = true;
      ht = app.UITable;
      dat = ht.Data;
      ptNames = dat(:,1);
      if htEnabled
        ht = cell2mat(dat(:,2));
        ht = find(ht);
      else
        ht = [];
      end
    end
    function moveEdgesToBack(app)
      
      hAx = app.UIAxes;
      hchil = get(hAx,'Children');
      hchil1 = [app.sklHIm;app.sklHTxt(:);app.sklHEdges(:);app.sklHEdgeSelected;app.sklHpts(:)];
      hleft = hchil;
      hleft(ismember(hleft,hchil1)) = [];
      set(hAx,'Children',flipud([hleft;hchil1]));
    end
  end
  
  % Callbacks that handle component events
  methods (Access = private)
    
    % Code that executes after component creation
    function startupFcn(app, varargin)
      [lblObj,edges,plotptsArgs,textArgs,...
        txtOffset,selMarkerSize,unselMarkerSize,...
        selMarker,unselMarker,...
        unselColor,selColor,...
        unselLineWidth,selLineWidth] = myparse(varargin,...
        'lObj',[],...
        'edges',[], ...
        'plotptsArgs',{},...
        'textArgs',{}, ...
        'txtOffset',2,...
        'selectedMarkerSize',12,...
        'unselectedMarkerSize',8,...
        'selectedMarker','o','unselectedMarker','+',...
        'unselectedColor',[.5,.5,.5],'selectedColor',[.8,0,.8],...
        'unselectedLineWidth',2,'selectedLineWidth',4 ...
        );
      
      if ~isempty(lblObj)
        centerOnParentFigure(app.KeypointsUIFigure,lblObj.hFig,'setParentFixUnitsPx',true);
      else
        centerfig(app.KeypointsUIFigure);
      end
      
      app.lObj = lblObj;
      
      app.selectedColor = selColor;
      app.selectedMarker = selMarker;
      app.selectedMarkerSize = selMarkerSize;
      app.selectedLineWidth = selLineWidth;
      app.unselectedColor = unselColor;
      app.unselectedMarker = unselMarker;
      app.unselectedMarkerSize = unselMarkerSize;
      app.unselectedLineWidth = unselLineWidth;

      slbl = app.parseLabelerState(app.lObj);
      app.pts = slbl.pts;

      % align axes
      app.UIAxes_ht.Position = app.UIAxes.Position;
      app.UIAxes_sp.Position = app.UIAxes.Position;
      
      app.sklEdges = slbl.skelEdges;
      app.initTabSkel(slbl,textArgs,plotptsArgs);
      app.htHead = slbl.head;
      app.initTabHeadTail(slbl,textArgs,plotptsArgs);
      app.spEdges = slbl.swaps;
      app.initTabSwap(slbl,textArgs,plotptsArgs);
      
      %app.updateTable(head);
    end
    
    function [hpts,htxt] = initPtsAx(app,hAx,slbl,plotptsArgs,textArgs)
      unselMarker = app.unselectedMarker;
      unselMarkerSize = app.unselectedMarkerSize;
      
      npts = size(app.pts,1);
      ptnames = slbl.skelNames;

      hpts = gobjects(npts,1);
      htxt = gobjects(npts,1);
      for i = 1:npts
        hpts(i) = plot(hAx,slbl.pts(i,1),slbl.pts(i,2),unselMarker,...
          'Color',slbl.labelCM(i,:),'MarkerFaceColor',slbl.labelCM(i,:),...
          'UserData',i,'MarkerSize',unselMarkerSize,plotptsArgs{:});
        set(hpts(i),'ButtonDownFcn',@(h,e)app.ptClicked(h,e));
        htxt(i) = text(hAx,app.pts(i,1)+slbl.txtOffset,app.pts(i,2)+slbl.txtOffset,...
          ptnames{i},'Color',slbl.labelCM(i,:),'PickableParts','none',textArgs{:});
      end
   
    end
    function [hEdges,hEdgeSel] = initEdgesAx(app,hAx,slbl,fldEdge)
      
      selColor = app.selectedColor;
      %       app.selectedMarker = selMarker;
      %       app.selectedMarkerSize = selMarkerSize;
      selLineWidth = app.selectedLineWidth;
      unselColor = app.unselectedColor;
      unselLineWidth = app.unselectedLineWidth;
      
      edges = slbl.(fldEdge);
      nedge = size(edges,1);
      hEdges = gobjects(1,nedge);
      for i = 1:nedge
        hEdges(i) = plot(hAx,slbl.pts(edges(i,:),1),slbl.pts(edges(i,:),2),'-',...
          'Color',selColor,'LineWidth',unselLineWidth,...
          'ButtonDownFcn',@(h,e)app.edgeClicked(h,e),...
          'UserData',edges(i,:),'Tag',sprintf('edge%d',i));
      end
      
      hEdgeSel = plot(hAx,nan(2,1),nan(2,1),'-','Color',unselColor,...
        'LineWidth',selLineWidth,'PickableParts','none');
    end
    function initTabSkel(app,slbl,textArgs,plotptsArgs)
      hAx = app.UIAxes;
      
      hIm = app.initImage(hAx,slbl);
      [hpts,htxt] = app.initPtsAx(hAx,slbl,plotptsArgs,textArgs);
      [hEdges,hEdgeSel] = app.initEdgesAx(hAx,slbl,'skelEdges');

      app.sklHpts = hpts;
      app.sklHEdges = hEdges;
      app.sklHIm = hIm;
      app.sklHTxt = htxt;
      app.sklHEdgeSelected = hEdgeSel;
      app.sklISelected = [];
    end
    function initTabHeadTail(app,slbl,textArgs,plotptsArgs)
      hAx = app.UIAxes_ht;
      
      hIm = app.initImage(hAx,slbl);
      [hpts,htxt] = app.initPtsAx(hAx,slbl,plotptsArgs,textArgs);
      if ~isempty(slbl.head)
        set(hpts(slbl.head),...
          'Marker',app.selectedMarker,...
          'MarkerSize',app.selectedMarkerSize)
      end

      app.htHpts = hpts;
      app.htHIm = hIm;
      app.htHTxt = htxt;
      app.htISelected = [];
    end
    function initTabSwap(app,slbl,textArgs,plotptsArgs)
      hAx = app.UIAxes_sp;
      
      hIm = app.initImage(hAx,slbl);
      [hpts,htxt] = app.initPtsAx(hAx,slbl,plotptsArgs,textArgs);
      [hEdges,hEdgeSel] = app.initEdgesAx(hAx,slbl,'swaps');

      app.spHpts = hpts;
      app.spHEdges = hEdges;
      app.spHIm = hIm;
      app.spHTxt = htxt;
      app.spHEdgeSelected = hEdgeSel;
      app.spISelected = [];
    end
    
    % Changes arrangement of the app based on UIFigure width
    function updateAppLayout(app, event)
      currentFigureWidth = app.KeypointsUIFigure.Position(3);
      if(currentFigureWidth <= app.onePanelWidth)
        % Change to a 2x1 grid
        app.GridLayout.RowHeight = {392, 392};
        app.GridLayout.ColumnWidth = {'1x'};
        app.RightPanel.Layout.Row = 2;
        app.RightPanel.Layout.Column = 1;
      else
        % Change to a 1x2 grid
        app.GridLayout.RowHeight = {'1x'};
        app.GridLayout.ColumnWidth = {448, '1x'};
        app.RightPanel.Layout.Row = 1;
        app.RightPanel.Layout.Column = 2;
      end
    end
    
    % Button pushed function: AddEdgeButton
    function AddEdgeButtonPushed(app, event)
      sklISelected = app.sklISelected;
      edges = app.sklEdges;
      sklHEdges = app.sklHEdges;
      pts = app.pts;
      
      if numel(sklISelected) < 2,
        fprintf('No edge selected. Edge must be selected to add.\n');
        return;
      end
      edge = sort(sklISelected);
      if ismember(edge,edges,'rows'),
        fprintf('Edge already selected.\n');
        return;
      end
      edges(end+1,:) = edge;
      sklHEdges(end+1) = plot(app.UIAxes,pts(edge,1),pts(edge,2),'-','Color',app.selectedColor,'LineWidth',app.unselectedLineWidth,...
        'ButtonDownFcn',@(h,e)app.edgeClicked(h,e),'UserData',edge,'Tag',sprintf('edge%d',numel(sklHEdges)+1));
      set(app.sklHEdgeSelected,'Color',app.selectedColor);
      
      app.sklEdges = edges;
      app.sklHEdges = sklHEdges;
      
      app.moveEdgesToBack();
    end
    
    % Button pushed function: RemoveEdgeButton
    function RemoveEdgeButtonPushed(app, event)
      sklISelected = app.sklISelected;
      edges = app.sklEdges;
      sklHEdges = app.sklHEdges;
      
      if numel(sklISelected) < 2,
        fprintf('No edge selected. Edge must be selected to add.\n');
        return;
      end
      edge = sort(sklISelected);
      [~,j] = ismember(edge,edges,'rows');
      if j == 0,
        fprintf('Edge is not included in skeleton, cannot remove.\n');
        return;
      end
      edges(j,:) = [];
      delete(sklHEdges(j));
      sklHEdges = [sklHEdges(1:j-1),sklHEdges(j+1:end)];
      
      set(app.sklHEdgeSelected,'Color',app.unselectedColor);
      
      app.sklEdges = edges;
      app.sklHEdges = sklHEdges;
    end
    
    % Callback function
    function SpecfyHeadLandmarkSwitchValueChanged(app, event)
      app.updateTable(1);
    end
    
    % Cell edit callback: UITable
    function UITableCellEdit(app, event)
      indices = event.Indices;
      HEADTAILCOL = 2;
      if indices(2)==HEADTAILCOL && ~event.PreviousData && event.NewData
        ht = app.UITable;
        dat = ht.Data;
        htcol = cell2mat(dat(:,HEADTAILCOL));
        if nnz(htcol)>1
          htcol(:) = 0;
          htcol(indices(1)) = 1;
          ht.Data(:,HEADTAILCOL) = num2cell(htcol);
        end
      end
    end
    
    % Close request function: KeypointsUIFigure
    function KeypointsUIFigureCloseRequest(app, event)
      
      app.lObj.setSkeletonEdges(app.sklEdges);
      [htEnabled,ptNames,ht] = app.getTableState();
      if htEnabled
        assert(isscalar(ht))
        %                 if nnz(ht)>1
        %                     warningNoTrace('Using only first head point specification: %d',ht(1));
        %                     ht = ht(1);
        %                 end
        app.lObj.setSkelHead(ht);
      else
        app.lObj.setSkelHead([]);
      end
      app.lObj.setSkelNames(ptNames);
      %app.lObj.maSetPtInfo(htEnabled,ptNames,ht);
      delete(app);
      
    end
  end
  
  % Component initialization
  methods (Access = private)
    
    % Create UIFigure and components
    function createComponents(app)
      
      % Create KeypointsUIFigure and hide until all components are created
      app.KeypointsUIFigure = uifigure('Visible', 'off');
      app.KeypointsUIFigure.AutoResizeChildren = 'off';
      app.KeypointsUIFigure.Position = [100 100 686 392];
      app.KeypointsUIFigure.Name = 'Keypoints';
      app.KeypointsUIFigure.CloseRequestFcn = createCallbackFcn(app, @KeypointsUIFigureCloseRequest, true);
      app.KeypointsUIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);
      
      % Create GridLayout
      app.GridLayout = uigridlayout(app.KeypointsUIFigure);
      app.GridLayout.ColumnWidth = {448, '1x'};
      app.GridLayout.RowHeight = {'1x'};
      app.GridLayout.ColumnSpacing = 0;
      app.GridLayout.RowSpacing = 0;
      app.GridLayout.Padding = [0 0 0 0];
      app.GridLayout.Scrollable = 'on';
      
      % Create LeftPanel
      app.LeftPanel = uipanel(app.GridLayout);
      app.LeftPanel.Layout.Row = 1;
      app.LeftPanel.Layout.Column = 1;
      
      % Create TabGroup
      app.TabGroup = uitabgroup(app.LeftPanel);
      app.TabGroup.Position = [6 6 436 376];
      
      % Create SkeletonTab
      app.SkeletonTab = uitab(app.TabGroup);
      app.SkeletonTab.Title = 'Skeleton';
      
      % Create UIAxes
      app.UIAxes = uiaxes(app.SkeletonTab);
      title(app.UIAxes, '')
      xlabel(app.UIAxes, '')
      ylabel(app.UIAxes, '')
      app.UIAxes.XTick = [];
      app.UIAxes.YTick = [];
      app.UIAxes.Position = [15 53 390 280];
      
      % Create AddEdgeButton
      app.AddEdgeButton = uibutton(app.SkeletonTab, 'push');
      app.AddEdgeButton.ButtonPushedFcn = createCallbackFcn(app, @AddEdgeButtonPushed, true);
      app.AddEdgeButton.Position = [118 17 100 23];
      app.AddEdgeButton.Text = 'Add Edge';
      
      % Create RemoveEdgeButton
      app.RemoveEdgeButton = uibutton(app.SkeletonTab, 'push');
      app.RemoveEdgeButton.ButtonPushedFcn = createCallbackFcn(app, @RemoveEdgeButtonPushed, true);
      app.RemoveEdgeButton.Position = [230 17 100 23];
      app.RemoveEdgeButton.Text = 'Remove Edge';
      
      % Create HeadTailTab
      app.HeadTailTab = uitab(app.TabGroup);
      app.HeadTailTab.Title = 'Head/Tail';
      
      % Create UIAxes_ht
      app.UIAxes_ht = uiaxes(app.HeadTailTab);
      title(app.UIAxes_ht, '')
      xlabel(app.UIAxes_ht, '')
      ylabel(app.UIAxes_ht, '')
      app.UIAxes_ht.XTick = [];
      app.UIAxes_ht.YTick = [];
      app.UIAxes_ht.Position = [23 30 390 291];
      
      % Create SpecifyHeadButton
      app.SpecifyHeadButton = uibutton(app.HeadTailTab, 'push');
      app.SpecifyHeadButton.Position = [118 15 100 23];
      app.SpecifyHeadButton.Text = 'Specify Head';
      
      % Create ClearButton
      app.ClearButton = uibutton(app.HeadTailTab, 'push');
      app.ClearButton.Position = [230 15 100 23];
      app.ClearButton.Text = 'Clear';
      
      % Create SwapPairsTab
      app.SwapPairsTab = uitab(app.TabGroup);
      app.SwapPairsTab.Title = 'Swap Pairs';
      
      % Create UIAxes_sp
      app.UIAxes_sp = uiaxes(app.SwapPairsTab);
      title(app.UIAxes_sp, '')
      xlabel(app.UIAxes_sp, '')
      ylabel(app.UIAxes_sp, '')
      app.UIAxes_sp.XTick = [];
      app.UIAxes_sp.YTick = [];
      app.UIAxes_sp.Position = [23 38 390 291];
      
      % Create AddPairButton
      app.AddPairButton = uibutton(app.SwapPairsTab, 'push');
      app.AddPairButton.Position = [118 17 100 23];
      app.AddPairButton.Text = 'Add Pair';
      
      % Create RemovePairButton
      app.RemovePairButton = uibutton(app.SwapPairsTab, 'push');
      app.RemovePairButton.Position = [230 17 100 23];
      app.RemovePairButton.Text = 'Remove Pair';
      
      % Create RightPanel
      app.RightPanel = uipanel(app.GridLayout);
      app.RightPanel.Layout.Row = 1;
      app.RightPanel.Layout.Column = 2;
      
      % Create UITable
      app.UITable = uitable(app.RightPanel);
      app.UITable.ColumnName = {'#'; 'Name'; 'Head'};
      app.UITable.ColumnWidth = {30, 'auto', 'auto'};
      app.UITable.RowName = {};
      app.UITable.ColumnEditable = [false true true];
      app.UITable.CellEditCallback = createCallbackFcn(app, @UITableCellEdit, true);
      app.UITable.Position = [21 133 195 233];
      
      % Show the figure after all components are created
      app.KeypointsUIFigure.Visible = 'on';
    end
  end
  
  % App creation and deletion
  methods (Access = public)
    
    % Construct app
    function app = keypoints_exported(varargin)
      
      % Create UIFigure and components
      createComponents(app)
      
      % Register the app with App Designer
      registerApp(app, app.KeypointsUIFigure)
      
      % Execute the startup function
      runStartupFcn(app, @(app)startupFcn(app, varargin{:}))
      
      if nargout == 0
        clear app
      end
    end
    
    % Code that executes before app deletion
    function delete(app)
      
      % Delete UIFigure when app is deleted
      delete(app.KeypointsUIFigure)
    end
  end
end