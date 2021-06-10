classdef landmark_specs < handle

  % Properties that correspond to app components
  properties (Access = public)
    hFig
    axSkel
    axHT
    axSwap
    
%     GridLayout         matlab.ui.container.GridLayout
    LeftPanel          %matlab.ui.container.Panel
    TabGroup           %matlab.ui.container.TabGroup
    SkeletonTab        %matlab.ui.container.Tab
    AddEdgeButton      %matlab.ui.control.Button
    RemoveEdgeButton   %matlab.ui.control.Button
    HeadTailTab        %matlab.ui.container.Tab
    SpecifyHeadButton  %matlab.ui.control.Button
    ClearButton        %matlab.ui.control.Button
    SwapPairsTab       %matlab.ui.container.Tab
    AddPairButton      %matlab.ui.control.Button
    RemovePairButton   %matlab.ui.control.Button
    RightPanel         %atlab.ui.container.Panel
    UITable            %matlab.ui.control.Table
  end 
  
  properties (Access = private)
    lObj
    pts % [npt x 2] xy coords for viz purposes
    ptNames % [npt] cellstr; working model/data for UI, to be written to lObj
    
    anyChangeMade = false;
    
    % Skeleton state
    sklEdges % [nedge x 2] working model/data for UI, to be written to lObj
    sklHpts
    sklHEdgeSelected
    sklHEdges
    sklHIm
    sklHTxt
    sklISelected
    
    % Head/Tail state
    htHead % [0 or 1] working model/data for UI, to be written to lObj
    htHpts
    htHIm
    htHTxt
    htISelected
    
    % swap state
    spEdges % [nswap x 2] working model/data for UI, to be written to lObj
    spHpts
    spHEdgeSelected
    spHEdges
    spHIm
    spHTxt
    spISelected
    
    % Cosmetics; applies to all tabs
    selectedColor
    selectedMarker
    selectedMarkerSize
    selectedLineWidth
    %markerLineWidth = 2;
    unselectedColor
    unselectedMarker
    unselectedMarkerSize
    unselectedLineWidth
  end
  
  methods (Access = public)
    
    function obj = landmark_specs(varargin)
      obj.createComponents();
      obj.startupFcn(varargin{:});
%       if nargout == 0
%         clear app
%       end
    end
    
    function delete(obj)
    end
  end
  
  %#OK
  methods (Static)
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
      if isempty(s.skelNames)
        s.skelNames = arrayfun(@(x)sprintf('pt%d',x),(1:size(pts,1))','uni',0);
      end
      s.im = im;
      s.pts = pts;
      s.imagescArgs = imagescArgs;
      s.labelCM = labelCM;
      s.axesProps = axesProps;
      s.txtOffset = txtOffset;
    end
    function hIm = initImage(hAx,slbl)
      hold(hAx,'off');
      hIm = imagesc(slbl.im,'Parent',hAx,slbl.imagescArgs{:});
      hold(hAx,'on');
      set(hAx,slbl.axesProps{:});
      colormap(hAx,'gray');
      axis(hAx,'off');
    end    
  end
  methods (Access=private) % cbks
    function edgeClicked(app,h,e)
      if h.Parent==app.axSkel
        app.edgeClickedSkel(h,e);
      elseif h.Parent==app.axSwap
        app.edgeClickedSwap(h,e);
      end
    end
    function edgeClickedSkel(app,h,e)
      edge = get(h,'UserData');
      
      iSeld = app.sklISelected;
      hpts = app.sklHpts;
      hEdgeSeld = app.sklHEdgeSelected;
      
      if ~isempty(iSeld),
        set(hpts(iSeld),'Marker',app.unselectedMarker,'MarkerSize',app.unselectedMarkerSize);
        set(app.sklHEdgeSelected,'XData',nan(2,1),'YData',nan(2,1));
      end
      if numel(iSeld)==2 && all(edge == sort(iSeld)),
        iSeld = [];
      else
        iSeld = edge;
        set(hpts(iSeld),'Marker',app.selectedMarker,'MarkerSize',app.selectedMarkerSize);
        set(hEdgeSeld,'XData',app.pts(iSeld,1),'YData',app.pts(iSeld,2));
        isSelected = ismember(sort(iSeld),app.sklEdges,'rows');
        if isSelected,
          set(hEdgeSeld,'Color',app.selectedColor);
        else
          % this should never happen
          set(hEdgeSeld,'Color',app.unselectedColor);
        end
      end
      
      app.sklISelected = iSeld;
    end
    function edgeClickedSwap(app,h,e)
      edge = get(h,'UserData');
      
      iSeld = app.spISelected;
      hpts = app.spHpts;
      hEdgeSeld = app.spHEdgeSelected;
      
      if ~isempty(iSeld),
        set(hpts(iSeld),'Marker',app.unselectedMarker,'MarkerSize',app.unselectedMarkerSize);
        set(app.spHEdgeSelected,'XData',nan(2,1),'YData',nan(2,1));
      end
      if numel(iSeld)==2 && all(edge == sort(iSeld)),
        iSeld = [];
      else
        iSeld = edge;
        set(hpts(iSeld),'Marker',app.selectedMarker,'MarkerSize',app.selectedMarkerSize);
        set(hEdgeSeld,'XData',app.pts(iSeld,1),'YData',app.pts(iSeld,2));
        isSelected = ismember(sort(iSeld),app.spEdges,'rows');
        if isSelected,
          set(hEdgeSeld,'Color',app.selectedColor);
        else
          % this should never happen
          set(hEdgeSeld,'Color',app.unselectedColor);
        end
      end
      
      app.spISelected = iSeld;
    end
    
    function ptClicked(app,h,e)
      if h.Parent==app.axSkel
        app.ptClickedSkel(h,e,'sklISelected','sklHpts','sklHEdgeSelected','sklEdges');
      elseif h.Parent==app.axHT
        app.ptClickedHT(h,e);
      elseif h.Parent==app.axSwap
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
    %#OK
    function cbkTabGroupSelChanged(obj,e)
      tab = e.NewValue;
      if tab==obj.SkeletonTab
        obj.updateTableSkel();
      elseif tab==obj.HeadTailTab
        obj.updateTableHT();
      elseif tab==obj.SwapPairsTab
        obj.updateTableSwap();
      end
    end
    function updateTableSkel(app)
      ht = app.UITable;      
      ht.RowName = 'numbered';
      ht.Data = app.ptNames(:);
      ht.ColumnName = {'Name'};
      ht.ColumnEditable = true;
      ht.ColumnWidth = {'auto'};
    end
    function updateTableHT(app)
      npts = size(app.pts,1);
      hvec = false(npts,1);
      hvec(app.htHead) = true;
      
      ht = app.UITable;      
      ht.RowName = 'numbered';
      ht.Data = [app.ptNames(:) num2cell(hvec)];
      ht.ColumnName = {'Name' 'Head'};
      ht.ColumnEditable = [true false];
      ht.ColumnWidth = {'auto' 'auto'};
    end
    function updateTableSwap(app)
      partners = repmat({'none'},size(app.ptNames(:)));
      swaps = app.spEdges;
      nswap = size(swaps,1);
      for iswap=1:nswap
        s0 = swaps(iswap,1);
        s1 = swaps(iswap,2);
        partners{s0} = app.ptNames{s1};
        partners{s1} = app.ptNames{s0};
      end
      
      ht = app.UITable;
      ht.RowName = 'numbered';
      ht.Data = [app.ptNames(:) partners];
      ht.ColumnName = {'Name' 'Partner'};
      ht.ColumnEditable = [true false];
      ht.ColumnWidth = {'auto' 'auto'};      
    end    
    function moveEdgesToBack(app,axfld,fldhim,fldhtxt,fldhedges, ...
        fldhedgesel,fldhpts)
      hAx = app.(axfld);
      hchil = get(hAx,'Children');
      hchil1 = [app.(fldhim);app.(fldhtxt)(:);app.(fldhedges)(:);...
                app.(fldhedgesel);app.(fldhpts)(:)];
      hleft = hchil;
      hleft(ismember(hleft,hchil1)) = [];
      set(hAx,'Children',flipud([hleft;hchil1]));
    end
  end
  
  methods 
    
    %#OK
    function startupFcn(obj, varargin)
      [lblObj,edges,plotptsArgs,textArgs,...
        txtOffset,selMarkerSize,unselMarkerSize,...
        selMarker,unselMarker,...
        unselColor,selColor,...
        unselLineWidth,selLineWidth] = myparse(varargin,...
        'lObj',[],...
        'edges',[], ...
        'plotptsArgs',{'linewidth',2},...
        'textArgs',{'fontsize',16}, ...
        'txtOffset',1,...
        'selectedMarkerSize',12,...
        'unselectedMarkerSize',8,...
        'selectedMarker','o','unselectedMarker','+',...
        'unselectedColor',[.5,.5,.5],'selectedColor',[.8,0,.8],...
        'unselectedLineWidth',2,'selectedLineWidth',4 ...
        );
      
      if ~isempty(lblObj)
        centerOnParentFigure(obj.hFig,lblObj.hFig,'setParentFixUnitsPx',true);
      else
        centerfig(obj.hFig);
      end
      
      obj.lObj = lblObj;
      
      obj.selectedColor = selColor;
      obj.selectedMarker = selMarker;
      obj.selectedMarkerSize = selMarkerSize;
      obj.selectedLineWidth = selLineWidth;
      obj.unselectedColor = unselColor;
      obj.unselectedMarker = unselMarker;
      obj.unselectedMarkerSize = unselMarkerSize;
      obj.unselectedLineWidth = unselLineWidth;

      slbl = obj.parseLabelerState(obj.lObj);
      obj.pts = slbl.pts;
      obj.ptNames = slbl.skelNames;

      % align axes
      obj.axHT.Position = obj.axSkel.Position;
      obj.axSwap.Position = obj.axSkel.Position;
      
      obj.sklEdges = slbl.skelEdges;
      obj.initTabSkel(slbl,textArgs,plotptsArgs);
      obj.htHead = slbl.head;
      obj.initTabHeadTail(slbl,textArgs,plotptsArgs);
      obj.spEdges = slbl.swaps;
      obj.initTabSwap(slbl,textArgs,plotptsArgs);
      
      obj.updateTableSkel();
    end
    
    %#OK
    function [hpts,htxt] = initPtsAx(obj,hAx,slbl,plotptsArgs,textArgs)
      unselMarker = obj.unselectedMarker;
      unselMarkerSize = obj.unselectedMarkerSize;
      
      npts = size(obj.pts,1);
      ptnames = slbl.skelNames;

      hpts = gobjects(npts,1);
      htxt = gobjects(npts,1);
      for i = 1:npts
        hpts(i) = plot(hAx,slbl.pts(i,1),slbl.pts(i,2),unselMarker,...
          'Color',slbl.labelCM(i,:),'MarkerFaceColor',slbl.labelCM(i,:),...
          'UserData',i,'MarkerSize',unselMarkerSize,plotptsArgs{:});
        set(hpts(i),'ButtonDownFcn',@(h,e)obj.ptClicked(h,e));
        htxt(i) = text(hAx,obj.pts(i,1)+slbl.txtOffset,obj.pts(i,2)+slbl.txtOffset,...
          ptnames{i},'Color',slbl.labelCM(i,:),'PickableParts','none',textArgs{:});
      end   
    end
    %#OK
    function [hEdges,hEdgeSel] = initEdgesAx(obj,hAx,slbl,fldEdge)
      
      selColor = obj.selectedColor;
      %       app.selectedMarker = selMarker;
      %       app.selectedMarkerSize = selMarkerSize;
      selLineWidth = obj.selectedLineWidth;
      unselColor = obj.unselectedColor;
      unselLineWidth = obj.unselectedLineWidth;
      
      edges = slbl.(fldEdge);
      nedge = size(edges,1);
      hEdges = gobjects(1,nedge);
      for i = 1:nedge
        hEdges(i) = plot(hAx,slbl.pts(edges(i,:),1),slbl.pts(edges(i,:),2),'-',...
          'Color',selColor,'LineWidth',unselLineWidth,...
          'ButtonDownFcn',@(h,e)obj.edgeClicked(h,e),...
          'UserData',edges(i,:),'Tag',sprintf('edge%d',i));
      end
      
      hEdgeSel = plot(hAx,nan(2,1),nan(2,1),'-','Color',unselColor,...
        'LineWidth',selLineWidth,'PickableParts','none');
    end
    %#OK
    function initTabSkel(obj,slbl,textArgs,plotptsArgs)
      hAx = obj.axSkel;
      
      hIm = obj.initImage(hAx,slbl);
      [hpts,htxt] = obj.initPtsAx(hAx,slbl,plotptsArgs,textArgs);
      [hEdges,hEdgeSel] = obj.initEdgesAx(hAx,slbl,'skelEdges');

      obj.sklHpts = hpts;
      obj.sklHEdges = hEdges;
      obj.sklHIm = hIm;
      obj.sklHTxt = htxt;
      obj.sklHEdgeSelected = hEdgeSel;
      obj.sklISelected = [];
    end
    %#OK
    function initTabHeadTail(obj,slbl,textArgs,plotptsArgs)
      hAx = obj.axHT;
      
      hIm = obj.initImage(hAx,slbl);
      [hpts,htxt] = obj.initPtsAx(hAx,slbl,plotptsArgs,textArgs);
      if ~isempty(slbl.head)
        set(hpts(slbl.head),...
          'Marker',obj.selectedMarker,...
          'MarkerSize',obj.selectedMarkerSize)
      end

      obj.htHpts = hpts;
      obj.htHIm = hIm;
      obj.htHTxt = htxt;
      obj.htISelected = [];
    end
    %#OK
    function initTabSwap(obj,slbl,textArgs,plotptsArgs)
      hAx = obj.axSwap;
      
      hIm = obj.initImage(hAx,slbl);
      [hpts,htxt] = obj.initPtsAx(hAx,slbl,plotptsArgs,textArgs);
      [hEdges,hEdgeSel] = obj.initEdgesAx(hAx,slbl,'swaps');

      obj.spHpts = hpts;
      obj.spHEdges = hEdges;
      obj.spHIm = hIm;
      obj.spHTxt = htxt;
      obj.spHEdgeSelected = hEdgeSel;
      obj.spISelected = [];
    end
    
%     % Changes arrangement of the app based on UIFigure width
%     function updateAppLayout(obj, event)
%       currentFigureWidth = obj.hFig.Position(3);
%       if(currentFigureWidth <= obj.onePanelWidth)
%         % Change to a 2x1 grid
%         obj.GridLayout.RowHeight = {392, 392};
%         obj.GridLayout.ColumnWidth = {'1x'};
%         obj.RightPanel.Layout.Row = 2;
%         obj.RightPanel.Layout.Column = 1;
%       else
%         % Change to a 1x2 grid
%         obj.GridLayout.RowHeight = {'1x'};
%         obj.GridLayout.ColumnWidth = {448, '1x'};
%         obj.RightPanel.Layout.Row = 1;
%         obj.RightPanel.Layout.Column = 2;
%       end
%     end
    
    %#OK
    function AddEdgeButtonPushed(obj,event)
      iSeld = obj.sklISelected;
      edges = obj.sklEdges;
      hedges = obj.sklHEdges;
      ptsxy = obj.pts;
      
      if numel(iSeld) < 2,
        fprintf('No edge selected. Edge must be selected to add.\n');
        return;
      end
      edge = sort(iSeld);
      if ismember(edge,edges,'rows'),
        fprintf('Edge already selected.\n');
        return;
      end
      edges(end+1,:) = edge;
      hedges(end+1) = plot(obj.axSkel,ptsxy(edge,1),ptsxy(edge,2),'-','Color',obj.selectedColor,'LineWidth',obj.unselectedLineWidth,...
        'ButtonDownFcn',@(h,e)obj.edgeClicked(h,e),'UserData',edge,'Tag',sprintf('edge%d',numel(hedges)+1));
      set(obj.sklHEdgeSelected,'Color',obj.selectedColor);
      
      obj.sklEdges = edges;
      obj.sklHEdges = hedges;
      
      obj.moveEdgesToBack('axSkel','sklHIm','sklHTxt','sklHEdges',...
        'sklHEdgeSelected','sklHpts');
      
      obj.anyChangeMade = true;
    end  
    %#OK
    function RemoveEdgeButtonPushed(obj, event)
      iSeld = obj.sklISelected;
      edges = obj.sklEdges;
      hedges = obj.sklHEdges;
      
      if numel(iSeld) < 2,
        fprintf('No edge selected. Edge must be selected to add.\n');
        return;
      end
      edge = sort(iSeld);
      [~,j] = ismember(edge,edges,'rows');
      if j == 0,
        fprintf('Edge is not included in skeleton, cannot remove.\n');
        return;
      end
      edges(j,:) = [];
      delete(hedges(j));
      hedges = [hedges(1:j-1),hedges(j+1:end)];
      
      set(obj.sklHEdgeSelected,'Color',obj.unselectedColor);
      
      obj.sklEdges = edges;
      obj.sklHEdges = hedges;
      obj.anyChangeMade = true;
    end
    %#OK
    function AddPairButtonPushed(obj, event)
      iSeld = obj.spISelected;
      edges = obj.spEdges;
      hedges = obj.spHEdges;
      ptsxy = obj.pts;
      
      if numel(iSeld) < 2, 
        fprintf('No edge selected. Edge must be selected to add.\n');
        return;
      end
      if any(ismember(iSeld(:),edges(:)))
        error('A landmark can have at most one swap partner. Please remove any existing/conflicting swap pairs.');
      end
      edge = sort(iSeld);
      edges(end+1,:) = edge;
      hedges(end+1) = plot(obj.axSwap,ptsxy(edge,1),ptsxy(edge,2),'-',...
        'Color',obj.selectedColor,'LineWidth',obj.unselectedLineWidth,...
        'ButtonDownFcn',@(h,e)obj.edgeClicked(h,e),'UserData',edge,...
        'Tag',sprintf('edge%d',numel(hedges)+1));
      set(obj.spHEdgeSelected,'Color',obj.selectedColor);
      
      obj.spEdges = edges;
      obj.spHEdges = hedges;
      
      obj.moveEdgesToBack('axSkel','sklHIm','sklHTxt','sklHEdges',...
        'sklHEdgeSelected','sklHpts');
  
      obj.updateTableSwap();
      obj.anyChangeMade = true;
    end
    %#OK
    function RemovePairButtonPushed(obj, event)
      iSeld = obj.spISelected;
      edges = obj.spEdges;
      hedges = obj.spHEdges;
      
      if numel(iSeld) < 2,
        fprintf('No edge selected. Edge must be selected to remove.\n');
        return;
      end
      edge = sort(iSeld);
      [~,j] = ismember(edge,edges,'rows');
      if j == 0,
        fprintf('Edge is a swap pair, cannot remove.\n');
        return;
      end
      edges(j,:) = [];
      delete(hedges(j));
      hedges = [hedges(1:j-1),hedges(j+1:end)];
      
      set(obj.spHEdgeSelected,'Color',obj.unselectedColor);
      
      obj.spEdges = edges;
      obj.spHEdges = hedges;
      
      obj.updateTableSwap();
      obj.anyChangeMade = true;
    end
    %#OK
    function SpecifyHeadButtonPushed(obj, event)
      if isempty(obj.htISelected)
        error('Please select a point.');
      end
      updateTableHeadPt(obj,obj.htISelected);
      obj.htHead = obj.htISelected;
      obj.anyChangeMade = true;
    end
    %#OK
    function updateTableHeadPt(obj,iSelected)
      npt = size(obj.pts,1);
      hvec = false(npt,1);
      hvec(iSelected) = true; 
      obj.UITable.Data(:,2) = num2cell(hvec);      
    end
    %#OK
    function ClearButtonPushed(obj, event)
      updateTableHeadPt(obj,[]);
      obj.anyChangeMade = true;
    end
    
%     % Callback function
%     function SpecfyHeadLandmarkSwitchValueChanged(app, event)
%       app.updateTable(1);
%     end
    
    %#OK
    function UITableCellEdit(obj, event)
      indices = event.Indices;
      NAMECOL = 1;
      HEADTAILCOL = 2;
      
      if indices(2)==NAMECOL
        row = indices(1);
        newName = event.NewData;
        obj.ptNames{row} = newName;
        obj.sklHTxt(row).String = newName;
        obj.htHTxt(row).String = newName;
        obj.spHTxt(row).String = newName;
        obj.anyChangeMade = true;
      elseif indices(2)==HEADTAILCOL && ~event.PreviousData && event.NewData
        assert(false);
%         ht = app.UITable;
%         dat = ht.Data;
%         htcol = cell2mat(dat(:,HEADTAILCOL));
%         if nnz(htcol)>1
%           htcol(:) = 0;
%           htcol(indices(1)) = 1;
%           ht.Data(:,HEADTAILCOL) = num2cell(htcol);
%         end
      end
    end
    
    %#OK
    function closereq(obj,src,event)
      if obj.anyChangeMade
        btn = questdlg('Save changes?','Exit','Yes','No','Yes');
        switch btn
          case 'Yes'
            obj.lObj.setSkeletonEdges(obj.sklEdges);
            obj.lObj.setSkelHead(obj.htHead);
            obj.lObj.setFlipLandmarkMatches(obj.spEdges);
            obj.lObj.setSkelNames(obj.ptNames);
          otherwise
            % none
        end
      end     
      delete(obj);      
    end
  end
  
  % Component initialization
  methods (Access = private)
    
    % Create UIFigure and components
    function createComponents(obj)
      
      % Create hFig and hide until all components are created
      obj.hFig = figure('Visible', 'off');
      obj.hFig.AutoResizeChildren = 'off';
      obj.hFig.Position = [100 100 686 392];
      obj.hFig.Name = 'Landmark Specifications';
      obj.hFig.CloseRequestFcn = @(src,evt)obj.closereq(src,evt);
%       app.hFig.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);
      
%       % Create GridLayout
%       app.GridLayout = uigridlayout(app.hFig);
%       app.GridLayout.ColumnWidth = {448, '1x'};
%       app.GridLayout.RowHeight = {'1x'};
%       app.GridLayout.ColumnSpacing = 0;
%       app.GridLayout.RowSpacing = 0;
%       app.GridLayout.Padding = [0 0 0 0];
%       app.GridLayout.Scrollable = 'on';
      
      obj.LeftPanel = uipanel('Parent',obj.hFig,'units','normalized',...
        'Position',[0 0 0.5 1]);
%       obj.LeftPanel.Layout.Row = 1;
%       obj.LeftPanel.Layout.Column = 1;

      obj.TabGroup = uitabgroup(obj.LeftPanel);
%       obj.TabGroup.Position = [6 6 436 376];
      obj.TabGroup.SelectionChangedFcn = @(s,e)obj.cbkTabGroupSelChanged(e);      
      
      obj.SkeletonTab = uitab(obj.TabGroup);
      obj.SkeletonTab.Title = 'Skeleton';
      
      % Create axSkel
      obj.axSkel = axes(obj.SkeletonTab);
      title(obj.axSkel, '')
      xlabel(obj.axSkel, '')
      ylabel(obj.axSkel, '')
      obj.axSkel.XTick = [];
      obj.axSkel.YTick = [];
%       obj.axSkel.Position = [15 53 390 280];
      
      % Create AddEdgeButton
      obj.AddEdgeButton = uicontrol(obj.SkeletonTab,'style','pushbutton');
      obj.AddEdgeButton.Callback = @(s,e)obj.AddEdgeButtonPushed(e);
      obj.AddEdgeButton.Position = [118 17 100 23];
      obj.AddEdgeButton.String = 'Add Edge';
      
      % Create RemoveEdgeButton
      obj.RemoveEdgeButton = uicontrol(obj.SkeletonTab,'style','pushbutton');
      obj.RemoveEdgeButton.Callback = @(s,e)obj.RemoveEdgeButtonPushed(e);
      obj.RemoveEdgeButton.Position = [230 17 100 23];
      obj.RemoveEdgeButton.String = 'Remove Edge';
      
      % Create HeadTailTab
      obj.HeadTailTab = uitab(obj.TabGroup);
      obj.HeadTailTab.Title = 'Head/Tail';
      
      % Create axHT
      obj.axHT = axes(obj.HeadTailTab);
      title(obj.axHT, '')
      xlabel(obj.axHT, '')
      ylabel(obj.axHT, '')
      obj.axHT.XTick = [];
      obj.axHT.YTick = [];
      obj.axHT.Position = [23 30 390 291];
      
      % Create SpecifyHeadButton
      obj.SpecifyHeadButton = uicontrol(obj.HeadTailTab,'style','pushbutton');
      obj.SpecifyHeadButton.Callback = @(s,e)obj.SpecifyHeadButtonPushed(e);
      obj.SpecifyHeadButton.Position = [118 15 100 23];
      obj.SpecifyHeadButton.String = 'Specify Head';
      
      % Create ClearButton
      obj.ClearButton = uicontrol(obj.HeadTailTab,'style','pushbutton');
      obj.ClearButton.Callback = @(s,e)obj.ClearButtonPushed(e);
      obj.ClearButton.Position = [230 15 100 23];
      obj.ClearButton.String = 'Clear';
      
      % Create SwapPairsTab
      obj.SwapPairsTab = uitab(obj.TabGroup);
      obj.SwapPairsTab.Title = 'Swap Pairs';
      
      % Create axSwap
      obj.axSwap = axes(obj.SwapPairsTab);
      title(obj.axSwap, '')
      xlabel(obj.axSwap, '')
      ylabel(obj.axSwap, '')
      obj.axSwap.XTick = [];
      obj.axSwap.YTick = [];
      obj.axSwap.Position = [23 38 390 291];
      
      % Create AddPairButton
      obj.AddPairButton = uicontrol(obj.SwapPairsTab,'style','pushbutton');
      obj.AddPairButton.Callback = @(s,e)obj.AddPairButtonPushed(e);
      obj.AddPairButton.Position = [118 17 100 23];
      obj.AddPairButton.String = 'Add Pair';
      
      % Create RemovePairButton
      obj.RemovePairButton = uicontrol(obj.SwapPairsTab,'style','pushbutton');
      obj.RemovePairButton.Callback = @(s,e)obj.RemovePairButtonPushed(e);
      obj.RemovePairButton.Position = [230 17 100 23];
      obj.RemovePairButton.String = 'Remove Pair';
      
      % Create RightPanel
      obj.RightPanel = uipanel('Parent',obj.hFig,'units','normalized',...
        'Position',[0.5 0 0.5 1]);
%       obj.RightPanel.Layout.Row = 1;
%       obj.RightPanel.Layout.Column = 2;
      
      % Create UITable
      obj.UITable = uitable(obj.RightPanel);
      obj.UITable.RowName = 'numbered';
%       app.UITable.Data = app.ptNames(:);
      obj.UITable.ColumnName = {'Name'};
      obj.UITable.ColumnEditable = true;
      obj.UITable.ColumnWidth = {'auto'};      
      obj.UITable.CellEditCallback = @(s,e)obj.UITableCellEdit(e);
      obj.UITable.Position = [21 133 195 233];
      
      % Show the figure after all components are created
      obj.hFig.Visible = 'on';
    end
  end 
 
end