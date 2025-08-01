classdef landmark_specs < handle

  % Properties that correspond to app components
  properties (Access = public)
    hFig
    gl
    axSkel
    axHT
    axSwap
    isStandAlone;
    
%     GridLayout         matlab.ui.container.GridLayout
    TabGroup           %matlab.ui.container.TabGroup
    SkeletonTab        %matlab.ui.container.Tab
    AddEdgeButton      %matlab.ui.control.Button
    RemoveEdgeButton   %matlab.ui.control.Button
    HeadTailTab        %matlab.ui.container.Tab
%     SpecifyHeadButton  %matlab.ui.control.Button
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
    ptNames % [nphyspts] cellstr; working model/data for UI, to be written to lObj
    nview
    nPhysPoints
    imagescArgs
    labelCM
    axesProps 
    
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
    htHead % [1] [], or pt index. working model/data for UI, to be written to lObj. 
    htTail % etc
    htHTpts
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
    txtOffset

    im
  end
  
  methods (Access = public)
    
    function obj = landmark_specs(varargin)
      [hParent,isVert,argsrest] = myparse_nocheck(varargin,'hParent',[],'isVert',false);
      obj.createComponents('hParent',hParent,'isVert',isVert);
      obj.startupFcn(argsrest{:});
%       if nargout == 0
%         clear app
%       end
    end
    
    function delete(obj)
      if obj.isStandAlone,
        delete(obj.hFig);
      else
        delete(obj.gl);
      end
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
        % none
%         app.ptClickedHT(h,e);
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
%     function ptClickedHT(app,h,e)
%       iClicked = get(h,'UserData');
%       
%       iSeld = app.htISelected; %#OK
%       hpts = app.htHTpts;
%       
%       if isempty(iSeld)
%         iSelect = iClicked;
%         iUnselect = [];
%       elseif iSeld==iClicked
%         iSelect = [];
%         iUnselect = iSeld;
%       else
%         iSelect = iClicked;
%         iUnselect = iSeld;
%       end
%       if ~isempty(iSelect),
%         set(hpts(iSelect),'Marker',app.selectedMarker,'MarkerSize',app.selectedMarkerSize);
%       end
%       if ~isempty(iUnselect),
%         set(hpts(iUnselect),'Marker',app.unselectedMarker,'MarkerSize',app.unselectedMarkerSize);
%       end
%       app.htISelected = iSelect;
%     end
    %#OK
    function setTab(obj,tab)
      if tab==obj.SkeletonTab
        obj.updateTableSkel();
      elseif tab==obj.HeadTailTab
        obj.updateTableHT();
        obj.updateHTPts();
      elseif tab==obj.SwapPairsTab
        obj.updateTableSwap();
      end
    end
    function cbkTabGroupSelChanged(obj,e)
      tab = e.NewValue;
      obj.setTab(tab);
    end
    function updateTableSkel(obj)
      ht = obj.UITable;      
      ht.RowName = 'numbered';
      ht.Data = obj.ptNames(:);
      ht.ColumnName = {'Name'};
      ht.ColumnEditable = true;
      ht.ColumnWidth = {'1x'};
      %ht.ColumnWidth = {pos(3)*.8};
    end
    function updateTableHT(obj)
      nphyspts = numel(obj.ptNames); 
      htmat = false(nphyspts,2);
      ipthead = obj.htHead;
      ipttail = obj.htTail;
      if ~isempty(ipthead) && ~isnan(ipthead)
        htmat(ipthead,1) = true;
      end
      if ~isempty(ipttail) && ~isnan(ipttail)
        htmat(ipttail,2) = true;
      end
      
      tbl = obj.UITable;      
      tbl.RowName = 'numbered';
      tbl.Data = [obj.ptNames(:) num2cell(htmat)];
      tbl.ColumnName = {'Name' 'Head' 'Tail'};
      tbl.ColumnEditable = [true true true];
      tbl.ColumnWidth = {'10x','3x','3x'};
      %tbl.ColumnWidth = {pos(3)*.5 pos(3)*.15 pos(3)*.15};
    end
    function updateTableSwap(obj)
      partners = repmat({'none'},size(obj.ptNames(:)));
      swaps = obj.spEdges;
      nswap = size(swaps,1);
      for iswap=1:nswap
        s0 = swaps(iswap,1);
        s1 = swaps(iswap,2);
        partners{s0} = obj.ptNames{s1};
        partners{s1} = obj.ptNames{s0};
      end
      
      ht = obj.UITable;
      ht.RowName = 'numbered';
      ht.Data = [obj.ptNames(:) partners];
      ht.ColumnName = {'Name' 'Partner'};
      ht.ColumnEditable = [false false];
      ht.ColumnWidth = {'1x','1x'};
      %ht.ColumnWidth = {pos(3)*.4 pos(3)*.4};
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
      [obj.lObj,state,plotptsArgs,textArgs,...
        obj.txtOffset,obj.selectedMarkerSize,obj.unselectedMarkerSize,...
        obj.selectedMarker,obj.unselectedMarker,...
        obj.unselectedColor,obj.selectedColor,...
        obj.unselectedLineWidth,obj.selectedLineWidth,waiton_ui,...
        startTabTitle] = myparse(varargin,...
        'lObj',[],...
        'state',struct, ...
        'plotptsArgs',{'linewidth',2},...
        'textArgs',{'fontsize',16}, ...
        'txtOffset',1,...
        'selectedMarkerSize',8,...
        'unselectedMarkerSize',8,...
        'selectedMarker','o','unselectedMarker','+',...
        'unselectedColor',[.5,.5,.5],'selectedColor',[.8,0,.8],...
        'unselectedLineWidth',2,'selectedLineWidth',4, ...
        'waiton_ui',false, ...
        'startTabTitle',''...
        );

      assert(~isempty(obj.lObj));
      if obj.isStandAlone,
        centerOnParentFigure(obj.hFig,obj.lObj.hFig,'setParentFixUnitsPx',true);
      end
      
      if ~obj.lObj.isPrevAxesModeInfoSet()
        errordlg('Please freeze a labeled reference image for use with this UI.',...
          'No Reference Image');
        return;
      end

      obj.parseLabelerState();

      % overwrite with optional input state
      fns = fieldnames(state);
      for i = 1:numel(fns),
        obj.(fns{i}) = state.(fns{i});
      end

      obj.initTabSkel(textArgs,plotptsArgs);
      obj.initTabHeadTail(textArgs,plotptsArgs);
      obj.initTabSwap(textArgs,plotptsArgs);
      if ~isempty(startTabTitle),
        tabs = obj.TabGroup.Children;
        tabtitles = {tabs.Title};
        i = find(strcmp(tabtitles,startTabTitle),1);
        if isempty(i),
          warningNoTrace(sprintf('No tab with title %s',startTabTitle));
        else
          obj.TabGroup.SelectedTab = tabs(i);
        end
      end
      obj.setTab(obj.TabGroup.SelectedTab);
      if waiton_ui
        uiwait(obj.hFig);
      end
    end
    
    %#OK
    function [hpts,htxt] = initPtsAx(obj,hAx,plotptsArgs,textArgs)
      unselMarker = obj.unselectedMarker;
      unselMarkerSize = obj.unselectedMarkerSize;
      
      %npts = size(obj.pts,1);
      nphyspts = obj.nPhysPoints; % dont want to plot all views' points on hAx
      %ptnames = slbl.skelNames;

      hpts = gobjects(nphyspts,1);
      htxt = gobjects(nphyspts,1);
      pat = [repmat(' ',[1,obj.txtOffset]),'%d'];
      for i = 1:nphyspts
        hpts(i) = plot(hAx,obj.pts(i,1),obj.pts(i,2),unselMarker,...
          'Color',obj.labelCM(i,:),'MarkerFaceColor',obj.labelCM(i,:),...
          'UserData',i,'MarkerSize',unselMarkerSize,plotptsArgs{:});
        set(hpts(i),'ButtonDownFcn',@(h,e)obj.ptClicked(h,e));
        htxt(i) = text(hAx,obj.pts(i,1),obj.pts(i,2),...
          sprintf(pat,i),'Color',obj.labelCM(i,:),'PickableParts','none',textArgs{:});
      end   
    end
    %#OK
    function [hEdges,hEdgeSel] = initEdgesAx(obj,hAx,edges)
      
      selColor = obj.selectedColor;
      %       app.selectedMarker = selMarker;
      %       app.selectedMarkerSize = selMarkerSize;
      selLineWidth = obj.selectedLineWidth;
      unselColor = obj.unselectedColor;
      unselLineWidth = obj.unselectedLineWidth;
      
      nedge = size(edges,1);
      hEdges = gobjects(1,nedge);
      for i = 1:nedge
        hEdges(i) = plot(hAx,obj.pts(edges(i,:),1),obj.pts(edges(i,:),2),'-',...
          'Color',selColor,'LineWidth',unselLineWidth,...
          'ButtonDownFcn',@(h,e)obj.edgeClicked(h,e),...
          'UserData',edges(i,:),'Tag',sprintf('edge%d',i));
      end
      
      hEdgeSel = plot(hAx,nan(2,1),nan(2,1),'-','Color',unselColor,...
        'LineWidth',selLineWidth,'PickableParts','none');
    end
    %#OK
    function initTabSkel(obj,textArgs,plotptsArgs)
      hAx = obj.axSkel;
      
      hIm = obj.initImage(hAx);
      [hpts,htxt] = obj.initPtsAx(hAx,plotptsArgs,textArgs);
      [hEdges,hEdgeSel] = obj.initEdgesAx(hAx,obj.sklEdges);

      obj.sklHpts = hpts;
      obj.sklHEdges = hEdges;
      obj.sklHIm = hIm;
      obj.sklHTxt = htxt;
      obj.sklHEdgeSelected = hEdgeSel;
      obj.sklISelected = [];
    end
    %#OK
    function initTabHeadTail(obj,textArgs,plotptsArgs)
      hAx = obj.axHT;
      
      hIm = obj.initImage(hAx);
      [hpts,htxt] = obj.initPtsAx(hAx,plotptsArgs,textArgs);         
      obj.htHTpts = hpts;
      obj.htHIm = hIm;
      obj.htHTxt = htxt;
      obj.htISelected = [];
    end
    %#OK
    function initTabSwap(obj,textArgs,plotptsArgs)
      hAx = obj.axSwap;
      
      hIm = obj.initImage(hAx);
      [hpts,htxt] = obj.initPtsAx(hAx,plotptsArgs,textArgs);
      [hEdges,hEdgeSel] = obj.initEdgesAx(hAx,obj.spEdges);

      obj.spHpts = hpts;
      obj.spHEdges = hEdges;
      obj.spHIm = hIm;
      obj.spHTxt = htxt;
      obj.spHEdgeSelected = hEdgeSel;
      obj.spISelected = [];
    end

    function parseLabelerState(obj)
      freezeInfo = obj.lObj.prevAxesModeInfo;
      imagescArgs = {'XData',freezeInfo.xdata,'YData',freezeInfo.ydata};
      im = obj.lObj.prevAxesModeInfo.im; %#ok<*PROP>
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
      
      lpos = obj.lObj.labelsGTaware;
      s = lpos{iMov};
      [~,p,~] = Labels.isLabeledFT(s,frm,iTgt);
      pts = reshape(p,numel(p)/2,2);
      nan_pts = isnan(pts(:,1));
      max_p = repmat(max(pts,[],1),[size(pts,1),1]);
      min_p = repmat(min(pts,[],1),[size(pts,1),1]);
      rp = rand(size(pts)).*(max_p-min_p) + min_p;
      pts(nan_pts,:) = rp(nan_pts,:);
      %pts = lpos{iMov}(:,:,frm,iTgt);
      if isrotated,
        pts = [pts,ones(size(pts,1),1)]*freezeInfo.A;
        pts = pts(:,1:2);
      end
      labelCM = obj.lObj.LabelPointColors;            
      if isempty(labelCM)
        labelCM = jet(size(pts,1));
      end
      
      state = obj.lObj.getKeypointParams();
      obj.sklEdges = state.sklEdges;
      obj.spEdges = state.spEdges;
      obj.htHead = state.htHead;
      obj.htTail = state.htTail;
      obj.ptNames = state.ptNames;

      obj.im = im;
      obj.pts = pts;
      obj.nview = obj.lObj.nview;
      obj.nPhysPoints = obj.lObj.nPhysPoints;
      obj.imagescArgs = imagescArgs;
      obj.labelCM = labelCM;
      obj.axesProps = axesProps;
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
        error('A keypoint can have at most one swap partner. Please remove any existing/conflicting swap pairs.');
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

%     function SpecifyHeadButtonPushed(obj, event)
%       if isempty(obj.htISelected)
%         error('Please select a point.');
%       end
%       updateTableHeadPt(obj,obj.htISelected);
%       obj.htHead = obj.htISelected;
%       obj.anyChangeMade = true;
%     end
    %#OK
%     function updateTableHeadPt(obj,iSelected)
%       npt = size(obj.pts,1);
%       hvec = false(npt,1);
%       hvec(iSelected) = true; 
%       obj.UITable.Data(:,2) = num2cell(hvec);      
%     end
    %#OK
%     function ClearButtonPushed(obj, event)
%       updateTableHeadPt(obj,[]);
%       obj.anyChangeMade = true;
%     end
    
%     % Callback function
%     function SpecfyHeadLandmarkSwitchValueChanged(app, event)
%       app.updateTable(1);
%     end
    
    %#OK
    function UITableCellEdit(obj, event)
      indices = event.Indices;
      NAMECOL = 1;
      HEADTAILCOL = [2 3];
      tblrow = indices(1);
      tblcol = indices(2);
      
      if tblcol==NAMECOL
        newName = event.NewData;
        obj.ptNames{tblrow} = newName;
%         obj.sklHTxt(tblrow).String = newName;
%         obj.htHTxt(tblrow).String = newName;
%         obj.spHTxt(tblrow).String = newName;
      elseif any(tblcol==HEADTAILCOL)
        tbl = obj.UITable;
        dat = tbl.Data;
        tf = cell2mat(dat(:,tblcol));        
%         if event.NewData && ~event.PreviousData
%           % no-check=>check
%           
%         elseif ~event.NewData && event.PreviousData
%           
%         end

        % enforce only one check per col
        if nnz(tf)>1
          tf(:) = 0;
          tf(tblrow) = 1;
          tbl.Data(:,tblcol) = num2cell(tf);
        end
        ipt = find(tf); % could be empty
        FLDS = {[] 'htHead' 'htTail'};
        obj.(FLDS{tblcol}) = ipt;
        obj.updateHTPts();
      end
      obj.anyChangeMade = true;
    end

    function updateHTPts(obj)
      hpts = obj.htHTpts;
      set(hpts,...
          'Marker',obj.unselectedMarker,...
          'MarkerSize',obj.unselectedMarkerSize);
        
      ipthead = obj.htHead;
      ipttail = obj.htTail;        
      if ~isempty(ipthead)
        set(hpts(ipthead),...
          'Marker',obj.selectedMarker,...
          'MarkerSize',obj.selectedMarkerSize)
      end
      if ~isempty(ipttail)
        set(hpts(ipttail),...
          'Marker',obj.selectedMarker,...
          'MarkerSize',obj.selectedMarkerSize)
      end
    end

    function s = getState(obj)

      s = struct;
      s.sklEdges = obj.sklEdges;
      s.htHead = obj.htHead;
      s.htTail = obj.htTail;
      s.spEdges = obj.spEdges;
      s.ptNames = obj.ptNames;

    end

    function accept(obj)
      obj.lObj.setKeypointParams(obj.getState());
    end

    function AcceptButtonPushed(obj,src,evt)
      obj.accept();
      delete(obj);
    end

    function CancelButtonPushed(obj,src,evt)
      delete(obj);
    end

    %#OK
    function closereq(obj,src,event)
      if obj.anyChangeMade
        btn = questdlg('Save changes?','Exit','Yes','No','Yes');
        switch btn
          case 'Yes'
            obj.accept();
          otherwise
            % none
        end
      end
      delete(obj);      
    end
    
  end
  
  % Component initialization
  methods (Access = private)

    function hIm = initImage(obj,hAx)
      hold(hAx,'off');
      hIm = imagesc(obj.im,'Parent',hAx,obj.imagescArgs{:});
      hold(hAx,'on');
      axis(hAx,'off','image');
      hAx.XTick = [];
      hAx.YTick = [];
      set(hAx,obj.axesProps{:});
      colormap(hAx,'gray');
    end

    
    % Create UIFigure and components
    function createComponents(obj,varargin)
      
      [hParent,isVert] = myparse(varargin,'hParent',[],'isVert',false);

      if ~isempty(hParent),
        obj.hFig = hParent;
        obj.isStandAlone = false;
      else
        % Create hFig and hide until all components are created
        obj.hFig = uifigure; %('Visible', 'off');
        obj.hFig.Position = [100 100 686 392];
        obj.hFig.Name = 'Keypoint Specifications';
        obj.hFig.MenuBar = 'none';
        obj.hFig.CloseRequestFcn = @(src,evt)obj.closereq(src,evt);
        obj.isStandAlone = true;
      end
      if isVert,
        obj.gl = uigridlayout(obj.hFig,[2,1],'RowHeight',{'4x','1x'});
      else
        obj.gl = uigridlayout(obj.hFig,[1,2],'ColumnWidth',{'3x','2x'});
      end

%       app.hFig.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);
      
%       % Create GridLayout
%       app.GridLayout = uigridlayout(app.hFig);
%       app.GridLayout.ColumnWidth = {448, '1x'};
%       app.GridLayout.RowHeight = {'1x'};
%       app.GridLayout.ColumnSpacing = 0;
%       app.GridLayout.RowSpacing = 0;
%       app.GridLayout.Padding = [0 0 0 0];
%       app.GridLayout.Scrollable = 'on';
      
      glbuttons = cell(1,3);

      gl2 = uigridlayout(obj.gl,[1,1],'ColumnSpacing',0,'RowSpacing',0,'Padding',zeros(1,4));
      obj.TabGroup = uitabgroup(gl2);
%       obj.TabGroup.Position = [6 6 436 376];
      obj.TabGroup.SelectionChangedFcn = @(s,e)obj.cbkTabGroupSelChanged(e);      
      
      obj.SkeletonTab = uitab(obj.TabGroup);
      obj.SkeletonTab.Title = 'Skeleton';

      gltab = uigridlayout(obj.SkeletonTab,[2,1],'RowHeight',{'1x','fit'},'Padding',zeros(1,4));

      % Create axSkel
      pan = uipanel(gltab,'BorderType','none');
      tl = tiledlayout(pan,'vertical','TileSpacing','compact','Padding','compact');
      obj.axSkel = nexttile(tl);
      obj.axSkel.XTick = [];
      obj.axSkel.YTick = [];
      
      % Create AddEdgeButton
      glbutton = uigridlayout(gltab,[1,4],'ColumnWidth',{'1x','1x','1x','1x'},'RowHeight',{'fit'});
      obj.AddEdgeButton = uibutton(glbutton,"push",'Text','Add Edge','ButtonPushedFcn',@(s,e)obj.AddEdgeButtonPushed(e));
      obj.RemoveEdgeButton = uibutton(glbutton,"push",'Text','Remove Edge','ButtonPushedFcn',@(s,e)obj.RemoveEdgeButtonPushed(e));
      glbuttons{1} = glbutton;
      
      % Create HeadTailTab
      obj.HeadTailTab = uitab(obj.TabGroup);
      obj.HeadTailTab.Title = 'Head/Tail';

      gltab = uigridlayout(obj.HeadTailTab,[2,1],'RowHeight',{'1x','fit'},'Padding',zeros(1,4));

      pan = uipanel(gltab,'BorderType','none');
      tl = tiledlayout(pan,'vertical','TileSpacing','compact','Padding','compact');

      % Create axHT
      obj.axHT = nexttile(tl);
      obj.axHT.XTick = [];
      obj.axHT.YTick = [];
      
      glbutton = uigridlayout(gltab,[1,4],'ColumnWidth',{'1x','1x','1x','1x'},'RowHeight',{'fit'});
      glbuttons{2} = glbutton;

      % Create SpecifyHeadButton
%       obj.SpecifyHeadButton = uicontrol(obj.HeadTailTab,'style','pushbutton');
%       obj.SpecifyHeadButton.Callback = @(s,e)obj.SpecifyHeadButtonPushed(e);
%       obj.SpecifyHeadButton.Position = [118 15 100 23];
%       obj.SpecifyHeadButton.String = 'Specify Head';
      
%       % Create ClearButton
%       obj.ClearButton = uicontrol(obj.HeadTailTab,'style','pushbutton');
%       obj.ClearButton.Callback = @(s,e)obj.ClearButtonPushed(e);
%       obj.ClearButton.Position = [230 15 100 23];
%       obj.ClearButton.String = 'Clear';
      
      % Create SwapPairsTab
      obj.SwapPairsTab = uitab(obj.TabGroup);
      obj.SwapPairsTab.Title = 'Correspondences';
      
      gltab = uigridlayout(obj.SwapPairsTab,[2,1],'RowHeight',{'1x','fit'},'Padding',zeros(1,4));

      % Create axSwap
      pan = uipanel(gltab,'BorderType','none');
      tl = tiledlayout(pan,'vertical','TileSpacing','compact','Padding','compact');
      obj.axSwap = nexttile(tl);
      obj.axSwap.XTick = [];
      obj.axSwap.YTick = [];
      
      % Create AddPairButton      
      glbutton = uigridlayout(gltab,[1,4],'ColumnWidth',{'1x','1x','1x','1x'},'RowHeight',{'fit'});
      obj.AddPairButton = uibutton(glbutton,"push",'Text','Add Pair','ButtonPushedFcn',@(s,e)obj.AddPairButtonPushed(e));
      obj.RemovePairButton = uibutton(glbutton,"push",'Text','Remove Pair','ButtonPushedFcn',@(s,e)obj.RemovePairButtonPushed(e));
      glbuttons{3} = glbutton;

      for i = 1:numel(glbuttons),
        acceptbutton = uibutton(glbuttons{i},"push",'Text','Accept','ButtonPushedFcn',@(s,e)obj.AcceptButtonPushed(s,e));
        acceptbutton.Layout.Column = 3;
        cancelbutton = uibutton(glbuttons{i},"push",'Text','Cancel','ButtonPushedFcn',@(s,e)obj.CancelButtonPushed(s,e));
        cancelbutton.Layout.Column = 4;
      end


      % Create UITable
      obj.UITable = uitable(obj.gl);
      obj.UITable.RowName = 'numbered';
%       app.UITable.Data = app.ptNames(:);
      obj.UITable.ColumnName = {'Name'};
      obj.UITable.ColumnEditable = true;
      obj.UITable.ColumnWidth = {'auto'};      
      obj.UITable.CellEditCallback = @(s,e)obj.UITableCellEdit(e);
      
      % Show the figure after all components are created
      if obj.isStandAlone,
        obj.hFig.Visible = 'on';
      end
    end
  end 
 
end