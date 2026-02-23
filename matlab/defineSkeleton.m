function edges = defineSkeleton(im,varargin)

if isa(im,'Labeler')
  lObj = im;
else
  pts = varargin{1};
  varargin = varargin(2:end);
end

[hFig,hAx,imagescArgs,labelCM,...
  plotptsArgs,axesProps,textArgs,...
  edges,txtOffset,selectedMarkerSize,unselectedMarkerSize,...
  selectedMarker,unselectedMarker,...
  unselectedColor,selectedColor,...
  unselectedLineWidth,selectedLineWidth] = myparse(varargin,'hfig',nan,'hax',nan,...
  'imagesc_args',{},'label_cm',[],'plotpts_args',{},...
  'axes_props',{},'text_args',{},...
  'edges',zeros(0,2),...
  'txtOffset',2,...
  'selectedMarkerSize',12,...
  'unselectedMarkerSize',8,...
  'selectedMarker','o','unselectedMarker','+',...
  'unselectedColor',[.5,.5,.5],'selectedColor',[.8,0,.8],...
  'unselectedLineWidth',2,'selectedLineWidth',4);

if isa(im,'Labeler')
  [im,pts,imagescArgs,labelCM,axesProps,txtOffset] = parseLabelerArgs(lObj);
end

if isempty(edges),
  edges = zeros(0,2);
end


if ~ishandle(hAx),
  if ~ishandle(hFig),
    hFig = figure;
  end
  hAx = axes('parent',hFig);
else
  hFig = get(hAx,'Parent');
  delete(findobj(hFig,'Style','pushbutton'));
end

hold(hAx,'off');
hIm = imagesc(im,'Parent',hAx,imagescArgs{:});
hold(hAx,'on');
colormap(hAx,'gray');
axis(hAx,'image','off');

npts = size(pts,1);
hpts = gobjects(npts,1);

if isempty(labelCM),
  labelCM = jet(npts);
end

htxt = gobjects(npts,1);
for i = 1:npts,
  htxt(i) = text(pts(i,1)+txtOffset,pts(i,2)+txtOffset,num2str(i),'Color',labelCM(i,:),'PickableParts','none',textArgs{:});
end

hEdgeSelected = plot(nan(2,1),nan(2,1),'-','Color',unselectedColor,'LineWidth',selectedLineWidth,'PickableParts','none');

for i = 1:npts,
  hpts(i) = plot(hAx,pts(i,1),pts(i,2),unselectedMarker,'Color',labelCM(i,:),'MarkerFaceColor',labelCM(i,:),...
    'UserData',i,'MarkerSize',unselectedMarkerSize,plotptsArgs{:});
  set(hpts(i),'ButtonDownFcn',@ptClicked);
end

hedges = gobjects(1,size(edges,1));
for i = 1:size(edges,1),
  hedges(i) = plot(hAx,pts(edges(i,:),1),pts(edges(i,:),2),'-','Color',selectedColor,'LineWidth',unselectedLineWidth,...
    'ButtonDownFcn',@edgeClicked,'UserData',edges(i,:),'Tag',sprintf('edge%d',i));
end

if ~isempty(axesProps),
  set(hAx,axesProps{:});
end

iSelected = [];

set(hFig,'Units','normalized');
set(hAx,'Units','normalized');
pos = get(hAx,'Position');
w = .2;
d = .01;
h = .05;
addPos = [pos(1)+pos(3)/2-1.5*(w-d/2),pos(2)-h-d,w,h];
removePos = addPos;
removePos(1) = addPos(1)+w+d;
donePos = removePos;
donePos(1) = removePos(1)+w+d;
pbAddEdge = uicontrol('Style','pushbutton','Parent',hFig,'Units','normalized','Position',addPos,'String','Add edge',...
  'BackgroundColor',[0,.6,0],'ForegroundColor','w','FontWeight','bold',...
  'Callback',@cbkPBAddEdge);
pbRemoveEdge = uicontrol('Style','pushbutton','Parent',hFig,'Units','normalized','Position',removePos,'String','Remove edge',...
  'BackgroundColor',[.6,0,0],'ForegroundColor','w','FontWeight','bold',...
  'Callback',@cbkPBRemoveEdge);
pbDone = uicontrol('Style','pushbutton','Parent',hFig,'Units','normalized','Position',donePos,'String','Done',...
  'BackgroundColor',[0,0,.8],'ForegroundColor','w','FontWeight','bold',...
  'Callback',@cbkPBDone);

uiwait(hFig);

  function ptClicked(h,e)
    iClicked = get(h,'UserData');
    j = find(iSelected==iClicked);
    if isempty(j),
      iSelect = iClicked;
      iUnselect = [];
      iSelected(end+1) = iClicked;
      if numel(iSelected) > 2,
        iUnselect = iSelected(1);
        iSelected = iSelected(end-1:end);
      end
    else
      iSelected(j) = [];
      iSelect = [];
      iUnselect = iClicked;
    end
    if ~isempty(iSelect),
      set(hpts(iSelect),'Marker',selectedMarker,'MarkerSize',selectedMarkerSize);
    end
    if ~isempty(iUnselect),
      set(hpts(iUnselect),'Marker',unselectedMarker,'MarkerSize',unselectedMarkerSize);
    end
    if numel(iSelected) >= 2,
      isSelected = ismember(sort(iSelected),edges,'rows');
      set(hEdgeSelected,'XData',pts(iSelected,1),'YData',pts(iSelected,2));
      if isSelected,
        set(hEdgeSelected,'Color',selectedColor);
      else
        set(hEdgeSelected,'Color',unselectedColor);
      end
    else
      set(hEdgeSelected,'XData',nan(2,1),'YData',nan(2,1));
    end
    
  end

  function edgeClicked(h,e)
    
    edge = get(h,'UserData');
    if ~isempty(iSelected),
      set(hpts(iSelected),'Marker',unselectedMarker,'MarkerSize',unselectedMarkerSize);
      set(hEdgeSelected,'XData',nan(2,1),'YData',nan(2,1));
    end
    if numel(iSelected==2) && all(edge == sort(iSelected)),
      iSelected = [];
    else
      iSelected = edge;
      set(hpts(iSelected),'Marker',selectedMarker,'MarkerSize',selectedMarkerSize);
      set(hEdgeSelected,'XData',pts(iSelected,1),'YData',pts(iSelected,2));
      isSelected = ismember(sort(iSelected),edges,'rows');
      if isSelected,
        set(hEdgeSelected,'Color',selectedColor);
      else
        % this should never happen
        set(hEdgeSelected,'Color',unselectedColor);
      end
    end
    
  end

  function cbkPBAddEdge(h,e)
    if numel(iSelected) < 2,
      fprintf('No edge selected. Edge must be selected to add.\n');
      return;
    end
    edge = sort(iSelected);
    if ismember(edge,edges,'rows'),
      fprintf('Edge already selected.\n');
      return;
    end
    edges(end+1,:) = edge;
    hedges(end+1) = plot(pts(edge,1),pts(edge,2),'-','Color',selectedColor,'LineWidth',unselectedLineWidth,...
      'ButtonDownFcn',@edgeClicked,'UserData',edge,'Tag',sprintf('edge%d',numel(hedges)+1)); 
    set(hEdgeSelected,'Color',selectedColor);
    moveEdgesToBack();
  end


  function cbkPBRemoveEdge(h,e)
    
    if numel(iSelected) < 2,
      fprintf('No edge selected. Edge must be selected to add.\n');
      return;
    end
    edge = sort(iSelected);
    [~,j] = ismember(edge,edges,'rows');
    if j == 0,
      fprintf('Edge is not included in skeleton, cannot remove.\n');
      return;
    end
    edges(j,:) = [];
    delete(hedges(j));
    hedges = [hedges(1:j-1),hedges(j+1:end)];
    set(hEdgeSelected,'Color',unselectedColor);
    
  end

  function cbkPBDone(h,e)
    
    delete(hFig);
    
  end

  function moveEdgesToBack()
    hchil = get(hAx,'Children');
    hchil1 = [hIm;htxt(:);hedges(:);hEdgeSelected;hpts(:)];
    hleft = hchil;
    hleft(ismember(hleft,hchil1)) = [];
    set(hAx,'Children',flipud([hleft;hchil1]));    
  end

  function [im,pts,imagescArgs,labelCM,axesProps,txtOffset] = ...
      parseLabelerArgs(lObj)
    
    freezeInfo = lObj.prevAxesModeTargetSpec;
    imagescArgs = {'XData',freezeInfo.xdata,'YData',freezeInfo.ydata};
    im = freezeInfo.im;
    axcProps = freezeInfo.prevAxesProps;
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
    
    s = lObj.labelsGTaware{iMov};
    [~,p,~] = Labels.isLabeledFT(s,frm,iTgt);
    pts = reshape(p,numel(p)/2,2);
    %pts = lpos{iMov}(:,:,frm,iTgt);
    if lObj.nview > 1
      pts = pts(1:(end/lObj.nview),:);
    end
    if isrotated,
      pts = [pts,ones(size(pts,1),1)]*freezeInfo.A;
      pts = pts(:,1:2);
    end
    labelCM = lObj.LabelPointColors;
    txtOffset = lObj.labelPointsPlotInfo.TextOffset;
    
  end

end