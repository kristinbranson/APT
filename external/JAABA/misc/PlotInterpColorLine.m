% h = PlotInterpColorLine(x,y,colors,alphas,...)
function h = PlotInterpColorLine(x,y,colors,alphas,varargin)

if nargin < 4,
  alphas = [];
end
if ischar(alphas),
  varargin = [alphas,varargin];
  alphas = [];
end

[usepatch,parent,leftovers] = myparse_nocheck(varargin,'usepatch',true,'Parent',[]);

x = x(:);
y = y(:);

if isempty(parent),
  parent = gca;
else
  leftovers = [leftovers,{'Parent',parent}];
end
  
if usepatch,

  if numel(alphas) > 1,
    alphas = [alphas;flipud(alphas)];
  end
  
  x = [x;flipud(x)];
  y = [y;flipud(y)];
  colors = [colors;flipud(colors)];

  
  if isempty(alphas),
    h = patch(x,y,[0,0,0],'FaceVertexCData',colors,'FaceColor','none','EdgeColor','interp',leftovers{:});
  elseif numel(alphas) > 1,
    h = patch(x,y,[0,0,0],'FaceVertexCData',colors,'FaceColor','none','EdgeColor','interp','FaceVertexAlphaData',...
      alphas,'EdgeAlpha','interp',leftovers{:});
  else
    h = patch(x,y,[0,0,0],'FaceVertexCData',colors,'FaceColor','none','EdgeColor','interp','EdgeAlpha',...
      alphas,leftovers{:});
  end

else
  
  h = nan(1,numel(x)-1);
  ish = ishold(parent);
  for i = 1:numel(x)-1,
    h(i) = plot(x(i:i+1),y(i:i+1),'Color',colors(i,:),leftovers{:});
    hold(parent,'on');
  end
  if ~ish,
    hold(parent,'off');
  end
end