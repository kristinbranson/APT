function [position,hline] = get2ptline(varargin)

if nargin < 1,
  hax = gca;
  hfig = gcf;
else
  h = varargin{1};
  htype = get(h,'type');
  if strcmpi(htype,'figure'),
    hfig = h;
    hax = get(hfig,'CurrentAxes');
  elseif strcmpi(htype,'axes'),
    hax = h;
    hfig = get(hax,'parent');
  else
    error('Parent handle must be figure or axes');
  end

end
position = [];
if nargin == 2,
  if ndims(size(varargin{2})) ~= 2 || ~all(size(varargin{2})==2),
    error('Input position must be of the form [X1 Y1; X2 Y2]');
  end
  position = varargin{2};
elseif nargin >= 3,
  if numel(varargin{2}) ~= 2 || numel(varargin{2}) ~= 2,
    error('Input end points X and Y must be of the form X = [X1 X2], Y =[Y1 Y2]');
  end
  position = [varargin{2}(:),varargin{3}(:)];
end

% allow user to click two points in the axes to set end points of line
% initially
holdstate = ishold;
buttondownfcn = get(hfig,'WindowButtonDownFcn');
winbuttonmotionfcn = get(hfig,'WindowButtonMotionFcn');
if isempty(position),
  axes(hax);
  fprintf('Click to set end points of line.\n');
  hold on;
  hline = plot([0,0],[0,0],'bs-','visible','off','userdata',0,'hittest','off');
  if ~holdstate,
    hold off;
  end
  set(hfig,'WindowButtonDownFcn',@(src,evt) bdcb1(src,evt,hax,hfig,hline));
  waitfor(hline,'userdata',1);
  x = get(hline,'xdata');
  y = get(hline,'ydata');
  position = [x(:),y(:)];
  set(hfig,'WindowButtonMotionFcn',winbuttonmotionfcn);
  set(hfig,'WindowButtonDownFcn',buttondownfcn);
  set(hline,'hittest','on');
else
  xlim = get(hax,'xlim');
  ylim = get(hax,'ylim');
  hold on;
  hline = plot(position(:,1),position(:,2),'bs-');
  if ~holdstate,
    hold off;
  end
  set(hax,'xlim',xlim,'ylim',ylim);
end

function wbmcb(src,evnt,hax,hfig,hline)

cp = get(hax,'CurrentPoint');
x = get(hline,'xdata');
y = get(hline,'ydata');
x(end) = cp(1,1);
y(end) = cp(1,2);
set(hline,'xdata',x,'ydata',y);

function bdcb1(src,evnt,hax,hfig,hline)

if strcmp(get(hfig,'SelectionType'),'normal') && ...
    hax == get(hfig,'CurrentAxes'),
  
  % check that we've clicked in the right axis
  h = gco;
  while true,
    htype = get(h,'type');
    if h == hax,
      break;
    end
    if ~ishandle(h) || h == 0 || strcmpi(htype,'figure') || (strcmpi(htype,'axes') && h ~= hax),
      return;
    end
    h = get(h,'parent');
  end
    
  cp = get(hax,'CurrentPoint');
  xinit = cp(1,1);
  yinit = cp(1,2);
  xlim = get(hax,'xlim');
  ylim = get(hax,'ylim');
  set(hline,'xdata',[xinit,xinit],'ydata',[yinit,yinit],'visible','on');
  set(hax,'xlim',xlim,'ylim',ylim);
  set(hfig,'WindowButtonMotionFcn',@(src,evt) wbmcb(src,evt,hax,hfig,hline));
  set(hfig,'WindowButtonDownFcn',@(src,evt) bdcb2(src,evt,hax,hfig,hline));
end

function bdcb2(src,evnt,hax,hfig,hline)

set(hline,'userdata',1);