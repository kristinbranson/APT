% [BOUNDX,BOUNDY] = LABEL_OPEN_EDGES(RGBI)
%
% Input: RGBI is a RGB image we want to label edges in. 
% This program allows the user to input any number of open
% edges. Here is the interactive usage:
% label_open_edges commands:
% LEFT CLICK with the mouse to add a new point.
% RIGHT CLICK with the mouse to add a new point on an image edge.
% MIDDLE CLICK with the mouse to finish the curve.
% Press ENTER to finish the curve.
% Press <CONTROL> z to undo the last curve piece added.
% Press H for a list of commands.
% Press z to zoom out.
% Press Z to put in zoomin mode, then draw a rectangle to zoomin on.
% Press - to decrease the first edge threshold.
% Press = to increase the first edge threshold.
% Press _ to decrease the second edge threshold.
% Press + to increase the second edge threshold.
% Press t to see the image with/without the edges.
% Press <CONTROL> q to quit.

function varargout = label_open_edges(arg1)

if nargin ~= 1,
  error('LABEL_OPEN_EDGES: Usage: label_open_edges(rgbI)');
end;

if strcmp(arg1,'Move'),
  move;
  return;
elseif strcmp(arg1,'ButtonDown'),
  buttondown;
  return;
elseif strcmp(arg1,'Drag'),
  drag;
  return;
elseif strcmp(arg1,'ButtonUp'),
  buttonup;
  return;
elseif strcmp(arg1,'KeyPress'),
  keypress;
  return;
end;

fprintf('Label edges: Type ''H'' for a list of commands.\n');

global FIG_H AXIS_H IMAGE_H NROWS NCOLS;
global BOUNDX BOUNDY CURVEN LASTLINE_H CURVEPIECESTART;
global EDGEI RGBI RGBI0 DISTEDGEI NNEDGEI THRESH ISEDGESHOWN;

% rgb image
RGBI0 = im2double(arg1);
RGBI = RGBI0;
[NROWS,NCOLS,ncolors] = size(RGBI0);

[EDGEI,THRESH] = edge(rgb2gray(RGBI0),'canny');

% draw the edges on the image
RGBI = reshape(RGBI,[NROWS*NCOLS],ncolors);
RGBI(EDGEI,1) = 1; RGBI(EDGEI,2) = .75; RGBI(EDGEI,3) = 1;
RGBI = reshape(RGBI,[NROWS,NCOLS,ncolors]);

% draw the image
clf;
image(RGBI); axis('image'); hold on;

% find the closest edge to each pixel
[DISTEDGE,NNEDGEI] = bwdist(EDGEI);

% save which image we are looking at
FIG_H = gcf;
AXIS_H = gca;
IMAGE_H = findobj(AXIS_H,'type','image');

% we will use IMAGE_H's UserData as a flag to see when we are done
set(IMAGE_H,'UserData',1);

% Initialize the lines drawn
BOUNDX{1} = []; BOUNDY{1} = [];
CURVEN = 1;
LASTLINE_H = [];
CURVEPIECESTART = [];
ISEDGESHOWN = 1;

% wait for the mouse to be pushed
set(IMAGE_H,'ButtonDownFcn','label_open_edges(''ButtonDown'')');
set(FIG_H,'WindowButtonMotionFcn','label_open_edges(''Move'')');

% or for a key to be pressed
set(FIG_H,'KeyPressFcn','label_open_edges(''KeyPress'')');

waitfor(IMAGE_H,'UserData','Done');

varargout = {BOUNDX , BOUNDY};

clear global BOUNDX BOUNDY;

% mouse moved
function move

global FIG_H IMAGE_H AXIS_H;
global BOUNDX BOUNDY CURVEN LASTLINE_H;
global EDGEI RGBI DISTEDGEI NNEDGEI;
global EDGEPOINTER_H;

% find the mouse location
[imageHandle,x,y] = over_image(AXIS_H);
if imageHandle ~= IMAGE_H | imageHandle == 0, return; end;

% find the closest edge point
[yedge,xedge] = ind2sub(size(EDGEI),NNEDGEI(round(y),round(x)));
if isempty(EDGEPOINTER_H) | ~ishandle(EDGEPOINTER_H),
  EDGEPOINTER_H = plot(xedge,yedge,'b.','markersize',12,'hittest','off');
else,
  set(EDGEPOINTER_H,'XData',xedge,'YData',yedge);
end;

% mouse button pressed down
function buttondown

global FIG_H IMAGE_H AXIS_H;
global BOUNDX BOUNDY CURVEN LASTLINE_H CURVEPIECESTART;
global EDGEI RGBI DISTEDGEI NNEDGEI;

% which mouse button was pushed?
selection_type = get(FIG_H,'SelectionType');

if strcmp(selection_type,'normal'),
  b = 1;
elseif strcmp(selection_type,'extend'),
  b = 2;
elseif strcmp(selection_type,'alt'),
  b = 3;
end;

% left mouse button: input a new point
if b == 1,
  % find the point clicked
  [imageHandle,x,y] = over_image(AXIS_H);
  if imageHandle ~= IMAGE_H | imageHandle == 0, return; end;

  i = length(BOUNDX{CURVEN});
  % add the point to the current curve
  BOUNDX{CURVEN} = [BOUNDX{CURVEN} ; x];
  BOUNDY{CURVEN} = [BOUNDY{CURVEN} ; y];
  
  % plot the new point
  if isempty(LASTLINE_H),
    LASTLINE_H = plot(BOUNDX{CURVEN},BOUNDY{CURVEN},...
		      'r','erasemode','none','hittest','off');
    CURVEPIECESTART = 1;
  else,
    set(LASTLINE_H,'XData',BOUNDX{CURVEN},'YData',BOUNDY{CURVEN});
    CURVEPIECESTART = [CURVEPIECESTART;i+1];
  end;
  
  set(FIG_H,'WindowButtonMotionFcn','label_open_edges(''Drag'')',...
	    'WindowButtonUpFcn','label_open_edges(''ButtonUp'')');
  set(IMAGE_H,'ButtonDownFcn','');

% right mouse button: input a new point on an edge
elseif b == 3,

  % input a point
  [imageHandle,x,y] = over_image(AXIS_H);
  if imageHandle ~= IMAGE_H | imageHandle == 0, return; end;
  
  i = length(BOUNDX{CURVEN});
  
  % find the closest edge point to the boundary point
  [yedge,xedge] = ind2sub(size(EDGEI),NNEDGEI(round(y),round(x)));
  
  % add the point to the current curve
  BOUNDX{CURVEN} = [BOUNDX{CURVEN} ; xedge];
  BOUNDY{CURVEN} = [BOUNDY{CURVEN} ; yedge];
  
  % plot the new point
  if isempty(LASTLINE_H),
    LASTLINE_H = plot(BOUNDX{CURVEN},BOUNDY{CURVEN},...
		      'r','erasemode','none','hittest','off');
    CURVEPIECESTART = 1;
  else,
    set(LASTLINE_H,'XData',BOUNDX{CURVEN},'YData',BOUNDY{CURVEN});
    CURVEPIECESTART = [CURVEPIECESTART;i+1];
  end;
  
  set(FIG_H,'WindowButtonMotionFcn','label_open_edges(''Drag'')',...
	    'WindowButtonUpFcn','label_open_edges(''ButtonUp'')');
  set(IMAGE_H,'ButtonDownFcn','');

  
% Done entering the current curve
elseif b == 2,
  set(LASTLINE_H,'color',[.75,.75,1],'linewidth',4);
  CURVEN = CURVEN + 1;
  LASTLINE_H = [];
  BOUNDX{CURVEN} = [];
  BOUNDY{CURVEN} = [];  
end;

% drag the mouse
function drag

global EDGEI RGBI DISTEDGEI NNEDGEI;
global FIG_H AXIS_H IMAGE_H BOUNDX BOUNDY CURVEN LASTLINE_H;

% which mouse button was pushed?
selection_type = get(FIG_H,'SelectionType');
if strcmp(selection_type,'normal'),
  b = 1;
elseif strcmp(selection_type,'extend'),
  b = 2;
elseif strcmp(selection_type,'alt'),
  b = 3;
end;

[imageHandle,x,y] = over_image(AXIS_H);

if imageHandle ~= 0 & imageHandle == IMAGE_H & (b == 1 | b == 3),
  % left mouse button: input a new point
  if b == 1,
    % set control point to the current mouse location
    BOUNDX{CURVEN} = [BOUNDX{CURVEN} ; x]; 
    BOUNDY{CURVEN} = [BOUNDY{CURVEN} ; y];
  elseif b == 3,
    % right mouse button: input a new point on an edge    
    [yedge,xedge] = ind2sub(size(EDGEI),NNEDGEI(round(y),round(x)));
    BOUNDX{CURVEN} = [BOUNDX{CURVEN} ; xedge];
    BOUNDY{CURVEN} = [BOUNDY{CURVEN} ; yedge];
  end;
  set(LASTLINE_H,'XData',BOUNDX{CURVEN},'YData',BOUNDY{CURVEN});
end;

% mouse button up
function buttonup

global EDGEI RGBI DISTEDGEI NNEDGEI;
global IMAGE_H AXIS_H FIG_H BOUNDX BOUNDY CURVEN LASTLINE_H;
selection_type = get(FIG_H,'SelectionType');
if strcmp(selection_type,'normal'),
  b = 1;
elseif strcmp(selection_type,'extend'),
  b = 2;
elseif strcmp(selection_type,'alt'),
  b = 3;
end;

[imageHandle,x,y] = over_image(AXIS_H);
if imageHandle ~= 0,
  if imageHandle == IMAGE_H,
    if b == 2, return;
    elseif b == 1,
      % set control point to the current mouse location
      BOUNDX{CURVEN} = [BOUNDX{CURVEN} ; x]; 
      BOUNDY{CURVEN} = [BOUNDY{CURVEN} ; y];
    elseif b == 3,
      % right mouse button: input a new point on an edge    
      [yedge,xedge] = ind2sub(size(EDGEI),NNEDGEI(round(y),round(x)));
      BOUNDX{CURVEN} = [BOUNDX{CURVEN} ; xedge];
      BOUNDY{CURVEN} = [BOUNDY{CURVEN} ; yedge];
    end;
    set(LASTLINE_H,'XData',BOUNDX{CURVEN},'YData',BOUNDY{CURVEN});
  end;
end;

set(FIG_H,'WindowButtonUpFcn','',...
	  'WindowButtonMotionFcn','label_open_edges(''Move'')');
set(IMAGE_H,'ButtonDownFcn','label_open_edges(''ButtonDown'')');

% key was pressed
function keypress

global IMAGE_H AXIS_H FIG_H;

CONTROLZ = 26;
CONTROLQ = 17;
ENTER = 13;

c = get(FIG_H,'CurrentCharacter');

if c == 'h' | c == 'H',
  le_help;
elseif c == CONTROLZ,
  undo;
elseif c == 'z',
  zoomout;
elseif c == 'Z',
  zoomin;
elseif c == '-',
  decreasethresh1;
elseif c == '=',
  increasethresh1;
elseif c == '_',
  decreasethresh2;
elseif c == '+',
  increasethresh2;
elseif c == CONTROLQ,
  finish;
elseif c == ENTER,
  closecurve;
elseif c == 't',
  toggleedgeimage;
else,
end;

function le_help

fprintf('label_open_edges commands:\n');
fprintf('LEFT CLICK with the mouse to add a new point.\n');
fprintf('RIGHT CLICK with the mouse to add a new point on an image edge.\n');
fprintf('MIDDLE CLICK with the mouse to finish the curve.\n');
fprintf('Press ENTER to close and then finish the curve.\n'),
fprintf('Press <CONTROL> z to undo the last curve piece added.\n');
fprintf('Press H for a list of commands.\n');
fprintf('Press z to zoom out.\n');
fprintf('Press Z to put in zoomin mode, then draw a rectangle to zoomin on.\n');
fprintf('Press - to decrease the first edge threshold.\n');
fprintf('Press = to increase the first edge threshold.\n');
fprintf('Press _ to decrease the second edge threshold.\n');
fprintf('Press + to increase the second edge threshold.\n');
fprintf('Press t to see the image with/without the edges.\n');
fprintf('Press <CONTROL> q to quit.\n');

% undo the last curve piece entered
function undo

global BOUNDX BOUNDY CURVEN CURVEPIECESTART LASTLINE_H;
if ~isempty(CURVEPIECESTART) & ~isempty(BOUNDX),
  BOUNDX{CURVEN} = BOUNDX{CURVEN}(1:CURVEPIECESTART(end)-1);
  BOUNDY{CURVEN} = BOUNDY{CURVEN}(1:CURVEPIECESTART(end)-1);
  CURVEPIECESTART = CURVEPIECESTART(1:end-1);
  set(LASTLINE_H,'XData',BOUNDX{CURVEN},'YData',BOUNDY{CURVEN});
  drawnow;
elseif isempty(CURVEPIECESTART) & length(BOUNDX) >= 2,
  BOUNDX = BOUNDX(1:CURVEN-1);
  BOUNDY = BOUNDY(1:CURVEN-1);
else,
  fprintf('Cannot undo -- no information left.\n');
end;

% zoom out so that the whole picture is in view
function zoomout

global AXIS_H NROWS NCOLS;

axes(AXIS_H);
axis([.5,NCOLS+.5,.5,NROWS+.5]);

% zoom in on a rectangle
function zoomin

global AXIS_H;
p = getrect(AXIS_H);
axis([p(1)-.5,p(3)+p(1)+.5,p(2)-.5,p(4)+p(2)+.5]);

function decreasethresh2

global IMAGE_H NROWS NCOLS;
global EDGEI RGBI RGBI0 DISTEDGEI NNEDGEI THRESH;

THRESH(2) = THRESH(2) *.95;
if THRESH(2) <= THRESH(1),
  THRESH(1) = THRESH(2) - eps;
end;
fprintf('Threshold is now [%f,%f]\n',THRESH);
% find edges in the image
EDGEI = edge(rgb2gray(RGBI0),'canny',THRESH);

% draw the edges on the image
ncolors = size(RGBI0,3);
RGBI = reshape(RGBI0,[NROWS*NCOLS],ncolors);
RGBI(EDGEI,1) = 1; RGBI(EDGEI,2) = .75; RGBI(EDGEI,3) = 1;
RGBI = reshape(RGBI,[NROWS,NCOLS,ncolors]);
set(IMAGE_H,'CData',RGBI);

% find the closest edge to each pixel
[DISTEDGE,NNEDGEI] = bwdist(EDGEI);

function decreasethresh1

global IMAGE_H NROWS NCOLS;
global EDGEI RGBI RGBI0 DISTEDGEI NNEDGEI THRESH;

THRESH(1) = THRESH(1) *.95;
if THRESH(1) == 0,
  THRESH(1) = 10^(-5);
end;
fprintf('Threshold is now [%f,%f]\n',THRESH);
% find edges in the image
EDGEI = edge(rgb2gray(RGBI0),'canny',THRESH);

% draw the edges on the image
ncolors = size(RGBI0,3);
RGBI = reshape(RGBI0,[NROWS*NCOLS],ncolors);
RGBI(EDGEI,1) = 1; RGBI(EDGEI,2) = .75; RGBI(EDGEI,3) = 1;
RGBI = reshape(RGBI,[NROWS,NCOLS,ncolors]);
set(IMAGE_H,'CData',RGBI);

% find the closest edge to each pixel
[DISTEDGE,NNEDGEI] = bwdist(EDGEI);

function increasethresh2

global IMAGE_H NROWS NCOLS;
global EDGEI RGBI RGBI0 DISTEDGEI NNEDGEI THRESH;

THRESH(2) = THRESH(2) /.95;

% find edges in the image
EDGEIt = edge(rgb2gray(RGBI0),'canny',THRESH);

if ~any(EDGEI(:)), 
  THRESH(2) = THRESH(2) * .95;
else, 
  EDGEI = EDGEIt;
  fprintf('Threshold is now %f %f\n',THRESH);

  % draw the edges on the image
  ncolors = size(RGBI0,3);
  RGBI = reshape(RGBI0,[NROWS*NCOLS],ncolors);
  RGBI(EDGEI,1) = 1; RGBI(EDGEI,2) = .75; RGBI(EDGEI,3) = 1;
  RGBI = reshape(RGBI,[NROWS,NCOLS,ncolors]);
  set(IMAGE_H,'CData',RGBI);
  
  % find the closest edge to each pixel
  [DISTEDGE,NNEDGEI] = bwdist(EDGEI);
end;

function increasethresh1

global IMAGE_H NROWS NCOLS;
global EDGEI RGBI RGBI0 DISTEDGEI NNEDGEI THRESH;

THRESH(1) = THRESH(1) /.95;
if THRESH(1) >= THRESH(2),
  THRESH(2) = THRESH(1) + eps;
end;
fprintf('Threshold is now [%f,%f]\n',THRESH);
% find edges in the image
EDGEI = edge(rgb2gray(RGBI0),'canny',THRESH);

% draw the edges on the image
ncolors = size(RGBI0,3);
RGBI = reshape(RGBI0,[NROWS*NCOLS],ncolors);
RGBI(EDGEI,1) = 1; RGBI(EDGEI,2) = .75; RGBI(EDGEI,3) = 1;
RGBI = reshape(RGBI,[NROWS,NCOLS,ncolors]);
set(IMAGE_H,'CData',RGBI);

% find the closest edge to each pixel
[DISTEDGE,NNEDGEI] = bwdist(EDGEI);

function closecurve

global BOUNDX BOUNDY CURVEN LASTLINE_H;

if isempty(BOUNDX{CURVEN}), return; end;
%BOUNDX{CURVEN} = [BOUNDX{CURVEN};BOUNDX{CURVEN}(1)];
%BOUNDY{CURVEN} = [BOUNDY{CURVEN};BOUNDY{CURVEN}(1)];
set(LASTLINE_H,'XData',BOUNDX{CURVEN},'YData',BOUNDY{CURVEN}),
set(LASTLINE_H,'color',[.75,.75,1],'linewidth',4);
CURVEN = CURVEN + 1;
LASTLINE_H = [];
BOUNDX{CURVEN} = [];
BOUNDY{CURVEN} = [];  

function finish

global IMAGE_H FIG_H;
set(FIG_H,'WindowButtonMotionFcn','');
set(IMAGE_H,'ButtonDownFcn','');
tmp = IMAGE_H;
clear global FIG_H AXIS_H IMAGE_H NROWS NCOLS;
clear global CURVEN LASTLINE_H CURVEPIECESTART;
clear global EDGEI RGBI RGBI0 DISTEDGEI NNEDGEI THRESH ISEDGESHOWN;

set(tmp,'UserData','Done');

function [imageHandle,x,y] = over_image(axesHandle)

% Return the index of which image we are over, and return a 0 if we
% aren't above an image.

imagefound = findobj(axesHandle, 'type', 'image');
if isempty(imagefound)
   imageHandle=0; x=0; y=0;
   return
end
% Make sure that the Image's Button Down & Up functions will queue
set(imagefound, 'Interruptible', 'off', 'BusyAction', 'Queue');
axHandle = get(imagefound, 'Parent');
axPosition = get(axHandle, 'Position');
axCurPt = get(axHandle, 'CurrentPoint');

% See if we are above the desired axes
imageHandle = 0;  
XLim = get(axHandle, 'XLim');
YLim = get(axHandle, 'YLim');
pt = axCurPt;
x = pt(1,1); y = pt(1,2);
if x>=XLim(1) & x<=XLim(2) & y>=YLim(1) & y<=YLim(2)
  imageHandle = imagefound;
else,
  x = 0; y = 0;
end;

function toggleedgeimage

global RGBI;
global RGBI0;
global ISEDGESHOWN;
global IMAGE_H;
if ISEDGESHOWN == 1,
  set(IMAGE_H,'CData',RGBI0);
  ISEDGESHOWN = 0;
else,
  set(IMAGE_H,'CData',RGBI);
  ISEDGESHOWN = 1;
end;
