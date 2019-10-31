function windowlevel(hImage, hCaller)
%WINDOWLEVEL Interactive Window/Level adjustment.
%   WINDOWLEVEL(HIMAGE, HCALLER) activates window/level interactivity on 
%   the target image. Clicking and dragging the mouse on the target image 
%   changes the image's window values. Dragging the mouse horizontally from
%   left to right changes the window width (i.e., contrast). Dragging the
%   mouse vertically up and down changes the window center (i.e.,
%   brightness). Holding down the CTRL key when clicking accelerates
%   changes. Holding down the SHIFT key slows the rate of change. Keys must
%   be pressed before clicking and dragging.
%   
%   HIMAGE is the handle to the target image. HCALLER is the handle to an
%   object that activates the WINDOWLEVEL behavior.  For instance, this may
%   be a button on another figure that turns on the WINDOWLEVEL behavior.
%   The windowlevel behavior is deactivated when HCALLER is deleted. 

%   Copyright 2005 The MathWorks, Inc.
%   $Revision: 1.1.6.4 $  $Date: 2011/07/19 23:57:47 $

% Get handles to the parent axes and figure
[hIm, hImAx, hImFig] = imhandles(hImage);

isDoubleData = strcmpi('double',class(get(hIm,'CData')));
histStruct = getHistogramData(hIm);
origCLim = histStruct.histRange;

% Define variables for function scope
cbidMotion = [];
cbidUp = [];
cbidDelFcn = [];
lastPointerPos = [];
WLSpeed = [];
wlMotionScale = getWLMotionScale(hIm);
newCLim = get(hImAx,'CLim');

if isDoubleData && (origCLim(1)>=0 && origCLim(2)<=1)
    valueFormatter = @(x) x;
else
    valueFormatter = @round;
end
    
% Start windowlevel action
wldown();

    %======================================================================
    function cancelWindowLevel(obj,evt)
        wlup();
    end %cancelWindowLevel

    %======================================================================
    function wldown(varargin)
        
        % Set the mouse event functions.
        cbidMotion = iptaddcallback(hImFig, 'WindowButtonMotionFcn', @wlmove);
        cbidUp = iptaddcallback(hImFig, 'WindowButtonUpFcn', @wlup);

        if nargin >= 3 && ~isempty(hCaller)
            % This prevents the windowlevel from functioning in the event that the
            % calling tool should close.  And also ensures that the appropriate
            % callbacks are detached from the image object's figure.
            cbidDelFcn = iptaddcallback(hCaller, 'DeleteFcn', @cancelWindowLevel);
        else
            hCaller = hImFig;
        end

        % Keep track of values needed to adjust window/level.
        lastPointerPos = get(hImFig, 'CurrentPoint');

        % Figure out how quickly to change the window/level based on key
        % modifiers.
        WLSpeed = getWLSpeed(hImFig) * wlMotionScale;

    end % wldown


    %======================================================================
    function wlup(varargin)

        % Stop tracking mouse motion and button up.
        iptremovecallback(hImFig, 'WindowButtonMotionFcn', cbidMotion);
        iptremovecallback(hImFig, 'WindowButtonUpFcn', cbidUp);
        iptremovecallback(hCaller, 'DeleteFcn', cbidDelFcn);
        
        % This is done so that the new clim is registered after the
        % figure's WindowButonUpFcn is called.
        set(hImAx, 'CLim', newCLim);
        
    end %wlup


    %======================================================================
    function wlmove(varargin)

        % Find out where the pointer has moved to.
        currentPos = get(hImFig, 'CurrentPoint');
        offset = currentPos - lastPointerPos;
        lastPointerPos = currentPos;

        % Determine the 
        % Get previous W/L.
        [windowWidth, windowCenter] = computeWindow(get(hImAx, 'CLim'));

        % Compute new window/level values and CLim endpoints.
        windowWidth = windowWidth + WLSpeed * offset(1);    % Contrast
        windowCenter = windowCenter + WLSpeed * offset(2);  % Brightness

        windowWidth = max(windowWidth, wlMotionScale);
        newCLim = zeros(1,2);
        [newCLim(1), newCLim(2)] = computeCLim(windowWidth, windowCenter);
                
        % Prevent endpoints from extending outside the bounds of the 
        % original CLim           
        newCLim(1) = max(newCLim(1), origCLim(1));
        newCLim(2) = min(newCLim(2), origCLim(2));
        
        % Ensure that the new CLim is increasing i.e. clim(1) < clim(2)
        if (~isDoubleData && ((newCLim(2)-1) < newCLim(1)))
          newCLim = get(hImAx, 'CLim');
        elseif (isDoubleData && (newCLim(2)<=newCLim(1)))
          newCLim = get(hImAx, 'CLim');
        end
      
        newCLim = valueFormatter(newCLim);        
        
        % Change the axes CLim
        set(hImAx, 'CLim', newCLim);

    end % wlmove

end % windowlevel

%==========================================================================
function [minPixel, maxPixel] = computeCLim(width, center)
%FINDWINDOWENDPOINTS   Process window and level values.

minPixel = (center - width/2);
maxPixel = minPixel + width;

end

%==========================================================================
function [width, center] = computeWindow(CLim)

width = CLim(2) - CLim(1);
center = CLim(1) + width ./ 2;

end

%==========================================================================
function scale = getWLMotionScale(hIm)

X = get(hIm, 'CData');

xMin = min(X(:));
xMax = max(X(:));

% Compute Histogram for the image.
switch (class(X))
 case 'uint8'
  scale = 1;
  
 case {'int16', 'uint16'}
  
  scale = 4;
  
 case {'double'}
  
  % Images with double CData often don't work well with IMHIST.
  % Convert all images to be in the range [0,1] and convert back
  % later if necessary.
  if (xMin >= 0) && (xMax <= 1)
    
    scale = 1/255;
  else
    
    if ((xMax - xMin) > 1023)
      
      scale = 4;
      
    elseif ((xMax - xMin) > 255)
      
      scale = 2;
      
    else
      
      scale = 1;
      
    end
  end   
 otherwise
  
  error(message('images:windowlevel:classNotSupported'))
  
end
end

%==========================================================================
function speed = getWLSpeed(hFig)

SelectionType = lower(get(hFig, 'SelectionType'));
%disp(SelectionType)

switch (SelectionType)
    case {'normal', 'open'}
        speed = 1;

    case 'extend'
        speed = 0.5;

    case {'alternate', 'alt'}
        speed = 2;

end
end
