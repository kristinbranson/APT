function imzoomin(obj,varargin) %#ok varargin needed by HG caller
%IMZOOMIN Interactive zoom in on scrollpanel.

%   Copyright 2004-2006 The MathWorks, Inc.
%   $Revision: 1.1.8.6 $  $Date: 2006/03/13 19:44:17 $

  hIm = obj;
  hScrollpanel = checkimscrollpanel(hIm,mfilename,'HIMAGE');
  apiScrollpanel = iptgetapi(hScrollpanel);

  hAx = get(obj,'parent');
  hFig = ancestor(obj, 'figure');
  
  [x, y] = getCurrentPoint(hAx);
  
  selectionType = get(hFig,'SelectionType');
  singleClick = strcmp(selectionType, 'normal');
  doubleClick = strcmp(selectionType, 'open');  
  altKeyPressed = strcmpi(get(hFig,'CurrentModifier'),'alt');

  % initialized for function scope
  prevButtonUpFcn = [];

  if singleClick
    
    handleSingleClick();
    
  elseif doubleClick
    apiScrollpanel.setMagnification(apiScrollpanel.findFitMag())
  end
    
  %----------------------------------------------------------
  function handleSingleClick()

    prevButtonUpFcn = get(hFig,'WindowButtonUpFcn');
    set(hFig,'WindowButtonUpFcn',@buttonReleased);

    % Disable the figure's pointer manager before calling rbbox. Otherwise
    % the pointer may change when it passes over objects containing pointer
    % behaviors. 
    iptPointerManager(hFig, 'disable');

    % Only using rbbox as a graphical affordance because rbbox is fraught with
    % problems.
    tmpRect = rbbox;
    
    % Enable the pointer manager.
    iptPointerManager(hFig, 'enable');

  end %handleSingleClick
  
  %----------------------------------------------------------  
  function buttonReleased(src,varargin) %#ok vargin needed by HG caller
      
      if altKeyPressed
          zoomOnAltClick();
      else
          % We are mimicking the behavior of rbbox.  We could not use it 
          % because it called some sort of drawnow which causes the 
          % ButtonUp Event to be fired before handleSingleClick has 
          % finished executing.  See gecks.
          
          [x2, y2] = getCurrentPoint(hAx);
          
          % Must get the current magnification and scale the rect 
          % so that it is in the right coordinates for methods on
          % apiScrollpanel.
          currentMag = apiScrollpanel.getMagnification();
          zoomRect = [0 0 abs(x2-x)*currentMag abs(y2-y)*currentMag];
          
          % This constant specifies the number of pixels the mouse
          % must move in order to do a rbbox zoom.
          % Note: same value as matlab/graphics/@graphics/@zoom/buttonupfcn2D.m
          maxPixels = 5; 
      
          tinyWidthOrHeight = any(zoomRect(3:4) < maxPixels);
          
          if tinyWidthOrHeight
              zoomOnClick();
              
          else
              [x2,y2] = getCurrentPoint(hAx);
              
              midPoint = [x+x2, y+y2] * 0.5;
              zoomOnDragRect(midPoint,zoomRect(3),zoomRect(4))      
          end
          
      end % if altKeyPressed
      
      set(hFig,'WindowButtonUpFcn',prevButtonUpFcn);
    
  end
  
  %----------------------------------------------------------
  function zoomOnDragRect(rectCenter,rectWidth,rectHeight)

    currentMag = apiScrollpanel.getMagnification();
    
    rectWidthImcoords = rectWidth / currentMag;
    rectHeightImcoords = rectHeight / currentMag;
    
    mag = apiScrollpanel.findMagnification(rectWidthImcoords,...
                                           rectHeightImcoords);
    
    apiScrollpanel.setMagnificationAndCenter(mag, rectCenter(1),rectCenter(2));
    
  end 
  
  %----------------------------------------------------------  
  function zoomOnClick
    newMag = findZoomMag('in',apiScrollpanel.getMagnification());
    apiScrollpanel.setMagnificationAndCenter(newMag,x,y)   
  end

  %----------------------------------------------------------
  function zoomOnAltClick
  % If the Alt key is pressed, zoom-out is performed
    newMag = findZoomMag('out',apiScrollpanel.getMagnification());
    apiScrollpanel.setMagnificationAndCenter(newMag,x,y) 
  end

end %imzoomin
