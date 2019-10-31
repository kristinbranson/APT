function w = getWorkArea
%getWorkArea Returns information about the usable onscreen working area.
%   s = getWorkArea returns a structure containing values in pixel units 
%   for height, width and the left  and bottom borders
%   of the screen's working area. The left,bottom, width and  height values
%   are the same as Matlab's ScreenSize values if there  are no screen
%   decorations.
%   
%   Working area is defined to be that area of the screen not obscured by
%   the Windows task bar or other OS 'decorations'.
%   The function uses MJUtilities.getScreenBounds which:
%   "Returns the usable screen rectangle, taking into account OS
%   decorations such as the Windows task bar." 
%     
%   If java is not available, the function returns a rectangle 
%   of the screen size 
%   e.g. if the screen resolution is 1280X1024, the rectangle will be:
%   left = 1, right = 1280; bottom= 1, top=1024
%
%  if the ScreenSize property is 0 or less, the screen size defaults to 
%  1024x768

%   Copyright 2004-2010 The MathWorks, Inc.
%   $Revision: 1.1.8.3 $  $Date: 2010/11/17 11:24:03 $

ss = get(0, 'ScreenSize');
screen_width = ss(3);
screen_height = ss(4);

% if screensize property is 1 or less, default
if screen_width <= 1
  screen_width = 1024;
end
if screen_height <= 1
  screen_height = 768;
end

import com.mathworks.mwswing.MJUtilities;
 
if internal.images.isFigureAvailable()
    r = MJUtilities.getScreenBounds;
    % move origin from top left to bottom left,
    % adjust coordinates from OS style origin to  Matlab style origin 
    %e.g. from (0,0,1280,1024) to  (1,1,1280,1024)
    w.left = r.x+1;
    w.bottom = screen_height - (r.y + r.height) +1;
    w.width = r.width;
    w.height = r.height;
    w.right = r.x + r.width;   
    w.top =  w.bottom + r.height - 1;
else
    % default to screen size 
    w.left = 1;
    w.bottom = 1;
    w.width = screen_width;
    w.height = screen_height;
    w.right = screen_width;
    w.top = screen_height;
end
