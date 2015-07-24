function initSize(im_handle,screen_per_image_pixel,isBorderTight)
%initSize Initialize size of axes and figure
%
%   initSize(imH,screenPerImagePixel,isBorderTight) adjusts the display 
%   size of an image by using the image size and the scale factor
%   screenPerImagePixel. If screenPerImagePixel==1,then the display has one
%   screen pixel for each image pixel. If isBorderTight is false, then
%   initSize adds gutters for displaying axes and tick labels.
% 
%   Note: The code assumes that it is calculating the size for a figure that
%   contains a single axes object with a single image. Other uicontrols
%   and uipanels are not taken into account.

%   Copyright 1993-2020 The MathWorks, Inc.
%   $Revision: 1.1.8.11 $  $Date: 2011/07/19 23:57:44 $

  ax_handle = ancestor(im_handle,'axes');
  fig_handle = ancestor(ax_handle,'figure');

  ax_units = get(ax_handle, 'Units');
  fig_units = get(fig_handle, 'Units');
  root_units = get(0, 'Units');
  
  image_width  = getImWidth(im_handle);
  image_height = getImHeight(im_handle); 
  
  if (image_width * image_height == 0)
      % Don't try to handle the degenerate case.
      return;
  end
     
  % Work in pixels
  set(ax_handle, 'Units', 'pixels');
  set(fig_handle, 'Units', 'pixels');
  set(0, 'Units', 'pixels');
  
  ax_pos = get(ax_handle, 'Position');
  fig_pos = get(fig_handle, 'Position');
 
  orig_fig_width  = fig_pos(3);
  orig_fig_height = fig_pos(4);

  % Declare so they're in function scope
  on_screen_image_width = [];
  on_screen_image_height = [];
  new_fig_width = [];
  new_fig_height = [];
  is_width_bigger_than_screen = false;
  is_height_bigger_than_screen = false;
  
  % get the size of the screen area available for display
  % excludes areas used by OS for taskbar, dock, etc.
  wa = getWorkArea;
  screen_width = wa.width;
  screen_height = wa.height;

  % get figure properties
  p = figparams;
  
  % to initialize dimensions
  calculateDimensions
  
  % adjust size until the figure fits on the screen
  warn_about_mag = false;
  while (is_width_bigger_than_screen || is_height_bigger_than_screen)
      screen_per_image_pixel = findZoomMag('out',screen_per_image_pixel);
      warn_about_mag = true;
      calculateDimensions % to update dimensions
  end
 
  if warn_about_mag
      warning(message('images:initSize:adjustingMag', round( screen_per_image_pixel*100 )));       
  end
  
  % Don't try to display a figure smaller than this:
  min_fig_width = 128; 
  min_fig_height = 128;
  new_fig_width  = max(new_fig_width, min_fig_width);
  new_fig_height = max(new_fig_height, min_fig_height);
  
  % Figure out where to place the axes object in the resized figure.
  ax_pos(1) = getAxesX;
  ax_pos(2) = getAxesY;
  ax_pos(3) = max(on_screen_image_width,1);
  ax_pos(4) = max(on_screen_image_height,1);
  
  % Calculate new figure position
  fig_pos(1) = max(1, fig_pos(1) - floor((new_fig_width - orig_fig_width)/2));
  fig_pos(2) = max(1, fig_pos(2) - floor((new_fig_height - orig_fig_height)/2));
  
  fig_pos(3) = new_fig_width;
  fig_pos(4) = new_fig_height;
  
  % Translate figure position if necessary, using size of work area,
  %  figure decoration sizes and figure position
  dx = (screen_width - p.RightDecoration) - (fig_pos(1) + fig_pos(3));
  if (dx < 0)
      fig_pos(1) = fig_pos(1) + dx;
  end
  dy = (screen_height - p.TopDecoration) - (fig_pos(2) + fig_pos(4));
  if (dy < 0)
      fig_pos(2) = fig_pos(2) + dy;
  end
  
  set(fig_handle, 'Position', fig_pos)
  set(ax_handle, 'Position', ax_pos);
  
  % Restore the units
  set(fig_handle, 'Units', fig_units);
  set(ax_handle, 'Units', ax_units);
  set(0, 'Units', root_units);
  
  constrainToWorkArea(fig_handle);
  
  %---------------------------
  function calculateDimensions
     on_screen_image_width = image_width * screen_per_image_pixel;
     on_screen_image_height = image_height * screen_per_image_pixel;

     new_fig_width  = on_screen_image_width  + getGutterWidth;
     new_fig_height = on_screen_image_height + getGutterHeight;   

     is_width_bigger_than_screen = ...
         (new_fig_width + p.horizontalDecorations) > screen_width;
     is_height_bigger_than_screen = ...
         (new_fig_height + p.verticalDecorations) > screen_height;
  end
  
  %--------------------------
  function w = getGutterWidth
  
     if isBorderTight
         w = 0;
     else
         w = p.looseBorderWidth;
     end
  
  end

  %---------------------------
  function h = getGutterHeight
  
     if isBorderTight
         h = 0;
     else
         h  = p.looseBorderHeight;
     end

  end

  %--------------------
  function x = getAxesX	  
	  
     if isBorderTight
		 x = 1;
		 % If the on screen image width is less than the new figure width,
		 % need to recenter the axes. This occurs for small images
		 % displayed with less than 128 pixels in width.
		 if new_fig_width > on_screen_image_width
                extra_width_in_pixels = new_fig_width - on_screen_image_width;
                x = extra_width_in_pixels / 2;
		 end
		 	 
     else
         x = p.YLabelWidth + 1;
     end
  
  end

  %--------------------
  function y = getAxesY
  
     if isBorderTight
         y = 1;
		 % If the on screen image height is less than the new figure height,
		 % need to recenter the axes. This occurs for small images
		 % displayed with less than 128 pixels in height. 
		 if new_fig_height > on_screen_image_height
                extra_height_in_pixels = new_fig_height - on_screen_image_height;
                y = extra_height_in_pixels / 2;
		 end
		 
     else
         y = p.XLabelHeight + 1;
     end
  
  end   
   
end % initSize

