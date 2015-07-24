function hout = imgridlines(h)
%IMGRIDLINES Superimpose pixel grid lines on a displayed image.
%   IMGRIDLINES adds dashed gray lines, forming a grid outlining each
%   pixel, to the current axes.  The current axes must contain an image. 
%
%   IMGRIDLINES(IMAGE_HANDLE) adds the lines to the axes containing the
%   specified image handle.  IMGRIDLINES(AXES_HANDLE) or
%   IMGRIDLINES(FIGURE_HANDLE) adds lines for the first image object
%   found within the specified axes or figure object.
%
%   HGROUP = IMGRIDLINES(...) returns a handle to the HGGROUP object that
%   contains the line objects comprising the pixel grid.

%   Copyright 1993-2010 The MathWorks, Inc.
%   $Revision: 1.1.8.11 $  $Date: 2011/07/19 23:57:43 $

if nargin < 1
    hAx  = get(get(0,'CurrentFigure'), 'CurrentAxes');
    hIm = findobj(hAx, 'Type', 'image');
    if isempty(hIm)
        error(message('images:imgridlines:noImage'));
    else
        hIm = hIm(1);
    end
else
    if ~ishghandle(h)
        error(message('images:imgridlines:notHandle'));
    end
    switch get(h, 'Type')
      case 'figure'
        hIm = findobj(get(h,'CurrentAxes'), 'Type', 'image');
        
      case 'axes'
        hIm = findobj(h, 'Type', 'image');
        
      case 'image'
        hIm = h;
        
      otherwise
        error(message('images:imgridlines:wrongTypeHandle'));
    end
    
    if isempty(hIm)
        error(message('images:imgridlines:noImage'));
    end
    
    hIm = hIm(1);
end

ax_handle = handle(ancestor(hIm, 'axes'));

% if the user has changed the default line width there is a very severe
% java performance penalty
factory_line_width = get(0,'factoryLineLineWidth');

hgrp = hggroup('Parent', ax_handle, ...
    'Hittest', 'off', ...
    'tag','imgridlines hgrp');

h_vertical_solid = line('LineStyle', '-', ...
    'LineWidth',factory_line_width,...
    'Color', [.5 .5 .5], ...
    'Hittest', 'off', ...
    'Parent', hgrp); %#ok<NASGU>

h_vertical_dotted = line('LineStyle', ':', ...
    'LineWidth',factory_line_width,...
    'Color', [.65 .65 .65], ...
    'Hittest', 'off', ...
    'Parent', hgrp); %#ok<NASGU>

h_horizontal_solid = line('LineStyle', '-', ...
    'LineWidth',factory_line_width,...
    'Color', [.5 .5 .5], ...
    'Hittest', 'off', ...
    'Parent', hgrp); %#ok<NASGU>

h_horizontal_dotted = line('LineStyle', ':', ...
    'LineWidth',factory_line_width,...
    'Color', [.65 .65 .65], ...
    'Hittest', 'off', ...
    'Parent', hgrp); %#ok<NASGU>

updateXYData();

updateFunction = @updateXYData;

xdata_listener = iptui.iptaddlistener(hIm, ... 
    'XData', ...
    'PostSet', ...
    updateFunction);
ydata_listener = iptui.iptaddlistener(hIm, ...
    'YData', ...
    'PostSet', ...
    updateFunction);

setappdata(hgrp, 'Listeners', [xdata_listener ydata_listener]);
% clear unused references to listeners
clear xdata_listener ydata_listener;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function updateXYData(varargin)
  
    im_xdata = get(hIm, 'XData');
    im_ydata = get(hIm, 'YData');
    
    M = size(get(hIm, 'CData'), 1);
    N = size(get(hIm, 'CData'), 2);
    
    % if we get a scalar XData or YData we compute the 2nd element based on
    % one unit per image pixel in that dimension (see the doc for the Image
    % object and the XData/YData property descriptions for more details)
    if isscalar(im_xdata)
        im_xdata(2) = im_xdata(1) + (N-1);
    end
    if isscalar(im_ydata)
        im_ydata(2) = im_ydata(1) + (M-1);
    end
    
    kids = get(hgrp, 'Children');
    
    if N <= 1
        x_per_pixel = 1;
    else
        x_per_pixel = diff(im_xdata) / (N - 1);
    end
    
    if M <= 1
        y_per_pixel = 1;
    else
        y_per_pixel = diff(im_ydata) / (M - 1);
    end
    
    x1 = im_xdata(1) - (x_per_pixel/2);
    x2 = im_xdata(2) + (x_per_pixel/2);
    y1 = im_ydata(1) - (y_per_pixel/2);
    y2 = im_ydata(2) + (y_per_pixel/2);
    
    x = linspace(x1, x2, N+1);
    xx = zeros(1, 2*length(x));
    xx(1:2:end) = x;
    xx(2:2:end) = x;
    yy = zeros(1, length(xx));
    yy(1:4:end) = y1;
    yy(2:4:end) = y2;
    yy(3:4:end) = y2;
    yy(4:4:end) = y1;
    
    set(kids(1), 'XData', xx, 'YData', yy);
    set(kids(2), 'XData', xx, 'YData', yy);
    
    y = linspace(y1, y2, M+1);
    yy = zeros(1, 2*length(y));
    yy(1:2:end) = y;
    yy(2:2:end) = y;
    xx = zeros(1, length(yy));
    xx(1:4:end) = x1;
    xx(2:4:end) = x2;
    xx(3:4:end) = x2;
    xx(4:4:end) = x1;
    
    set(kids(3), 'XData', xx, 'YData', yy);
    set(kids(4), 'XData', xx, 'YData', yy);
    
  end

if nargout > 0
    hout = hgrp;
end

end
