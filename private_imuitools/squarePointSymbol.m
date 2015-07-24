function draw_API = squarePointSymbol
%squarePointSymbol Creates renderer for square point symbol.  
%   DRAW_API = squarePointSymbol creates a DRAW_API for use in association with
%   IMPOINT that draws square points with wings. DRAW_API is a structure of
%   function handles that are used by IMPOINT to draw the point and update its
%   properties.
%
%       DRAW_API.initialize(h_group)
%       DRAW_API.setColor(color)
%       DRAW_API.updateView(position)  
%       DRAW_API.setString(string)   
%       DRAW_API.getBoundingBox() 
%       DRAW_API.clear()

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.3 $  $Date: 2008/12/22 23:48:06 $

  % initialize variables needing function scope
  [bounding_box, h_group, h_square_line, h_dot, h_text, h_patch] = deal([]);
  
  draw_API.initialize       = @initialize;  
  draw_API.setColor         = @setColor;
  draw_API.updateView       = @updateView;  
  draw_API.setString        = @setString;  
  draw_API.getBoundingBox   = @getBoundingBox;
  draw_API.clear            = @clear;
    
  %----------------------------
  function pos = getBoundingBox
    pos = bounding_box;
  end
  
  %---------------------------
  function initialize(h)

    h_group = h;
    
    % The line objects should have a width of one screen pixel.
    line_width = ceil(getPointsPerScreenPixel);
    h_square_line = line(...
                         'LineStyle', '-', ...
                         'LineWidth', line_width, ...
                         'HitTest', 'off', ...
                         'Parent', h_group);
  
    h_dot         = line(...
                         'LineStyle', '-', ...                       
                         'LineWidth', line_width, ...
                         'HitTest', 'off', ...
                         'Parent', h_group);
  
    h_text        = text(...,
                         'FontName','SansSerif',...
                         'FontSize',8,...
                         'HorizontalAlignment','left',...
                         'VerticalAlignment','top',...
                         'Clipping','on',...
                         'HitTest', 'off', ...
                         'Parent', h_group);
    
    h_patch = patch('FaceColor', 'none', 'EdgeColor', 'none', ...
                    'HitTest', 'off', ...
                    'Parent', h_group);
        
  end

  %-------------
  function clear

    delete([h_square_line h_dot h_text h_patch])
    
  end

  %-------------------
  function setColor(c)
    if ishghandle(h_square_line)
      set([h_square_line h_dot h_text], 'Color', c);
    end
  end

  %----------------------------
  function updateView(position)
    
    pos_x = position(1);
    pos_y = position(2);
  
    if ~ishghandle(h_group)
        return;
    end
    
    h_axes = ancestor(h_group, 'axes');
  
    [dx_per_screen_pixel, dy_per_screen_pixel] = getAxesScale(h_axes);

    rect_side_pixels = 4;
    wing_size_pixels = 3;

    x_mid = pos_x / dx_per_screen_pixel;
    x_left = x_mid - rect_side_pixels;
    x_right = x_mid + rect_side_pixels;
    x_wing_size = wing_size_pixels;

    % Note that y_min and y_max are reversed in terms of how the shape is drawn if
    % get(h_axes,'YDir') is 'reverse'.
    y_mid = pos_y / dy_per_screen_pixel;
    y_min = y_mid - rect_side_pixels;
    y_max = y_mid + rect_side_pixels;
    y_wing_size = wing_size_pixels;

    x1 = x_left - x_wing_size;
    x2 = x_left;
    x3 = x_mid;
    x4 = x_right;
    x5 = x_right + x_wing_size;
    
    y1 = y_min - y_wing_size;
    y2 = y_min;
    y3 = y_mid;
    y4 = y_max;
    y5 = y_max + y_wing_size;

    % (x,y) is a polygon that strokes the square line.  Here it is in
    % screen pixel units.
    x = [x1 x2 x2 x3 x3 x3 x4 x4 x5 x4 x4 x3 x3 x3 x2 x2 x1];
    y = [y3 y3 y2 y2 y1 y2 y2 y3 y3 y3 y4 y4 y5 y4 y4 y3 y3];
    
    % Convert the (x,y) polygon back to data units.
    [x,y] = pixel2DataUnits(h_axes,x,y);
    
    % Set the bounding box to include the entire extent of the drawn
    % rectangle, including decorations.
    bounding_box = findBoundingBox(x,y);
    
    if ~isequal(get(h_square_line, 'XData'), x) || ...
       ~isequal(get(h_square_line, 'YData'), y)
      set(h_square_line, 'XData', x, 'YData', y);
      set(h_dot,'XData',pos_x,'YData',pos_y);
      set(h_patch, 'XData', x, 'YData', y);      
    end

    % Figure out text position depending on axes 'YDir'
    x_right = x_right * dx_per_screen_pixel;
    if strcmp(get(h_axes,'YDir'),'reverse')
      y_max = y_max * dy_per_screen_pixel;
      text_pos = [x_right y_max];
    else
      y_min = y_min * dy_per_screen_pixel;
      text_pos = [x_right y_min];        
    end
    set(h_text,'Position',text_pos);
    
  end
  
  %--------------------
  function setString(s)
    if ~isempty(s)
      set(h_text,'String',s);
    end
  end
  
end
