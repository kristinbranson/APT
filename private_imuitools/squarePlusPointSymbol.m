function draw_API = squarePlusPointSymbol
%squarePlisPointSymbol Creates renderer for square-plus point symbol.  
%   DRAW_API = squarePlusPointSymbol creates a DRAW_API for use in association
%   with IMPOINT that draws square points with wings. DRAW_API is a structure of
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
%   $Revision: 1.1.6.3 $  $Date: 2008/12/22 23:48:05 $
  
  % initialize variables needing function scope
  [bounding_box, h_group, h_square, h_plus, h_text] = deal([]);

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
    
    h_square      = line(...
                         'Marker', 'square', ...
                         'HitTest', 'off', ...
                         'Parent', h_group);
    
    h_plus        = line(...
                         'Marker', '+', ...                       
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

  end

  %-------------
  function clear

    delete([h_square h_plus h_text])
    
  end

  %-------------------------
  function setActive(active)

    is_active = active;
    
  end
  
  %-------------------
  function setColor(c)
    if ishghandle(h_group) 
      set([h_square h_plus h_text], 'Color', c);
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
    active_rect_side = 10;
    
    points_per_screen_pixel = getPointsPerScreenPixel;
    rect_marker_size_points = 2*rect_side_pixels*points_per_screen_pixel;    
    plus_marker_size_points = rect_marker_size_points + ...
        2*wing_size_pixels*points_per_screen_pixel;        
    set(h_square,'MarkerSize',rect_marker_size_points)
    set(h_plus,'MarkerSize',plus_marker_size_points)

    x_mid = pos_x / dx_per_screen_pixel;
    x_left = x_mid - rect_side_pixels;
    x_right = x_mid + rect_side_pixels;

    % Note that y_min and y_max are reversed in terms of how the shape 
    % is drawn if get(h_axes,'YDir') is 'reverse'.
    y_mid = pos_y / dy_per_screen_pixel;
    y_min = y_mid - rect_side_pixels;
    y_max = y_mid + rect_side_pixels;

    % Convert the x and y extrema to data units.
    [x,y] = pixel2DataUnits(h_axes,[x_left x_right],[y_min y_max]);
  
    % Set the bounding box to include the entire extent of the drawn
    % rectangle, including decorations.
    bounding_box = findBoundingBox(x,y);
    
    if ~isequal(get(h_square, 'XData'), pos_x) || ...
       ~isequal(get(h_square, 'YData'), pos_y)
      set([h_square h_plus], 'XData', pos_x, 'YData', pos_y);
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
