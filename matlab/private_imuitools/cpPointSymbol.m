function draw_API = cpPointSymbol
%cpPointSymbol Creates renderer for control point symbol.  
%   DRAW_API = cpPointSymbol creates a DRAW_API for use in association with
%   IMPOINT that draws control points. DRAW_API is a structure of function
%   handles that are used by IMPOINT to draw the point and update its
%   properties.
%
%       DRAW_API.initialize(h_group)
%       DRAW_API.setColor(color)
%       DRAW_API.updateView(position)  
%       DRAW_API.setString(string)   
%       DRAW_API.getBoundingBox() 
%       DRAW_API.clear()
%       DRAW_API.setVisible()
%       DRAW_API.showActiveDecoration(show) % used for control point tasks  
%       DRAW_API.showPredictedDecoration(show) % used for control point tasks    

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.9 $  $Date: 2008/12/22 23:47:49 $
  
  % initialize variables needing function scope
  [bounding_box, h_group, h_axes, h_circle, h_plus, h_text,...
   h_active_circle, h_predicted_text, ...
   is_ydir_reverse, circle_diameter_pixels, active_circle_diameter,...
   half_delta_x_data_units, half_delta_y_data_units,...
   text_nudge_data_units] = deal([]);

  [show_predicted_decoration,show_active_decoration] = deal(false);
  
  draw_API.initialize              = @initialize;
  draw_API.setColor                = @setColor;
  draw_API.updateView              = @updateView;  
  draw_API.translateView           = @translateView;    
  draw_API.setString               = @setString;  
  draw_API.getBoundingBox          = @getBoundingBox;
  draw_API.clear                   = @clear;
  draw_API.setVisible              = @setVisible;
  draw_API.showActiveDecoration    = @showActiveDecoration;
  draw_API.showPredictedDecoration = @showPredictedDecoration;
  
  %----------------------------
  function pos = getBoundingBox
    pos = bounding_box;
  end

  %---------------------------
  function initialize(h)

    h_group = h;
    
    h_circle      = line(...
                         'Marker', 'o', ...
                         'HitTest', 'off', ...
                         'Parent', h_group,...
                         'Visible','off');
    
    h_plus        = line(...
                         'Marker', '+', ...                       
                         'HitTest', 'off', ...
                         'Parent', h_group,...
                         'Visible','off');
  
    h_text        = text(...,
                         'FontName','SansSerif',...
                         'FontSize',8,...
                         'HorizontalAlignment','left',...
                         'VerticalAlignment','top',...
                         'Clipping','on',...
                         'HitTest', 'off', ...
                         'Parent', h_group,...
                         'BackgroundColor','w',...
                         'Visible','off',...    
                         'Tag','cpText'); %Tag used for testing.

    h_active_circle  = line(...
                            'Marker', 'o', ...
                            'HitTest', 'off', ...
                            'Parent', h_group,...
                            'Visible','off');
    
    h_predicted_text = text(...,
                            'FontName','SansSerif',...
                            'FontSize',8,...
                            'HorizontalAlignment','left',...
                            'VerticalAlignment','bottom',...
                            'Clipping','on',...
                            'HitTest', 'off', ...
                            'Parent', h_group,...
                            'String','P',...
                            'BackgroundColor','y',...
                            'Visible','off',...
                            'Tag','cpPredictedText'); %Tag used for testing.

    h_axes = iptancestor(h_group, 'axes');

    is_ydir_reverse = strcmp(get(h_axes,'YDir'),'reverse');
    
    circle_diameter_pixels = 3;
    wing_size_pixels = 3;
    active_circle_diameter = 12;

    points_per_screen_pixel = getPointsPerScreenPixel;
    circle_size_points = 2*circle_diameter_pixels * points_per_screen_pixel;
    plus_size_points = ...
        circle_size_points + 2*wing_size_pixels*points_per_screen_pixel; 
    active_circle_size_points = ...
        2*active_circle_diameter*points_per_screen_pixel;    

    set(h_circle,'MarkerSize',circle_size_points)
    set(h_plus,'MarkerSize',plus_size_points)    
    set(h_active_circle,'MarkerSize',active_circle_size_points)
    
    enterFcn = @(f,cp) set(f, 'Pointer', 'fleur');
    iptSetPointerBehavior(h_group, enterFcn);
    
  end

  %------------------------------
  function setVisible(TF)
     
      mode_invariant_obj = [h_circle,h_plus,h_text];
      mode_variant_obj = [h_active_circle,h_predicted_text];
      all_obj = [mode_invariant_obj,mode_variant_obj];
      
      if TF
          set(mode_invariant_obj,'Visible','on');
          manageDecorationVisibility();
      else
          set(all_obj,'Visible','off');
      end
          
  end    
  
  %-------------
  function clear

    delete([h_circle h_plus h_text h_active_circle h_predicted_text])
    
  end

  %-------------------------------
  function showActiveDecoration(b)
    
    show_active_decoration = b;
    manageDecorationVisibility();
        
  end

  %-------------------------------
  function showPredictedDecoration(b)
    
    show_predicted_decoration = b;  
    manageDecorationVisibility();  
        
  end

  %----------------------------------
  function manageDecorationVisibility
   
      predicted_vis = logical2onoff(show_predicted_decoration);
      active_vis = logical2onoff(show_active_decoration);
      set(h_predicted_text,'Visible',predicted_vis);
      set(h_active_circle,'Visible',active_vis);
      
  end    
  
  %-------------------
  function setColor(c)
    if ishghandle(h_group) 
      set([h_plus h_active_circle], 'Color', c);
      set(h_circle,'MarkerFaceColor',c,...
                   'MarkerEdgeColor',c);                     
    end
  end

  %----------------------------
  function updateView(position)
    
    pos_x = position(1);
    pos_y = position(2);
  
    if ~ishghandle(h_group)
        return;
    end

    [dx_per_screen_pixel, dy_per_screen_pixel] = getAxesScale(h_axes);
    
    x_mid = pos_x / dx_per_screen_pixel;
    x_left = x_mid - circle_diameter_pixels;
    x_right = x_mid + circle_diameter_pixels;

    % Note that y_min and y_max are reversed in terms of how the shape 
    % is drawn if get(h_axes,'YDir') is 'reverse'.
    y_mid = pos_y / dy_per_screen_pixel;
    y_min = y_mid - circle_diameter_pixels;
    y_max = y_mid + circle_diameter_pixels;

    % Convert the x and y extrema to data units and clip it to be one
    % pixel inside the axes limits.
    [x,y] = pixel2DataUnits(h_axes,[x_left x_right],[y_min y_max]);

    % Calculate deltas to use when we are just translating
    half_delta_x_data_units = (x(2) - x(1))/2;
    half_delta_y_data_units = (y(2) - y(1))/2;

    % move the text over a bit
    nudge = active_circle_diameter/2; 
    text_nudge_data_units = nudge * dx_per_screen_pixel;
    
    % Set the bounding box to include the entire extent of the drawn
    % rectangle, including decorations.
    bounding_box = findBoundingBox(x,y);
    
    if ~isequal(get(h_circle, 'XData'), pos_x) || ...
       ~isequal(get(h_circle, 'YData'), pos_y)
      set([h_circle h_plus h_active_circle],...
          'XData', pos_x, 'YData', pos_y);
    end

    % This needs to be outside of conditional in case user zoomed and so 
    % text stays right distance from the symbol.
    moveText(x,y)
    
  end

  %-------------------------------
  function translateView(position)
    
    pos_x = position(1);
    pos_y = position(2);

    dx2 = half_delta_x_data_units;
    x = [pos_x-dx2, pos_x+dx2];

    dy2 = half_delta_y_data_units;
    y = [pos_y-dy2, pos_y+dy2];
    
    % Set the bounding box to include the entire extent of the drawn
    % rectangle, including decorations.
    bounding_box = findBoundingBox(x,y);
    
    if ~isequal(get(h_circle, 'XData'), pos_x) || ...
       ~isequal(get(h_circle, 'YData'), pos_y)
      set([h_circle h_plus h_active_circle],...
          'XData', pos_x, 'YData', pos_y);
      moveText(x,y)
    end
    
  end

  %---------------------
  function moveText(x,y)
    
    % Figure out text position depending on axes 'YDir'
    x_right = max(x) + text_nudge_data_units;
    y_min = min(y); 
    y_max = max(y); 
    if is_ydir_reverse
      text_pos = [x_right y_max];
      pred_text_pos = [x_right y_min];
    else
      text_pos = [x_right y_min];    
      pred_text_pos = [x_right y_max];      
    end
    set(h_text,'Position',text_pos);
    set(h_predicted_text,'Position',pred_text_pos);    
    
  end
  
  %--------------------
  function setString(s)
    if ~isempty(s)
      set(h_text,'String',s);
      api = iptgetapi(h_group);
      updateView(api.getPosition());
      updateAncestorListeners(h_group,@(varargin) updateView(api.getPosition()));
    end
  end

  
end
