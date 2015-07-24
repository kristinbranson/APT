function draw_API = ellipseSymbol
%ellipseSymbol Creates renderer for ellipses.    
%   DRAW_API = ellipseSymbol creates a DRAW_API for use in association with
%   IMELLIPSE that draws ellipses. DRAW_API is a structure of function
%   handles that are used by IMELLIPSE to draw the ellipse and update its
%   properties.
%
%       DRAW_API.initialize(h_group)
%       DRAW_API.getColor  
%       DRAW_API.setColor(color)
%       DRAW_API.updateView(position)  
%       DRAW_API.clear()
%       DRAW_API.setVisible()
%       DRAW_API.isBody
%       DRAW_API.isSide
%       DRAW_API.isCorner
%       DRAW_API.isWing
%       DRAW_API.findSelectedSide
%       DRAW_API.findSelectedVertex    

%   Copyright 2007-2010 The MathWorks, Inc.
%   $Revision: 1.1.6.7 $  $Date: 2010/06/07 16:32:30 $
    
% Initialize all function scoped variables to empty. 
[mode_invariant_obj,mode_variant_obj,all_obj,num_markers,...
side_marker_tags,corner_marker_tags,...
is_resizable,is_fixed_aspect_ratio,h_axes,...
h_top_line,h_bottom_line,h_patch,...
h_corner_markers,h_side_markers,h_group] = deal([]);

draw_API.initialize            = @initialize;
draw_API.getColor              = @getColor;
draw_API.setColor              = @setColor;
draw_API.clear                 = @clearGraphics;
draw_API.updateView            = @updateView;
draw_API.setResizable          = @setResizable;
draw_API.setFixedAspectRatio   = @setFixedAspectRatio;
draw_API.setVisible            = @setVisible;
draw_API.isBody                = @isBody;
draw_API.isSide                = @isSide;
draw_API.isCorner              = @isCorner;
draw_API.isWing                = @isWing;
draw_API.findSelectedSide      = @findSelectedSide;
draw_API.findSelectedVertex    = @findSelectedVertex;

  %---------------------------
  function initialize(h,translateFcn,sideResizeFcn,cornerResizeFcn)
      
	  h_group = h;
	  
      h_axes = iptancestor(h_group,'axes');

      % This is a workaround to an HG bug g349263. 
      buttonDown = getappdata(h_group,'buttonDown');
      
      % The line objects should have a width of one screen pixel.
      line_width = getPointsPerScreenPixel();
      
      % Determines whether ellipse is drawn to indicate fixed
      % aspect ratio.
      is_fixed_aspect_ratio = false;
      
      % Determines whether ellipse is drawn to indicate it is
      % resizable.
      is_resizable = true;
      
      % Markers are drawn along axes directions and diagonal directions to allow
      % ellipse to be resized. There are four of each type.
      num_markers = 4;
      
      side_marker_tags   = {'minx' 'maxy' 'maxx' 'miny'};
      corner_marker_tags = {'minx miny', 'minx maxy',...
                            'maxx maxy', 'maxx miny'};
      
      h_bottom_line = line('Color', 'w', ...
                           'LineStyle', '-', ...
                           'LineWidth', 3*line_width, ...
                           'HitTest', 'off', ...
                           'Parent', h_group,...
                           'Tag','bottom line',...
                           'Visible','off');

      h_top_line = line('Color', 'w', ...
                        'LineStyle', '-', ...
                        'LineWidth', line_width, ...
                        'HitTest', 'on', ...
                        'Parent', h_group,...
                        'Tag','top line',...
                        'Visible','off');
      
      h_patch = patch('FaceColor', 'none', 'EdgeColor', 'none', ...
                      'HitTest', 'on', ...
                      'Parent', h_group,...
                      'Tag','patch',...
                      'ButtonDownFcn',translateFcn,...
                      'Visible','off');
      
      for i = 1:num_markers
    
          h_side_markers(i) = line('Marker','square',...
                                   'MarkerSize',6,...
                                   'HitTest','on',...
                                   'ButtonDownFcn',sideResizeFcn,...
                                   'Parent',h_group,...
                                   'Tag',[side_marker_tags{i} ' side marker'],...
                                   'Visible','off');
          
          h_corner_markers(i) = line('Marker','square',...
                                     'MarkerSize',6,...
                                     'HitTest','on',...
                                     'Parent',h_group,...
                                     'Tag',[corner_marker_tags{i} ' corner marker'],...
                                     'ButtonDownFcn',cornerResizeFcn,...
                                     'Visible','off'); 
      end

      mode_invariant_obj = [h_top_line,h_bottom_line,h_patch];
      mode_variant_obj = [h_corner_markers,h_side_markers];
      all_obj = [mode_invariant_obj, mode_variant_obj];

      setupCursorManagement(); 
      
  end

  %-------------
  function clearGraphics

    delete(all_obj)
    
  end

   %----------------------
   function setVisible(TF)
           
       if TF
           set(mode_invariant_obj,'Visible','on');
           drawModeAffordances();
       else
           set(all_obj,'Visible','off');
       end
       
   end    

    %----------------------------
    function updateView(position)

        if ~ishghandle(h_group)
            return;
        end

        x_side_markers = [position(1), position(1)+position(3)/2,...
                          position(1)+position(3), position(1)+position(3)/2];
        
        y_side_markers = [position(2)+position(4)/2, position(2)+position(4),...
                          position(2)+position(4)/2, position(2)];
                
        vert = getEllipseVertices(h_axes,position);
        
        x_vert = vert(:,1);
        y_vert = vert(:,2);
       
        theta_corners = [5,3,1,7] .* pi/4;
        cx = mean([position(1),position(1) + position(3)]);
        cy = mean([position(2),position(2) + position(4)]);
        a = cx - position(1);
        b = cy - position(2);

        % Define "corner" markers at diagonal positions of ellipse using
        % parameterized ellipse
        x_corner_markers = cx + a*cos(theta_corners);
        y_corner_markers = cy + b*sin(theta_corners);
       
        setXYDataIfChanged(x_vert, y_vert,...
                           x_side_markers, y_side_markers,...
                           x_corner_markers,y_corner_markers);
                       
    end %updateView

    %------------------
    function setColor(c)
        
        handlesAreValid = all(ishghandle([h_top_line,...
                                        h_side_markers,...
                                        h_corner_markers]));
                                
        if handlesAreValid
            set(h_top_line, 'Color', c);
            set([h_side_markers,h_corner_markers],'Color',c)
        end
        
    end %setColor

    %--------------------
    function c = getColor
       
        c = get(h_top_line,'Color');
        
    end %getColor

    %-------------------------------
    function setFixedAspectRatio(TF)
    
        is_fixed_aspect_ratio = TF;
        drawModeAffordances();
        
    end %setFixedAspectRatio
    
    %------------------------
    function setResizable(TF)
    
        is_resizable = TF;
        drawModeAffordances();
        
    end %setResizable
    
    %---------------------------
    function drawModeAffordances
   
        if is_resizable
            set(h_corner_markers,'Visible','on');
            if is_fixed_aspect_ratio
                set(h_side_markers,'Visible','off');
                set(h_top_line,'hittest','off');
            else
                set(h_side_markers,'Visible','on');
                set(h_top_line,'hittest','on');
            end
        else
            set([h_corner_markers,h_side_markers],'Visible','off');
            set(h_top_line,'hittest','off');
        end        
    
    end %drawModeAffordances
    
    %----------------------------------
    function setXYDataIfChanged(x, y, x_side, y_side,...
                                x_corner,y_corner)
    
        % Set XData and YData of HG object h to x and y if they are different.  h
        % must be a valid HG handle to an object having XData and YData properties.
        % No validation is performed.
        h = [h_top_line,h_bottom_line,h_patch];
        if ~isequal(get(h_top_line, 'XData'), x) || ~isequal(get(h_top_line, 'YData'), y)
            set(h, 'XData', x, 'YData', y);
            for i=1:num_markers
                set(h_corner_markers(i),'XData',x_corner(i),'YData',y_corner(i));
                set(h_side_markers(i),'XData', x_side(i), 'YData',y_side(i));
            end          
        end
    end % setXYDataIfChanged
    
    %-------------------------------
    function TF =  isBody(h_hit)
    
        TF = h_hit == h_patch;

    end %isBody
    
    %--------------------------
    function TF = isSide(h_hit)
    
        TF = any(h_hit == h_side_markers);
        
    end %isSide
        
    %----------------------------
    function TF = isCorner(h_hit)

        % In ellipse, corner markers are markers in diagnonal positions which are
        % not along semi-minor or semi-major axes directions
        TF = any(h_hit == h_corner_markers);
        
    end %isCorner

    %--------------------------
	function TF = isWing(h_hit) %#ok imrect requires function signature
		% Ellipse doesn't have wings, but imrect requires the existence of
		% isWing in draw_api. Provide dummy function that always returns
		% false for ellipse draw_api.
		TF = false;
		
	end 

    %--------------------------------------------
    function side_index = findSelectedSide(h_hit)
        
        side_index = find(h_hit == h_side_markers);
        
    end
    
    %------------------------------------------------
	function vertex_index = findSelectedVertex(h_hit)
        
        vertex_index = find(h_hit == h_corner_markers);
        
    end
                    
    %-----------------------------    
    function setupCursorManagement
        h_fig = iptancestor(h_axes,'figure');
        iptPointerManager(h_fig);
        
        iptSetPointerBehavior(h_patch,@(h_fig,current_point) set(h_fig,'Pointer','fleur'));
                
        % Need listeners to react to changes in xdir/ydir of associated axes so that
        % pointer orientations always point in the left/right/up/down
        % directions as seen by the user.
        makeAxesDirectionListener = ...
            @(hax,prop) iptui.iptaddlistener(hax,prop,...
            'PostSet',@setupCornerPointerBehavior);
        
        listeners(1)     = makeAxesDirectionListener(h_axes,'XDir');
        listeners(end+1) = makeAxesDirectionListener(h_axes,'YDir');
        
        setappdata(h_group,'CornerPointerListeners',listeners);
        % clear unused references to listeners
        clear listeners;
                        
        setupCornerPointerBehavior();
        setupSidePointerBehavior();
        
        %--------------------------------
        function setupSidePointerBehavior
            
        % left/right/up/down cursors are oriented correctly independent of xdir/ydir
        % of axes. Don't need to worry about these cases as with corner
        % pointers.
            cursor_names = {'left','top','right','bottom'};
            for i = 1:num_markers
                enterFcn = @(h_fig,currentPoint) set(h_fig,'Pointer',cursor_names{i});
                iptSetPointerBehavior(h_side_markers(i),enterFcn);
            end
                        
        end %setupSidePointerBehavior
        
        %----------------------------------
        function setupCornerPointerBehavior
            
             xFlipped = strcmp(get(h_axes,'xdir'),'reverse');
             yFlipped  = strcmp(get(h_axes,'ydir'),'reverse');
             
             if xFlipped && yFlipped
                 cursor_names = {'topr','botr','botl','topl'};
             elseif xFlipped
                 cursor_names = {'botr','topr','topl','botl'};
             elseif yFlipped
                 cursor_names = {'topl','botl','botr','topr'};
             else
                 cursor_names = {'botl','topl','topr','botr'};
             end
          
             for i = 1:num_markers
                 enterFcn = @(h_fig,currentPoint) set(h_fig,'Pointer',cursor_names{i});
                 iptSetPointerBehavior(h_corner_markers(i),enterFcn); 
             end
                
        end %setupCornerPointerBehavior
        
    end %setupCursorManagement
   
end %wingedRect