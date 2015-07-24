function draw_API = wingedRect
%wingedRect Creates renderer for winged rectangle symbol.
%   DRAW_API = wingedRect(H_GROUP) creates a DRAW_API for use in association
%   with IMRECT that draws rectangles with wings that show only if the rectangle
%   is very small. DRAW_API is a structure of function handles that are used by
%   IMRECT to draw the rectangle and update its properties.
%
%       DRAW_API.getColor
%       DRAW_API.setColor
%       DRAW_API.updateView
%       DRAW_API.getBoundingBox
%       DRAW_API.clear    
%       DRAW_API.setResizable
%       DRAW_API.setFixedAspectRatio
%       DRAW_API.setVisible
%       DRAW_API.isRectBody
%       DRAW_API.isSide
%       DRAW_API.isCorner
%       DRAW_API.isWing
%       DRAW_API.findSelectedSide
%       DRAW_API.findSelectedVertex   
    
%   Copyright 2005-2010 The MathWorks, Inc.
%   $Revision: 1.1.6.16 $  $Date: 2010/09/24 14:32:30 $

% Initialize function scoped variables to empty
[h_bottom_line,h_wing_line,h_patch,h_corner_markers,h_side_markers,...
h_top_lines,is_fixed_aspect_ratio,is_resizable,num_vert,line_tags,...
side_marker_tags,corner_marker_tags,h_axes,h_group,bounding_box] = deal([]);

draw_API.initialize          = @initialize;
draw_API.getColor            = @getColor;
draw_API.setColor            = @setColor;
draw_API.updateView          = @updateView;
draw_API.getBoundingBox      = @getBoundingBox;
draw_API.clear               = @clear;
draw_API.setResizable        = @setResizable;
draw_API.setFixedAspectRatio = @setFixedAspectRatio;
draw_API.setVisible          = @setVisible;
draw_API.findSelectedSide    = @findSelectedSide;
draw_API.findSelectedVertex  = @findSelectedVertex;

   %---------------------
   function initialize(h,translateRectFcn,sideResizeFcn,cornerResizeFcn)
	   
	   h_group = h;
	  
	   h_axes = iptancestor(h_group,'axes');

	   % The line objects should have a width of one screen pixel.
	   % line_width = ceil(getPointsPerScreenPixel);
	   line_width = getPointsPerScreenPixel();
	   h_bottom_line = line('Color', 'w', ...
		   'LineStyle', '-', ...
		   'LineWidth', 3*line_width, ...
		   'HitTest', 'off', ...
		   'Parent', h_group,...
		   'Tag','bottom line',...
		   'Visible','off');
	   h_wing_line = line(...
		   'LineStyle', '-', ...
		   'LineWidth', line_width, ...
		   'HitTest', 'on', ...
		   'Parent', h_group,...
		   'Tag','wing line',...
		   'ButtonDownFcn',translateRectFcn,...
		   'Visible','off');

	   h_patch = patch('FaceColor', 'none', 'EdgeColor', 'none', ...
		   'HitTest', 'on', ...
		   'Parent', h_group,...
		   'Tag','patch',...
		   'ButtonDownFcn',translateRectFcn,...
		   'Visible','off');

	   % Function scope. Determines whether rectangle is drawn to indicate fixed
	   % aspect ratio.
	   is_fixed_aspect_ratio = false;

	   % Function scope. Determines whether rectangle is drawn to indicate it is
	   % resizable.
	   is_resizable = true;

	   % Function scope. Number of vertices in rectangle.
	   num_vert = 4;

	   line_tags = {'minx' 'maxy' 'maxx' 'miny'};

	   % Preallocate arrays containing HG objects to clean up lint.
	 %  [h_top_lines,h_side_markers,h_corner_markers] = deal(zeros(1,4));

	   for i = 1:num_vert

		   h_top_lines(i) = line('LineStyle', '-', ...
			   'LineWidth', line_width, ...
			   'HitTest', 'on', ...
			   'Parent', h_group,...
			   'Tag', [line_tags{i} ' top line'],...
			   'ButtonDownFcn',sideResizeFcn,...
			   'Visible','off');
	   end

	   side_marker_tags   = {'minx' 'maxy' 'maxx' 'miny'};
	   corner_marker_tags = {'minx miny', 'minx maxy',...
		   'maxx maxy', 'maxx miny'};

	   for i = 1:num_vert

		   h_side_markers(i) = line('Marker','square',...
			   'MarkerSize',6,...
			   'HitTest','off',...
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

	   setupCursorManagement();
	   
   end

    %-------------
    function clear
        
        delete(h_wing_line);
        delete(h_bottom_line);
        delete(h_top_lines);
        delete(h_patch);
        delete(h_corner_markers);
        delete(h_side_markers);
        
    end
    
    %----------------------
    function setVisible(TF)
        
        if TF
            setModeInvariantObjVisible('on');
            drawModeAffordances();
        else
            setAllObjVisible('off');
        end
        
    end

    %----------------------------
    function setAllObjVisible(onOff)
        
        setModeInvariantObjVisible(onOff);
        set(h_corner_markers,'Visible',onOff);
        set(h_side_markers,'Visible',onOff);
        
    end

    %----------------------------
    function setModeInvariantObjVisible(onOff)
        
         set(h_wing_line,'Visible',onOff);
         set(h_bottom_line,'Visible',onOff);
         set(h_top_lines,'Visible',onOff);
         set(h_patch,'Visible',onOff);
                 
    end

    %----------------------------
    function pos = getBoundingBox
        pos = bounding_box;
    end % getBoundingBox

    %----------------------------
    function updateView(position)

        if ~ishghandle(h_group)
            return;
        end

        x_side_markers = [position(1), position(1)+position(3)/2,...
                          position(1)+position(3), position(1)+position(3)/2];
        
        y_side_markers = [position(2)+position(4)/2, position(2)+position(4),...
                          position(2)+position(4)/2, position(2)];
        
        x_top_lines = [position(1), position(1);...
                       position(1), position(1)+position(3);...
                       position(1)+position(3),position(1)+position(3);...
                       position(1)+position(3),position(1)];
        
        y_top_lines = [position(2), position(2)+position(4);...
                       position(2)+position(4),position(2)+position(4);...
                       position(2)+position(4),position(2);...
                       position(2),position(2)];
   
        [dx_per_screen_pixel, dy_per_screen_pixel] = getAxesScale(h_axes);

        min_decorated_rect_size = 30;
        x_left = position(1) / dx_per_screen_pixel;
        x_right = (position(1) + position(3)) / dx_per_screen_pixel;
        x_wing_size = max(ceil((min_decorated_rect_size - ...
            (x_right - x_left)) / 2), 0);

        y_bottom = position(2) / dy_per_screen_pixel;
        y_top = (position(2) + position(4)) / dy_per_screen_pixel;
        y_wing_size = max(ceil((min_decorated_rect_size - ...
            (y_top - y_bottom)) / 2), 0);

        x1 = x_left - x_wing_size;
        x2 = x_left;
        x3 = (x_left + x_right) / 2;
        x4 = x_right;
        x5 = x_right + x_wing_size;

        y1 = y_bottom - y_wing_size;
        y2 = y_bottom;
        y3 = (y_bottom + y_top) / 2;
        y4 = y_top;
        y5 = y_top + y_wing_size;
        
        % (x,y) is a polygon that strokes the middle line.  Here it is in
        % screen pixel units.
        x = [x1 x2 x2 x3 x3 x3 x4 x4 x5 x4 x4 x3 x3 x3 x2 x2 x1];
        y = [y3 y3 y2 y2 y1 y2 y2 y3 y3 y3 y4 y4 y5 y4 y4 y3 y3];
        
        % Convert the (x,y) polygon back to data units.
        [x,y] = pixel2DataUnits(h_axes,x,y);
        
        xx1 = x1 - 1;
        xx2 = x2 - 1;
        xx3 = x3 - 1;
        xx4 = x3 + 1;
        xx5 = x4 + 1;
        xx6 = x5 + 1;

        yy1 = y1 - 1;
        yy2 = y2 - 1;
        yy3 = y3 - 1;
        yy4 = y3 + 1;
        yy5 = y4 + 1;
        yy6 = y5 + 1;

        % (xx,yy) is a polygon that strokes the outer line.  Here it is in
        % screen pixel units.
        xx = [xx1 xx2 xx2 xx3 xx3 xx4 xx4 xx5 xx5 xx6 xx6 xx5 xx5 ...
            xx4 xx4 xx3 xx3 xx2 xx2 xx1 xx1];
        yy = [yy3 yy3 yy2 yy2 yy1 yy1 yy2 yy2 yy3 yy3 yy4 yy4 yy5 ...
            yy5 yy6 yy6 yy5 yy5 yy4 yy4 yy3];

        % Convert the (xx,yy) polygon back to data units.
        [xx,yy] = pixel2DataUnits(h_axes,xx,yy);

        % Set the outer position to include the entire extent of the drawn
        % rectangle, including decorations.
        bounding_box = findBoundingBox(xx,yy);
        
        setXYDataIfChanged(x, y,...
                           x_side_markers, y_side_markers,...
                           x_top_lines,y_top_lines);
                       
    end %updateView

    %------------------
    function setColor(c)
        
        handlesAreValid = all(ishghandle([h_top_lines,...
                                        h_wing_line,...
                                        h_side_markers,...
                                        h_corner_markers]));
                                
        if handlesAreValid
            set(h_top_lines,'Color',c);
            set(h_wing_line,'Color',c);
            set(h_side_markers,'Color',c);
            set(h_corner_markers,'Color',c);
        end
        
    end %setColor

    %--------------------
    function c = getColor
       
        c = get(h_wing_line,'Color');
        
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
                set(h_top_lines,'hittest','off');
            else
                set(h_side_markers,'Visible','on');
                set(h_top_lines,'hittest','on');
            end
        else
            set([h_corner_markers,h_side_markers],'Visible','off');
            set(h_top_lines,'hittest','off');
        end        
    
    end %drawModeAffordances
    
    %----------------------------------
    function setXYDataIfChanged(x, y, x_side_markers, y_side_markers,...
                                x_top_lines,y_top_lines)
    
        % Set XData and YData of HG object h to x and y if they are different.  h
        % must be a valid HG handle to an object having XData and YData properties.
        % No validation is performed.
        h = [h_wing_line,h_bottom_line,h_patch];
        if ~isequal(get(h, 'XData'), x) || ~isequal(get(h, 'YData'), y)
            set(h, 'XData', x, 'YData', y);
            for j= 1:num_vert
                set(h_top_lines(j),'XData',x_top_lines(j,:),'YData',y_top_lines(j,:));
                set(h_corner_markers(j),'XData',x_top_lines(j,1),'YData',y_top_lines(j,1));   
                set(h_side_markers(j),'XData',x_side_markers(j),'YData',y_side_markers(j));
            end
          
        end
    end % setXYDataIfChanged
           
    %--------------------------------------------
    function side_index = findSelectedSide(h_hit)
        
        side_index = find(h_hit == h_top_lines);
        
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
        
        % provide affordance to indicate that clicking on wing lines will
        % translate rectangle when rectangle is small
        iptSetPointerBehavior(h_wing_line,@(h_fig,current_point)...
            set(h_fig,'Pointer','fleur'));
        
        setupCornerPointerBehavior();
        setupSidePointerBehavior();
        
        %--------------------------------
        function setupSidePointerBehavior
            
        % left/right/up/down cursors are oriented correctly independent of xdir/ydir
        % of axes. Don't need to worry about these cases as with corner
        % pointers.
            cursor_names = {'left','top','right','bottom'};
            for j = 1:num_vert
                enterFcn = @(h_fig,currentPoint) set(h_fig,'Pointer',cursor_names{j});
                iptSetPointerBehavior(h_top_lines(j),enterFcn);
            end
                        
        end %setupSidePointerBehavior
        
        %----------------------------------
        function setupCornerPointerBehavior
            
             if ~all(ishghandle(h_corner_markers))
                 return
             end
            
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
             
             for j = 1:num_vert
                 enterFcn = @(h_fig,currentPoint) set(h_fig,'Pointer',cursor_names{j});
                 iptSetPointerBehavior(h_corner_markers(j),enterFcn); 
             end
                
        end %setupCornerPointerBehavior
        
    end %setupCursorManagement
   
end %wingedRect
