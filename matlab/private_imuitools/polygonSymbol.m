function DRAW_API = polygonSymbol
%polygonSymbol Creates renderer for polygon ROIs.
%   DRAW_API = polygonSymbol creates a DRAW_API for use in association
%   with IMPOLY.  DRAW_API is a structure of function handles that are used by
%   IMPOLY to draw a polygon and update its properties.
    
%   Copyright 2007-2009 The MathWorks, Inc.
%   $Revision: 1.1.6.11 $ $Date: 2010/09/13 16:14:22 $
    
% Initialize all function scoped variables to empty
    [h_top_line,h_bottom_line,h_patch,...
     h_close_line_top,h_close_line_bottom,h_group,...
	 mode_invariant_obj,mode_variant_obj,all_obj,...
	 show_vertices,is_closed,buttonDown,line_width,...
     current_color,point_cursor,a_down,h_fig,id_up,id_down,...
     pointer_managed] = deal([]);
    
    DRAW_API.getColor               = @getColor;
    DRAW_API.setColor               = @setColor;
    DRAW_API.initialize             = @initialize;
    DRAW_API.setVisible             = @setVisible;
    DRAW_API.setClosed              = @setClosed;
	DRAW_API.updateView             = @updateView;
	DRAW_API.showVertices           = @showVertices;
    DRAW_API.pointerManagePolygon   = @pointerManagePolygon;
    DRAW_API.unwireShiftKeyPointAffordance = @unwireShiftKeyPointAffordance;

    
    %---------------------
    function initialize(h)
       
		line_width = getPointsPerScreenPixel();
        h_group = h;
		show_vertices = true;
	
		% This is a workaround to an HG bug g349263.
		buttonDown = getappdata(h_group,'buttonDown');
       
		h_patch = patch('FaceColor', 'none',...
			'EdgeColor', 'none', ...
			'HitTest', 'on', ...
			'Parent', h_group,...
			'ButtonDown',buttonDown,...
			'Tag','patch',...
			'Visible','off');
        
        h_fig = iptancestor(h_group,'figure');
        		         
        h_close_line_bottom = line('Color', 'w', ...
                                   'LineStyle', '-', ...
                                   'LineWidth', 3*line_width, ...
                                   'HitTest', 'on', ...
								   'ButtonDown',buttonDown, ...
                                   'Parent', h_group,...
                                   'Tag','close line bottom',...
                                   'Visible','off');
        
        h_close_line_top = line('LineStyle', '-', ...
                          'LineWidth', line_width, ...
                          'HitTest', 'on', ...
                          'Parent', h_group,...
						  'ButtonDown',buttonDown,...
                          'Tag','close line top',...
                          'Visible','off');
        
        pointer_managed = false;              
        point_cursor = makePointInsertCursor();              

    end
    
    %--------------------------------------
    function cursor = makePointInsertCursor
        
        cursor = imread(fullfile(ipticondir,'point.png'));
        cursor = double(rgb2gray(cursor));
		ind = logical(cursor);
		cursor(ind) = 1;
		cursor(~ind) = NaN;
       
    end

    %----------------------
    function resortChildren
    % resortChildren adjusts the drawing order of the vertices and lines used to
    % draw the polygon such that the impoint vertices are always on top of
    % line objects. The vertices must be drawn on top of line objects to
    % allow the vertices to receive buttonDown during a vertex drag.
       
        h_vertices = getVertices();
        h_children = get(h_group,'Children');
        vert_idx = ismember(h_children,h_vertices);
        h_children_new = [h_children(vert_idx);h_children(~vert_idx)];
        set(h_group,'Children',h_children_new);
        
    end
    
    %---------------------
    function setClosed(TF)
        
        is_closed = TF;
        drawModeAffordances();
        
    end
    
    %------------------------
    function showVertices(TF)
		
		show_vertices = TF;
        drawModeAffordances();
        
    end %setVerticesDraggable

   %----------------------
   function setVisible(TF)
       
       mode_invariant_obj = [h_bottom_line,h_top_line,h_patch];
	   mode_variant_obj = [h_close_line_bottom,h_close_line_top,getVertices()'];
	   all_obj = [mode_invariant_obj,mode_variant_obj];
       if TF
           set(mode_invariant_obj,'Visible','on');
           drawModeAffordances();
       else
           set(all_obj,'Visible','off');
       end
       
   end    

    %-------------------------------
	function vertices =  getVertices
		% This is the only way to find all impoints within h_group until
        % impoint is a real object.
		vertices = findobj(h_group,'tag','impoly vertex');
	end

    %---------------------------
    function drawModeAffordances
   
		h_vertices = getVertices();
        if show_vertices
			set(h_vertices,'Visible','on');
		else
			set(h_vertices,'Visible','off');
        end
        
        h_close = [h_close_line_top,h_close_line_bottom,h_patch];
        if is_closed
            set(h_close,'Visible','on');
        else
            set(h_close,'Visible','off');
        end
    
    end %drawModeAffordances
    
    %----------------------------
    function n =  getNumVert(pos)
       
        n = size(pos,1);
        
    end

    %----------------------------
    function h_line = makeTopLine
        
        h_line = line('LineStyle', '-', ...
             'LineWidth', line_width, ...
             'Color',current_color,...       
             'HitTest', 'on', ...
             'ButtonDown',buttonDown,...
             'Parent', h_group,...
             'Tag','top line',...
             'Visible','on');
         
         if pointer_managed
             setupLinePointerManagement(h_line);
         end
         
    end
    
    %-------------------------------
    function h_line = makeBottomLine
        
        h_line = line('Color', 'w', ...
             'LineStyle', '-', ...
             'LineWidth', 3*line_width, ...
             'HitTest', 'on', ...
             'ButtonDown',buttonDown, ...
             'Parent', h_group,...
             'Tag','bottom line',...
             'Visible','on');
         
         if pointer_managed
             setupLinePointerManagement(h_line);
         end
         
    end
       
    %--------------------------------
    function manageNumberOfLines(pos)
        % ManageNumberOfLines dynamically adds and removes line HG objects
        % so that a N-vertex polygon will be drawn with N separate HG line
        % objects.
                
        if getNumVert(pos)-1 > length(h_top_line)
            for i = length(h_top_line)+1:getNumVert(pos)-1
                h_bottom_line(i) = makeBottomLine();
                h_top_line(i) = makeTopLine();
            end
            resortChildren();
        elseif getNumVert(pos)-1 < length(h_top_line)
            % Begin removing excess lines.
            
            % Use max to ensure that lowest index into line arrays cannot
            % be less than 1.
            min_line_idx = max(1,getNumVert(pos));
            
            lines_to_delete = length(h_top_line):-1:min_line_idx;      
            delete(h_top_line(lines_to_delete));
            delete(h_bottom_line(lines_to_delete));
            h_top_line(lines_to_delete) = [];
            h_bottom_line(lines_to_delete) = [];
        end
        
    end
    
    %---------------------------
    function updateView(new_pos)
               
        manageNumberOfLines(new_pos);
        
        if ~ isempty(new_pos)
            set(h_patch,...
                'XData',new_pos(:,1),...
                'YData',new_pos(:,2));
        else
            set(h_patch,'XData',[],...
                        'YData',[]);
        end
        
        
        for j = 1:getNumVert(new_pos)-1
        
            set([h_bottom_line(j),h_top_line(j)],...
                'XData',[new_pos(j,1),new_pos(j+1,1)],...
                'YData',[new_pos(j,2),new_pos(j+1,2)]);
        end
         
        if ~isempty(new_pos)
            close_line_x_data = [new_pos(end,1) new_pos(1,1)];
            close_line_y_data = [new_pos(end,2) new_pos(1,2)];
        else
            close_line_x_data = [];
            close_line_y_data = [];
        end
        
        set([h_close_line_top,h_close_line_bottom],...
            'XData',close_line_x_data,...
			'YData',close_line_y_data);

	end %updateView

	%-------------------
	function setColor(c)
        
            current_color = c;
			h_vertices = getVertices();
            set([h_top_line,h_close_line_top],'Color',c);
            
			for i = 1:numel(h_vertices)
				vertex_api = iptgetapi(h_vertices(i));
				vertex_api.setColor(c);
			end
			
    end %setColor

    %--------------------
    function c = getColor
       
        c = get(h_close_line_top,'Color');
        
    end
    
    %------------------------------------------
    function setupLinePointerManagement(h_line)

        lineBehavior.enterFcn = [];
        lineBehavior.exitFcn = [];
        lineBehavior.traverseFcn = @linePointerManagement;

        iptSetPointerBehavior(h_line,lineBehavior)

    end
    
    %---------------------------------------------
    function linePointerManagement(h_fig,varargin)

        if a_down
            set(h_fig,...
                'Pointer','Custom',...
                'PointerShapeCData',point_cursor,...
                'PointerShapeHotSpot',[8 8]);
        else
            set(h_fig,'Pointer','fleur');
        end

    end

    %--------------------------------------------
    function setupPatchPointerManagement(h_patch)

        patchBehavior.enterFcn = [];
        patchBehavior.exitFcn = [];
        patchBehavior.traverseFcn = @(h_fig,pos) set(h_fig,'Pointer','fleur');
        
        iptSetPointerBehavior(h_patch,patchBehavior);
        
    end

    %-----------------------------------
    function wireShiftKeyPointAffordance

        id_down = iptaddcallback(h_fig,'WindowKeyPressFcn',@keyDown);
        id_up = iptaddcallback(h_fig,'WindowKeyReleaseFcn',@keyUp);

        %-------------------------
        function keyDown(h_obj,ed) %#ok

            if strcmp(ed.Key,'a')
                a_down = true;
                % Enabling pointer management forces pointer manager to
                % update. Update to allow pointer manager to discover if
                % point is over polygon line
                iptPointerManager(h_fig,'Enable');
            end

        end

        %-----------------------
        function keyUp(h_obj,ed) %#ok
            
            if strcmp(ed.Key,'a')
                a_down = false;
                iptPointerManager(h_fig,'Enable');
            end
      
        end

    end

    %-------------------------------------
    function unwireShiftKeyPointAffordance
       
        iptremovecallback(h_fig,'WindowKeyPressFcn',id_down);
        iptremovecallback(h_fig,'WindowKeyReleaseFcn',id_up);
        
    end

    %--------------------------------
    function pointerManagePolygon(TF)
       
        pointer_managed = TF;
        if TF
            
            wireShiftKeyPointAffordance();
            setupPatchPointerManagement(h_patch);
            setupLinePointerManagement(h_close_line_top);
            setupLinePointerManagement(h_close_line_bottom);
                
            for i = 1:length(h_top_line)
                
                setupLinePointerManagement(h_top_line(i));
                setupLinePointerManagement(h_bottom_line(i));
                
            end
            
        else
            
              unwireShiftKeyPointAffordance();
              iptSetPointerBehavior(h_patch,[]);
              iptSetPointerBehavior(h_close_line_top,[]);
              iptSetPointerBehavior(h_close_line_bottom,[]);
              
              for i = 1:length(h_top_line)
              
                  iptSetPointerBehavior(h_top_line(i),[]);
                  iptSetPointerBehavior(h_bottom_line(i),[]);
              
              end
              
        end
                
    end % pointerManagePolygon

end %polygonSymbol
    