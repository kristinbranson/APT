function DRAW_API = freehandSymbol
%freehandSymbol Creates renderer for a freehand polygon.
%   DRAW_API = freehandSymbol creates a DRAW_API for use in association
%   with IMFREEHAND.  DRAW_API is a structure of function handles that are used by
%   IMFREEHAND to draw the freehand region and update its properties.
    
%   Copyright 2007-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.4 $  $Date: 2008/12/22 23:47:54 $
    
% Initialize all function scoped variables to empty
    [h_top_line,h_bottom_line,h_patch,...
     h_close_line_top,h_close_line_bottom,h_group,...
	 mode_invariant_obj,mode_variant_obj,all_obj,...
	 is_closed] = deal([]);
    
    DRAW_API.setColor               = @setColor;
    DRAW_API.getColor               = @getColor;
    DRAW_API.initialize             = @initialize;
    DRAW_API.setVisible             = @setVisible;
    DRAW_API.setClosed              = @setClosed;
	DRAW_API.updateView             = @updateView;
    
    %---------------------
    function initialize(h)
       
		line_width = getPointsPerScreenPixel();
        h_group = h;
	
		% This is a workaround to an HG bug g349263.
		buttonDown = getappdata(h_group,'buttonDown');
       
		h_patch = patch('FaceColor', 'none',...
			'EdgeColor', 'none', ...
			'HitTest', 'on', ...
			'Parent', h_group,...
			'ButtonDown',buttonDown,...
			'Tag','patch',...
			'Visible','off');
		
        h_bottom_line = line('Color', 'w', ...
                             'LineStyle', '-', ...
                             'LineWidth', 3*line_width, ...
                             'HitTest', 'on', ...
							 'ButtonDown',buttonDown, ...
                             'Parent', h_group,...
                             'Tag','bottom line',...
                             'Visible','off');

        h_top_line = line('LineStyle', '-', ...
                          'LineWidth', line_width, ...
                          'HitTest', 'on', ...
                          'ButtonDown',buttonDown,...
                          'Parent', h_group,...
                          'Tag','top line',...
                          'Visible','off');
        
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

        mode_invariant_obj = [h_top_line,h_bottom_line,h_patch];
        setPointerBehavior();
			
	end
    
    %---------------------
    function setClosed(TF)
        
        is_closed = TF;
        drawModeAffordances();
        
    end
    
   %----------------------
   function setVisible(TF)
       
	   mode_variant_obj = getVertices();
	   all_obj = [mode_invariant_obj,mode_variant_obj'];
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
		vertices = findobj(h_group,'tag','impoint');
	end

    %---------------------------
    function drawModeAffordances
           
        h_close = [h_close_line_top,h_close_line_bottom,h_patch];
        if ~all(ishghandle(h_close))
            return
        end
        
        if is_closed
            set(h_close,'Visible','on');
        else
            set(h_close,'Visible','off');
        end
    
    end %drawModeAffordances
    
    %---------------------------
    function updateView(new_pos)
        
        if ~ishghandle(h_group)
            return;
        end
        
        set([h_patch,h_top_line,h_bottom_line],...
            'XData',new_pos(:,1),...
            'YData',new_pos(:,2));
		
        close_line_x_data = [new_pos(end,1) new_pos(1,1)];
        close_line_y_data = [new_pos(end,2) new_pos(1,2)];
		
		set([h_close_line_top,h_close_line_bottom],...
			'XData',close_line_x_data,...
			'YData',close_line_y_data);
        
	end %updateView

	%-------------------
	function setColor(c)

        set([h_top_line,h_close_line_top], 'Color', c);
						
	end %setColor

	%--------------------
	function c = getColor

        c = get(h_top_line,'Color');
						
	end %getColor

  %--------------------------
  function setPointerBehavior
          
  	h_obj = [h_close_line_top,h_close_line_bottom,h_top_line,h_bottom_line,h_patch];
    enterFcn = @(h_fig,pos) set(h_fig,'Pointer','fleur');
    
  	for i = 1:length(h_obj)
  		iptSetPointerBehavior(h_obj(i),enterFcn);
  	end
  	  			           
  end %setPolygonPointerBehavior
   
end
    