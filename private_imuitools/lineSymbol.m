function draw_API = lineSymbol(h_group,translateLineFcn,resizeLineFcn)
%lineSymbol Creates renderer for line.  
%   draw_API = lineSymbol(H_GROUP,GET_POSITION_FCN) creates a
%   renderer for use in association with IMLINE that draws lines.
%   draw_API is a structure of function handles that are used
%   by IMLINE to draw the line and update its properties.
%
%       draw_API.getColor
%       draw_API.setColor         
%       draw_API.updateView
%       draw_API.setVisible
%       draw_API.getHitPoint
%       draw_API.isEndPoint
%       draw_API.isLineBody    
       
%   Copyright 2005-2010 The MathWorks, Inc.
%   $Revision: 1.1.6.8 $  $Date: 2010/06/07 16:32:33 $
      
  % The line objects should have a width of one screen pixel.
  line_width = ceil(getPointsPerScreenPixel);
  
  % want hg objects created by lineSymbol to be created but not visible until
  % client is ready to display line.
  set(h_group,'Visible','off');
   
  h_under_line = line(...
    'LineStyle', '-', ...
    'LineWidth', 2*line_width, ...
    'HitTest', 'on', ...
    'Parent', h_group,...
    'Color','w',...
    'Tag','bottom line',...
    'ButtonDownFcn', translateLineFcn);

  h_top_line = line(...
    'LineStyle', '-', ...
    'LineWidth', line_width, ...
    'HitTest', 'on', ...
    'Parent', h_group,...
    'Tag','top line',...
    'ButtonDownFcn', translateLineFcn);
  
   h_end_points(1) = line('Marker','square',...
                          'MarkerSize',6,...
                          'HitTest','on',...
                          'Parent',h_group,...
                          'ButtonDownFcn', resizeLineFcn,...
                          'Tag','end point 1');
   
   h_end_points(2) = line('Marker','square',...
                          'MarkerSize',6,...
                          'HitTest','on',...
                          'Parent',h_group,...
                          'ButtonDownFcn', resizeLineFcn,...
                          'Tag','end point 2');
  
  draw_API.setColor         = @setColor;
  draw_API.getColor         = @getColor;
  draw_API.updateView       = @updateView;
  draw_API.getHitPoint      = @getHitPoint;
  draw_API.setVisible       = @setVisible;
  draw_API.isEndPoint       = @isEndPoint;
  draw_API.isLineBody       = @isLineBody;
  
  setupPointerManagement();

  %----------------------
  function setVisible(TF)
      
      if TF
          set(h_group,'Visible','on');
          
      else
          set(h_group,'Visible','off');
      end
      
  end
  
  %-------------------
  function setColor(c)
    if ishghandle(h_top_line)
      set([h_top_line, h_end_points],'Color', c);
    end
  end

  %--------------------
  function c = getColor
     
      c = get(h_top_line,'Color');
      
  end

  %----------------------------
  function updateView(position)

        if ~ishghandle(h_group)
            return;
        end
        
        x_pos = position(:,1);
        y_pos = position(:,2);
        
        line_handles=[h_under_line, h_top_line];

        if ~isequal(get(h_top_line, 'XData'), x_pos) || ...
                ~isequal(get(h_top_line, 'YData'), y_pos)
            set(line_handles,'XData', x_pos,...
                'YData', y_pos);           
            
            set(h_end_points(1),'XData',x_pos(1),'YData',y_pos(1));
            set(h_end_points(2),'XData',x_pos(2),'YData',y_pos(2));
            
        end

  end
    
  %-------------------------------
  function hit_point = getHitPoint(h_selected_point)
    
      hit_point = [false false];
      hit_point(h_selected_point == h_end_points) = true;
      
  end
    
  %------------------------------
  function setupPointerManagement

      h_fig = iptancestor(h_group,'figure');
      iptPointerManager(h_fig)
      
      enterFcnLine = @(f,cp) set(f, 'Pointer', 'fleur');
      enterFcnEndPoints = @(f,cp) set(f,'Pointer','hand');
      
      iptSetPointerBehavior(h_top_line, enterFcnLine);
      iptSetPointerBehavior(h_end_points, enterFcnEndPoints);

  end
    
end %end lineSymbol