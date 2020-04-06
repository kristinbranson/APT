function api = basicPolygon(h_group,draw_api,positionConstraintFcn)
%basicPolygon creates the common API shared by imfreehand and impoly.
%   API = basicPolygon(H_GROUP,DRAW_API) creates a base API for use in
%   defining a draggable polygon. H_GROUP specifies the hggroup that
%   contains the polygon or freehand ROI. DRAW_API is the draw_api used by
%   an ROI. basicPolygon returns an API. positionConstraintFcn is a function
%   handle used to constrain the position of an ROI.
    
%   Copyright 2007-2010 The MathWorks, Inc.    
%   $Revision: 1.1.6.14 $  $Date: 2010/09/24 14:32:29 $
     
  h_fig = iptancestor(h_group,'figure');
  h_axes = iptancestor(h_group,'axes');
  
  position = [];
  
  % Pattern for set associated with callbacks that get called as a
  % result of the set.
  insideSetPosition = false;
      
  dispatchAPI = roiCallbackDispatcher(@getPosition);
    
  % This is a workaround to HG bug g349263. There are problems with the figure
  % selection mode when both the hggroup and its children have a
  % buttonDownFcn. Need the hittest property of the hgobjects defined in
  % wingedRect to be on to determine what type of drag action to take.  When
  % the hittest of hggroup children is on, the buttonDownFcn of the hggroup
  % doesn't fire. Instead, pass buttonDownFcn to children inside the appdata of
  % the hggroup.
  setappdata(h_group,'buttonDown',@startDrag);
  
  setappdata(h_group,'ButtonDownListener',...
      iptui.iptaddlistener(h_fig,...
      'WindowMousePress',@cacheCurrentPoint));
  current_mouse_pos = [];
  
  draw_api.initialize(h_group);
  
  % In the other ROI tools, the initial color is defined in
  % createROIContextMenu. It is necessary to create the context menu after
  % interactive placement for impoly/imfreehand, so we need to initialize color here.
  color_choices = iptui.getColorChoices();
  draw_api.setColor(color_choices(1).Color);
        
  % Alias updateView.
  updateView = draw_api.updateView;
  
  api.startDrag                 = @startDrag;
  api.addNewPositionCallback    = dispatchAPI.addNewPositionCallback;
  api.removeNewPositionCallback = dispatchAPI.removeNewPositionCallback;
  api.setPosition               = @setPosition;
  api.setConstrainedPosition    = @setConstrainedPosition;
  api.getPosition               = @getPosition;
  api.delete                    = @deletePolygon;
  api.getPositionConstraintFcn  = @getPositionConstraintFcn;
  api.setPositionConstraintFcn  = @setPositionConstraintFcn;
  api.updateView                = draw_api.updateView;
  api.setVisible                = draw_api.setVisible;
  api.setClosed                 = draw_api.setClosed;
  
  %----------------------------------
    function cacheCurrentPoint(~,ed)
        % This function caches the CurrentPoint field of the event data of the
        % WindowButtonDownEvent at function scope. The current point passed in the
        % event data is guaranteed to always be in pixel units. The current point
        % cached at function scope is used in ipthittest to ensure that the cursor
        % affordance shown and the button down action taken are consistent.

        if feature('HGUsingMATLABClasses')
            current_mouse_pos = ed.Point;
        else
            current_mouse_pos = ed.CurrentPoint;
        end

    end
  
  %-------------------------------
  function deletePolygon(varargin)
     
      if ishghandle(h_group)
        delete(h_group)
      end
      
  end %deletePolygon
       
  %-------------------------
  function pos = getPosition
    
      pos = position;    
    
  end %getPosition

  %------------------------
  function setPosition(pos)
     
      % Pattern to break recursion
      if insideSetPosition
          return
      else
          insideSetPosition = true;
      end
      
      position = pos;
      updateView(pos);
      
      % User defined newPositionCallbacks may be invalid. Wrap
      % newPositionCallback dispatches inside try/catch to ensure that
      % insideSetPosition will be unset if newPositionCallback errors.
      try
          dispatchAPI.dispatchCallbacks('newPosition');
      catch ME
          insideSetPosition = false;
          rethrow(ME);
      end
      
      % Pattern to break recursion
      insideSetPosition = false;
      
  end %setPosition
  
  %-----------------------------------
  function setConstrainedPosition(pos)
     
      pos = positionConstraintFcn(pos);
      setPosition(pos);
      
  end %setConstrainedPosition
  
  %---------------------------------
  function setPositionConstraintFcn(fun)
  
      positionConstraintFcn = fun;
    
  end %setPositionConstraintFcn
    
  %---------------------------------
  function fh = getPositionConstraintFcn
      
      fh = positionConstraintFcn;
  
  end %getPositionConstraintFcn
  
  %---------------------------
  function startDrag(hSrc,~)
  
    if feature('HGUsingMATLABClasses')
        hit_obj = hSrc;
    else
        hit_obj = hittest(h_fig,current_mouse_pos);
    end
 
    hit_vertex = strcmp(get(hit_obj,'tag'),'impoly vertex');
    
    % g392299 Workaround hg hittest issue where the buttonDownFcn of the
    % polygon is firing even though the current hittest object is the
    % vertex. Ensure that cursor management stays in sync with actual drag
    % gesture.
    if hit_vertex
       startDrag = get(hit_obj,'ButtonDownFcn');
       startDrag(hit_obj,[]);
       return;  
    end
      
    mouse_selection = get(h_fig,'SelectionType');
    is_normal_click = strcmp(mouse_selection,'normal');
     
    if ~is_normal_click
        return
    end
     
    start_position = getPosition();
    
    [start_x,start_y] = getCurrentPoint(h_axes);
              	 
    % Disable the figure's pointer manager during the drag.
    iptPointerManager(h_fig, 'disable');
  
    drag_motion_callback_id = iptaddcallback(h_fig, ...
                                             'WindowButtonMotionFcn', ...
                                             @dragMotion);
    
    drag_up_callback_id = iptaddcallback(h_fig, ...
                                             'WindowButtonUpFcn', ...
                                             @stopDrag);
    									 	 
      %----------------------------
      function dragMotion(varargin)
        
          if ~ishghandle(h_axes)
              return;
          end
        
          [new_x,new_y] = getCurrentPoint(h_axes);      
          delta_x = new_x - start_x;
          delta_y = new_y - start_y;
          
          candidate_position = bsxfun(@plus, start_position, [delta_x, delta_y]);          
          new_position = positionConstraintFcn(candidate_position);
            
          % Only fire setPosition/callback dispatch machinery if position has
          % actually changed
          if ~isequal(new_position,getPosition())
              setPosition(new_position)
          end
      
      end
      
      %--------------------------
      function stopDrag(varargin)
            
            dragMotion();
            
            iptremovecallback(h_fig, 'WindowButtonMotionFcn', ...
                              drag_motion_callback_id);
            iptremovecallback(h_fig, 'WindowButtonUpFcn', ...
                              drag_up_callback_id);
            
            % Enable the figure's pointer manager.
            iptPointerManager(h_fig, 'enable');
        
      end % stopDrag
      	
  end %startDrag
    
end % basicPolygon

