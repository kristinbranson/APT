function placement_aborted = manageInteractivePlacement(varargin)
%manageInteractivePlacement Manage interactive placement of new graphics object.    
%   placement_aborted = manageInteractivePlacement(h_axes,h_new,placementFcn)
%   manages interactive placement of a graphics object specified by
%   h_new into an axes h_axes. placementFcn is a function provided by clients
%   that specifies how placement of a graphics object should take place.
%      
%   placementFcn is a handle to a function that expects 2 input arguments and
%   1 output argument. The function signature of placementFcn is:
%      
%       function completed = placementFcn(x_pos,y_pos)
%      
%   where x_pos and y_pos are scalars which specify the x,y position in
%   spatial coordinates where a user initially clicked. The output parameter
%   completed signals whether or not placement is completed.
%    
%   placement_aborted = manageInteractivePlacement(...,buttonUpFcn)
%   allows clients to provide a custom buttonUpFcn. The function signature of buttonUpFcn is:
%    
%   function completed = buttonUpFcn()
%    
%   By default, placement of ROIs ends on buttonUp.    
       
%   Copyright 2006-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.16 $ $Date: 2011/03/28 04:32:55 $
    
      h_axes = varargin{1};
      h_new = varargin{2};
      placementFcn = varargin{3};
      h_fig = iptancestor(h_axes,'figure');
 
      if nargin == 4
          buttonUpFcn = varargin{4};
      else
          % By default, placement will only start and end from left click
          % buttonDown/buttonUp.
          buttonUpFcn = @() strcmp(get(h_fig,'SelectionType'),'normal');
      end
     
      % Bring the figure containing the specified axes into focus.
      figure(h_fig);
      
      % flag declared at function scope which tracks whether or not interactive
      % placement was aborted by closing figure or pressing escape key.
      placement_aborted = false;
    
      % Store hg object returned by hit test during figure buttonDownFcn at
      % function scope. Want to prevent buttonDown interactions with
      % existing hg objects during interactive placement. Temporarily set
      % buttonDownFcn of hit object to empty until buttonDown functions have
      % ceased firing, then revert back to cached callback function.
      h_hit = [];
      hit_callback_list = {};
      
      % Need to disable axes stretch-to-fit behavior during interactive placement
      % to ensure that axes resizing will not move object being placed
      % out from under the current mouse position. Cache the current
      % state of the XLimMode and YLimMode so that we can revert back
      % once placementFcn has executed.
      x_lim_mode_old = get(h_axes,'XLimMode');
      y_lim_mode_old = get(h_axes,'YLimMode');

      % Order in which WindowButtonDownFcn callbacks are added matters. Want
      % buttonDown interactions with children of figure to be blocked before
      % beginning placement. Callbacks are evaluated in order they were
      % added to callback list.
      block_hit_obj_id = iptaddcallback(h_fig,'WindowButtonDownFcn',@blockHitButtonDownFcn);
      button_down_id = iptaddcallback(h_fig,'WindowButtonDownFcn',@buttonDownPlaceObject);
      button_up_id = [];
      
      escape_key_id = iptaddcallback(h_fig,'WindowKeyPressFcn', @wireEscapeKey);
      close_figure_id = iptaddcallback(h_fig,'CloseRequestFcn',@abortPlacement);
      
      % Temporarily set up pointer behavior over axes while in interactive
      % placement mode
	  iptPointerManager(h_fig,'Enable');
      oldPointerBehavior = iptGetPointerBehavior(h_axes);
      
      % cell array of pointer managed hg objects and their associated
      % pointer manager structure. cell array is of form
      % {h_obj1,pManager1;h_obj2,pManager2;...h_objN,pManagerN}
      cached_pointer_behavior = cell(1,2);
     
      % Turn off pointer management in all child objects of the axes
      unsetChildPointerManagement();

      % Turn on the crosshair affordance to signal to the user that they
      % can begin interactive placement over the axes.
      iptSetPointerBehavior(h_axes,@(hfig,cp) set(hfig,'Pointer','crosshair'));
 
      % Protect against other callbacks deleting the object while placement
      delete_listener = iptui.iptaddlistener(h_new,'ObjectBeingDestroyed',@abortAndResume); %#ok<NASGU>
      
      % Wait until user clicks inside axes area to begin interactive placement
      % before continuing execution.
      uiwait(h_fig);
      
      % Restore buttonDownFcn of whatever was clicked on during interactive
      % placement.
      for i = 1:length(h_hit)
		if ishghandle(h_hit(i))  
			set(h_hit(i),'ButtonDownFcn',hit_callback_list{i});
		end
      end
      
      resetChildPointerManagement();
      
	  % Wrap inside conditional to prevent errors in case figure was
	  % closed. If figure was closed, h_axes is no longer a valid handle.
	  if ishghandle(h_axes)
		  % Restore previous pointer behavior of axes.
		  iptSetPointerBehavior(h_axes,oldPointerBehavior);
          % The purpose of this is to call the nested function updatePointer
          % within the pointerManager. We already know that at this point
          % in the code the pointer manager is enabled.
          iptPointerManager(h_fig,'enable');
          
          % Restore previous XLimMode and YLimMode
          set(h_axes,'XLimMode',x_lim_mode_old,'YLimMode',y_lim_mode_old);
	  end
      
      iptremovecallback(h_fig,'WindowButtonDownFcn',block_hit_obj_id);
      iptremovecallback(h_fig,'WindowButtonDownFcn',button_down_id);
      iptremovecallback(h_fig,'WindowButtonUpFcn',button_up_id);
      iptremovecallback(h_fig,'WindowKeyPressFcn',escape_key_id);
      iptremovecallback(h_fig,'CloseRequestFcn',close_figure_id);
      clear delete_listener;
      
      %-----------------------------------   
      function unsetChildPointerManagement
      
          h_all_children = findall(get(h_axes,'Children'));
          
          % Want to turn off cursor management on all descendents of the axes
          % except descendents of the object currently being placed.
          h_existing_children = setdiff(h_all_children,get(h_new,'Children'));
          
          for k = 1:length(h_existing_children);
              if ~isempty(iptGetPointerBehavior(h_existing_children(k)));
          
                  cached_pointer_behavior{end+1,1} = h_existing_children(k); %#ok
                  cached_pointer_behavior{end,2} = ...
                      iptGetPointerBehavior(h_existing_children(k));
                                             
                  iptSetPointerBehavior(h_existing_children(k),[]);
                  
              end
          end
              
      end
            
      %-----------------------------------
      function resetChildPointerManagement
          for k = 1:size(cached_pointer_behavior,1)
              iptSetPointerBehavior(cached_pointer_behavior{k,1},...
                                    cached_pointer_behavior{k,2});
          end
      end
      
      %---------------------------------------
      function blockHitButtonDownFcn(varargin)
              
      % Because windowButtonDownFcn always fires before children buttonDown
      % functions, setting child buttonDown functions to empty during
      % windowButtonDown function prevents children buttonDown functions
      % from firing.
          h_hit_test = hittest(h_fig);
  
          hit_callback = get(h_hit_test,'ButtonDownFcn');
          % If this is first time HG object has been clicked on, store
          % handle of hit object and set ButtonDown to empty to be restored
          % after interactive placement is completed.
          if ~isempty(hit_callback)
            h_hit(end+1) = h_hit_test;  
            hit_callback_list{end+1} = hit_callback;
            set(h_hit_test,'ButtonDownFcn','');
          end
           
      end %blockHitButtonDownFcn
      
      %---------------------------------------
      function buttonDownPlaceObject(~,evt)
          
          
      
         [x_init,y_init] = getCurrentPoint(h_axes);
         
         % With large images or very fast buttonDown interactions as the
         % figure is first appearing, it is possible that user can initiate
         % a buttonDown callback before uiwait has finished putting the
         % figure into a blocking state. The buttonDown/buttonUp placement
         % code assumes that uiwait has completed before we get into this
         % callback. If that is not the case, we need to throw away the
         % buttonDown event and have the user begin placement again when we
         % are ready.
         blockingSetupIsComplete = strcmp(get(h_fig,'waitstatus'),'waiting');
         
         if ( isOverAxes(evt) && blockingSetupIsComplete )
    
             % Only add button up callback on first click
             if isempty(button_up_id)
                 button_up_id = iptaddcallback(h_fig,'WindowButtonUpFcn',@buttonUpPlaceObject);
             end
             
             set(h_axes,'XLimMode','Manual','YLimMode','Manual');
             completed = placementFcn(x_init,y_init);
             
             if completed
          	   uiresume(h_fig)
             end
         end
                   
      end %startPlacement
      
      %-------------------------------------
      function buttonUpPlaceObject(varargin)
         
          completed = buttonUpFcn();
          if completed
              uiresume(h_fig)
          end
          
      end
      
      %-------------------------------
      function wireEscapeKey(~,ed)
          
        switch (ed.Key)
            case {'delete','escape','backspace'}
              abortAndResume();
        end
          
      end %wireEscapeKey
     
      %--------------------------------
      function abortAndResume(varargin)
          % needs varargin because can be called directly or as a callback
          abortPlacement();
          uiresume(h_fig);
          
      end %abortAndResume
     
     %--------------------------------
     function abortPlacement(varargin)
 
     % When escape is pressed, need to clean up hggroup that was parented to
     % axes. Return empty to communicate that creation of rectangle
     % was aborted.
         if ishghandle(h_new)
             delete(h_new);
         end
         placement_aborted = true;
         
     end % abortPlacement
     
     %------------------------------
     function over = isOverAxes(evt)
        
        
         % If we are using HG2, the object that was clicked on is contained
         % in evt.Source. In HG1, we have to use hittest to determine this
         % information because evt is empty.
         if feature('HGUsingMATLABClasses')
            hit_obj = evt.HitObject;
         else 
            hit_obj = hittest(h_fig);
         end
         
         hit_obj_parent_axes = ancestor(hit_obj,'axes');
         over = ~isempty(hit_obj_parent_axes) &&...
                 ( hit_obj_parent_axes == h_axes );
         
     end % isOverAxes

 end % manageInteractivePlacement
