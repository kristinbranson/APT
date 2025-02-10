function set_standard_callback_if_none_bang(gh, controller)
  if isempty(gh) ,
    return
  end
  if ~isscalar(gh) ,
    error('Called with non-scalar gh') ;
  end
  if ~isprop(gh, 'Callback') || ~isempty(gh.Callback) ,
    % Don't want to overwrite an exiting callback
    return
  end
  tag = gh.Tag ;
  if isequal(get(gh,'Type'),'uimenu') ,
    if ~isempty(gh.Children) ,
      % do nothing for menus with submenus
    else
      set(gh,'Callback',@(source,event)(controller.controlActuated(tag,source,event)));
    end
  elseif isequal(get(gh,'Type'),'uicontrol') && ~isequal(get(gh,'Style'),'text') ,
    % set the callback for any uicontrol that is not a text
    set(gh,'Callback',@(source,event)(controller.controlActuated(tag,source,event)));
  end
end
