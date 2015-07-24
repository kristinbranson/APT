function export_successful = imExportToWorkspace(hParent)
%imExportToWorkspace Export a displayed image to the workspace  
%  STATUS = imExportToWorkspace(HPARENT) brings up a dialog and prompts the user
%  to specify a variable name for storing the image data to the base workspace.
%  If the image is an indexed image, the colormap may also be specified.
%  EXPORT_SUCCESSFUL returns true if the image was indeed exported to the
%  workspace or false if the operation was canceled.

% Copyright 2004-2011 The MathWorks, Inc.


% Get the image type
  [image_data,flag] = getimage(hParent);
  
  % Determine if we are working with an indexed image.
  % The FLAG output from GETIMAGE returns 1 for indexed images
  % If so, we need the cmap
  image_needs_cmap = flag == 1;
  
  % Define the prompts
  prompts = {getString(message('images:imExportToWorkspace:imageVariableName')),...
      getString(message('images:imExportToWorkspace:colormapVariableName'))};
  
  % Define the title
  dlg_title = getString(message('images:imExportToWorkspace:dialogTitle'));
  num_of_lines_per_prompt = 1;
  
  switch flag
   case 1
    default_answer = {'X','map'};
   case {2,3}
    default_answer = {'I'};
   case 4
    default_answer = {'RGB'};
   case 5
    default_answer = {'BW'};
   otherwise
    % no image in figure, hParent
    export_successful = false;
    return
  end
  
  export_successful = false;
  
  while ~export_successful
    
    % get the original class of image data from the 
    % image handle
    hIm = imhandles(hParent);
    img_model = getimagemodel(hIm);
    image_class = getClassType(img_model);
    % this is OK with the compiler because image_class fcns are built-in fcns.
    image_data = feval(image_class,image_data);
    
    % Display the input dialog using INPUTDLG from MATLAB
    % If the image requires a colormap, two prompts are displayed
    if image_needs_cmap

      cmap_data = get(hParent,'Colormap');
      data_to_assign = {image_data, cmap_data};
      outvar_user = inputdlg(prompts,dlg_title,num_of_lines_per_prompt,default_answer);
      
    else
      
      data_to_assign = {image_data};
      outvar_user = inputdlg(prompts{1},dlg_title,num_of_lines_per_prompt,default_answer);
    
    end
    
    user_pressed_cancel = isempty(outvar_user);
    if user_pressed_cancel
      export_successful = false;
      return;
    end
      
    ind_with_values = ~strcmp('',outvar_user);
    
    % we use whatever string the user specified the next time around
    % so we cache it.
    default_answer = outvar_user;
    data_to_assign = data_to_assign(ind_with_values);
    outvar_user = outvar_user(ind_with_values);
    
    export_successful = assignVarInBaseWorkspace(outvar_user, data_to_assign);
    
  end %while

end % imExportToWorkspace


%-----------------------------------------------------------------------
function successful = assignVarInBaseWorkspace(variable_name, variable_data)
  % Assigns variable_data to variable_name in the base workspace
  % Both arguments are cell arrays of the same length

  if isempty(variable_name) || isempty(variable_data)
    str1 = getString(message('images:imExportToWorkspace:noVariableNames'));
    uiwait(errordlg(str1,getString(message('images:imExportToWorkspace:exportToWorkspace'))));
    successful = false;
    return
  end
  
  % Grab all the workspace variables and store their names in a cell
  % array.
  ws_vars = evalin('base','whos');
  [ws_var_names{1:length(ws_vars)}] = deal(ws_vars.name);
  
  user_response = '';
  valid_var_name = genvarname(variable_name);
  
  for n = 1:length(variable_name)

    var_already_exists = any(strcmp(valid_var_name{n}, ws_var_names),2);
    
    user_spec_var_name_changed = ~strcmpi(variable_name{n}, valid_var_name{n});
    
    skip_next_check = false;
    
    if user_spec_var_name_changed
        
      str = getString(message('images:imExportToWorkspace:invalidVarName',...
                      variable_name{n},...
                      valid_var_name{n}));
                  
      question_str = getString(message('images:imExportToWorkspace:useSuggestedName',...
                      str));
                      
      user_response = questdlg(question_str,getString(message('images:imExportToWorkspace:nameChange')),'Yes');
      
      if strcmpi(user_response,'no')
        skip_next_check = true;
      end
      
    end
    
    
    if ~isempty(var_already_exists) && var_already_exists && ~skip_next_check
      
      question_str = getString(message('images:imExportToWorkspace:variableInWS',valid_var_name{n}));  
      
      question_str = sprintf('%s\n%s',question_str,...
          getString(message('images:imExportToWorkspace:wantToOverwrite')));
      
      user_response = questdlg(question_str,...
          getString(message('images:imExportToWorkspace:variableAlreadyExists')),...
          'No');
      
    end
    
    %Handle user response appropriately
    switch lower(user_response)
     case {'yes',''} % user answered yes or didn't have to specify any responses
      % good! now do it again
     case 'no'
      successful = false;
      return;
     case 'cancel'
      successful = true;
      return
    end
    
  end
  
  for n = 1:length(variable_name)
    
    assignin('base',valid_var_name{n}, variable_data{n});
    successful = true;
    
  end

  
end % assignVarInBaseWorkspace
