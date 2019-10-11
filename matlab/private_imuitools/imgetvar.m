function [image_data,cmap_data,image_var_name,cmap_var_name,user_canceled] = ...
      imgetvar(varargin)
%IMGETVAR Image variable open dialog box.
%  [IMAGE_DATA,CMAP_DATA,IMAGE_VAR_NAME,CMAP_VAR_NAME,USER_CANCELED] =
%  IMGETVAR(HFIG) displays a dialog box for selecting an image variable from the
%  MATLAB workspace.
%
%  The user may also specify a colormap either by entering a valid MATLAB
%  expression or opening a separate dialog box.
%  
%  The selected image data, colormap data, and their variable names are returned
%  in IMAGE_DATA, CMAP_DATA, IMAGE_VAR_NAME and CMAP_VAR_NAME.  If the user
%  closes the dialog or presses the Cancel button, USER_CANCELED will return
%  TRUE.  Otherwise USER_CANCELED will return FALSE.
%
%  The listed workspace variables are filtered based on the selection in the
%  "Filter" drop-down menu.  The drop-down menu choices are:
%  
%         All           variables that qualify as binary, intensity or truecolor
%         Binary        M-by-N logical variables
%         Indexed       M-by-N variables of *standard types with integer values
%         Intensity     M-by-N variables of *standard types and int16
%         Truecolor     M-by-N-by-3 variables of *standard types and int16
%
%  The *standard supported MATLAB classes (types) for image data are "double",
%  "single", "uint8", "uint16", "int16".  The exceptions are binary images that
%  are logical types and indexed images that do not supported int16 types.

%   Copyright 2004-2011 The MathWorks, Inc.
%   $Revision: 1.1.8.7 $  $Date: 2011/10/11 15:49:30 $
  
  
  narginchk(0, 1);
    
  if nargin == 1
    % client needs to be a figure
    hFig = varargin{1};
    iptcheckhandle(hFig,{'figure'},mfilename,'HCLIENT',1);
  end
  
  % Output variables for function scope
  image_data = [];
  cmap_data = [];
  image_var_name = '';
  cmap_var_name = '';
  user_canceled = true;
  
  hImportFig = figure('Toolbar','none',...
                      'Menubar','none',...
                      'NumberTitle','off',...
                      'IntegerHandle','off',...
                      'Tag','imImportFromWS',...
                      'Visible','off',...
                      'HandleVisibility','callback',...
                      'Name',getString(message('images:privateUIString:importFromWorkspace')),...
                      'WindowStyle','modal',...
                      'Resize','off');
  
  
  % Layout management
  fig_height = 360;
  fig_width  = 300;
  fig_size = [fig_width fig_height];
  left_margin = 10;
  right_margin = 10;
  bottom_margin = 10;
  spacing = 5;
  default_panel_width = fig_width - left_margin -right_margin;
  button_size = [60 25];  
  b_type = 'none';
  
  last_selected_value = [];
  
  set(hImportFig,'Position',getCenteredPosition);
  
  % Get workspace variables and store variable names for 
  % accessibility in nested functions
  workspace_vars = evalin('base','whos');
  num_of_vars = length(workspace_vars);
  [all_var_names{1:num_of_vars}] = deal(workspace_vars.name);

  custom_bottom_margin = fig_height - 50;
  custom_top_margin = 10;
  hFilterPanel = createFilterMenu;

  hCMapPanelObjs = [];
  
  custom_bottom_margin = 10;
  custom_top_margin = fig_height - custom_bottom_margin - button_size(2);
  createButtonPanel;

  custom_bottom_margin = custom_bottom_margin + 20 + 2*spacing;
  custom_top_margin = 50 + 2*spacing;
  display_panels(1) = createImportPanel('All');
  display_panels(2) = createImportPanel('Binary');  
  display_panels(3) = createImportPanel('Indexed');
  display_panels(4) = createImportPanel('Intensity');
  display_panels(5) = createImportPanel('RGB');  
  num_of_panels = length(display_panels);
  

  % force to run callback after creation to do some filtering
  hAllList = findobj(display_panels(1),'Tag','allList');
  if ~isempty(get(hAllList,'String'))
    listSelected(hAllList,[]);
  end

  set(display_panels(1),'Visible','on');
  set(hImportFig,'Visible','on');
  
  all_list_boxes = findobj(hImportFig,'Type','uicontrol','Style','listbox');
  set(all_list_boxes,'BackgroundColor','white');

  % This blocks until the user explicitly closes the tool.
  uiwait(hImportFig);
 
  %-------------------------------
  function pos = getPanelPos
     % Returns the panel Position based on the custom_bottom_margin
     % and custom_top_margin.  Useful for layout mngment
     height = fig_height - custom_bottom_margin - custom_top_margin;
     pos = [left_margin, custom_bottom_margin, default_panel_width, height];
  end
  
  %--------------------------
  function showPanel(src,evt) %#ok<INUSD>
    % Makes the panel associated with the selected image type visible
      
    ind = get(src,'Value');
    
    set(display_panels(ind),'Visible','on');
    set(display_panels(ind ~= 1:num_of_panels),'Visible','off');

    % if image is rgb disable the colormap selection button
    is_rgb_panel = (ind == 4);
    
    disableCmapForRGBVar(is_rgb_panel);
     
  end %showPanel
  
  %----------------------------------
  function pos = getCenteredPosition
  % Returns the position of the import dialog 
  % centered on the screen.
    
    old_units = get(0,'Units');
    set(0,'Units','Pixels');
    screen_size = get(0,'ScreenSize');
    set(0,'Units', old_units);
    
    lower_left_pos = 0.5 * (screen_size(3:4) - fig_size);
    pos = [lower_left_pos fig_size];
  end % getCenteredPosition
  

  %----------------------------------
  function hPanel = createFilterMenu
    % Creates the image type selection panel
      
    panelPos = getPanelPos;
    
    hPanel = uipanel('parent',hImportFig,...
                     'Units','Pixels',...
                     'Tag','filterPanel',...
                     'BorderType',b_type,...
                     'Position',panelPos);

    setChildColorToMatchParent(hPanel, hImportFig);
    
    hFilterLabel = uicontrol('parent',hPanel,...
                                'Style','Text',...
                                'String',getString(message('images:privateUIString:filterLabel')),...
                                'HorizontalAlignment','left',...
                                'Units','pixels');
    
    label_extent = get(hFilterLabel,'extent');
    posY = bottom_margin;
    label_position = [left_margin, posY, label_extent(3:4)];
    
    set(hFilterLabel,'Position',label_position);
    
    setChildColorToMatchParent(hFilterLabel,hPanel);
    
    image_type_str = {getString(message('images:privateUIString:imageTypeAll')),...
                      getString(message('images:privateUIString:imageTypeBinary')),...
                      getString(message('images:privateUIString:imageTypeIndexed')),...
                      getString(message('images:privateUIString:imageTypeIntensity')),...
                      getString(message('images:privateUIString:imageTypeTruecolor'))};

    max_width = panelPos(3)-left_margin-right_margin-label_extent(3)-spacing;
    pmenu_width = min([panelPos(3)-label_extent(3)-left_margin*2,...
                       max_width]);
    
    pmenu_pos = [left_margin + label_extent(3) + spacing,...
                 posY,pmenu_width, 20];
    
    hFilterMenu = uicontrol('parent',hPanel,...
                               'Style','popupmenu',...
                               'Tag','filterPMenu',...
                               'Units','pixels',...
                               'Callback',@showPanel,...
                               'String',image_type_str,...
                               'Position',pmenu_pos);
    
    setChildColorToMatchParent(hFilterMenu,hPanel);
    
    if ispc
      % Sets the background color for the popup menu to be white
      % This matches with how the imgetfile dialog looks like
      set(hFilterMenu,'BackgroundColor','white');
    end
    
  end %createFilterMenu

  %----------------------------------
  function hPanel = createImportPanel(im_type)
    % Panel that displays all qualifying (image) workspace
    % variables

    is_indexed_type = strcmpi(im_type,'indexed'); 
    
    panelPos = getPanelPos;
    
    hPanel = uipanel('parent',hImportFig,...
                     'Tag',sprintf('%sPanel',lower(im_type)),...
                     'Units','pixels',...
                     'BorderType',b_type,...
                     'Position',panelPos,...
                     'Visible','off');
    
    setChildColorToMatchParent(hPanel,hImportFig);
    
    hLabel = uicontrol('parent',hPanel,...
                       'Style','text',...
                       'Units','pixels',...
                       'HorizontalAlignment','left',...
                       'String',getString(message('images:privateUIString:imgetvarVariablesLabel')));
    
    setChildColorToMatchParent(hLabel,hPanel);
    
    label_extent = get(hLabel,'Extent');
    label_posX = left_margin;
    label_posY = panelPos(4) - label_extent(4) - spacing;
    label_width = label_extent(3);
    label_height = label_extent(4);
    label_position = [label_posX label_posY label_width label_height];
    
    set(hLabel,'Position',label_position);
    
    cmap_panel_height = 0;    
    if is_indexed_type 
      
      % create colormap panel
      hCMapPanelObjs = createCMapPanel(hPanel);
      cmap_panel_height = button_size(2)+2*spacing;

    end
    
    
    hVarList = uicontrol('parent',hPanel,...
                         'Style','listbox',...
                         'fontname','Courier',...
                         'Value',1,...
                         'Units','pixels',...
                         'Tag',sprintf('%sList',lower(im_type)));
    
    setChildColorToMatchParent(hVarList,hPanel);
    
    list_posX = left_margin;
    list_posY = bottom_margin + cmap_panel_height;
    list_width = panelPos(3) - 2*list_posX;
    list_height = panelPos(4) - list_posY - label_height - spacing;
    list_position = [list_posX list_posY list_width list_height];
    
    set(hVarList,'Position',list_position);
    set(hVarList,'Callback',@listSelected);

    varInd = filterWorkspaceVars(workspace_vars,im_type);
    displayVarsInList(workspace_vars(varInd),hVarList);
        
  end %createImportPanel

  %-----------------------------
  function listSelected(src,evt) %#ok<INUSD>
   % callback for the  list boxes
   % we disable the colormap panel controls for an RGB image

    ind = get(src,'Value');
    list_str = get(src,'String');
    
    if isempty(list_str)
      return
    else
      sel_str = list_str{ind};
      sel_str = strtok(sel_str);
      
      % get index of specified variable from the list of workspace variables
      var_ind = find(strcmp(sel_str,all_var_names));
      
      % get the size off the variable
      tmp_size = workspace_vars(var_ind).size;
      
      is_rgb_var = (length(tmp_size) == 3 && tmp_size(3) == 3);

      disableCmapForRGBVar(is_rgb_var);
    end
    
    double_click = strcmp(get(hImportFig,'SelectionType'),'open');
    clicked_same_list_item = last_selected_value == ind;
    
    if double_click && clicked_same_list_item && getVars      
      user_canceled = false;
      close(hImportFig);
    else
      set(hImportFig,'SelectionType','normal');
    end
    
    last_selected_value = ind;
    
  end %listSelected
  
  %------------------------------------------------
  function createButtonPanel
    % panel containing the OK and Cancel buttons
      
    panelPos = getPanelPos;
    hButtonPanel = uipanel('parent',hImportFig,...
                           'Tag','buttonPanel',...
                           'Units','pixels',...
                           'Position',panelPos,...
                           'BorderType',b_type);
        
    setChildColorToMatchParent(hButtonPanel,hImportFig);
    
    % add buttons
    button_strs_n_tags = {'OK', 'okButton';...
                          'Cancel','cancelButton'};
    
    num_of_buttons = length(button_strs_n_tags);

    button_spacing = (panelPos(3)-(num_of_buttons * button_size(1)))/(num_of_buttons+1);
    posX = button_spacing;
    posY = 0;
    buttons = zeros(num_of_buttons,1);
    
    for n = 1:num_of_buttons
      buttons(n) = uicontrol('parent',hButtonPanel,...
                             'Style','pushbutton',...
                             'String',button_strs_n_tags{n,1},...
                             'Tag',button_strs_n_tags{n,2});
      
      setChildColorToMatchParent(buttons(n), hButtonPanel);
      
      set(buttons(n),'Position',[posX, posY, button_size]);
      set(buttons(n),'Callback',@doButtonPress);
      posX = posX + button_size(1) + button_spacing;
      
    end
        
  end % createButtonPanel
  
  %------------------------------
  function doButtonPress(src,evt) %#ok<INUSD>
    % call back function for the OK and Cancel buttons
    tag = get(src,'tag');
    
    switch tag
     case 'okButton'
      
      if getVars
        user_canceled = false;
        close(hImportFig);
      end
      
     case 'cancelButton'
      image_var_name = '';
      cmap_var_name = '';
      close(hImportFig);
      
    end
    
  end %doButtonPress
  
  %------------------------------------------------
  function status = getVars
    
    SUCCESS = true;
    FAILURE = false;
    
    status = SUCCESS;
    
    % get the listbox in the active display panel
    im_type_menu = findobj(hFilterPanel,'tag','filterPMenu');
    im_type_ind = get(im_type_menu,'Value');
    hVarList = findobj(display_panels(im_type_ind),'Type','uicontrol',...
                       'style','listbox');
    list_str = get(hVarList,'String');

    is_indexed_image = strcmpi('indexedList',get(hVarList,'tag'));
    
    % return if there are no variables listed in current panel
    if isempty(list_str)
      hAllVarList = findobj(display_panels(1),'Type','uicontrol','style','listbox');
      all_str = get(hAllVarList,'String');
      if isempty(all_str)
        error_str = getString(message('images:privateUIString:noVariablesErrorStr'));
        
      else
        error_str = getString(message('images:privateUIString:noSelectedVariableStr'));
      end
      errordlg(error_str);
      status = FAILURE;
      return;
    end

    ind = get(hVarList,'Value');

    image_var_name = strtok(list_str{ind});
    
    % see if a colormap string has been specified
    cmap_str = get(hCMapPanelObjs(1),'String');
    colormap_is_specified = ~isempty(cmap_str);
    
    if colormap_is_specified
      % a colormap was specified and the image is not rgb or binary
      
      % check if the variable or function indeed exists
      [cmap_data,eval_passed] = evaluateVariable(cmap_str);
      status = eval_passed;
      
      if ~eval_passed
        return;
      end
      
      sz = size(cmap_data);
      if (length(sz) ~= 2 || sz(2) ~= 3)
        error_str = sprintf('%s %s',cmap_str,getString(message('images:privateUIString:doesNotQualityAsColormapStr')));
        errordlg(error_str);
        status = FAILURE;
        return;
      end

      cmap_var_name = cmap_str;
    elseif is_indexed_image
        % we open an indexed image with no colormap as a grayscale image
        cmap_data = gray(256);
    end

    [image_data, eval_passed] = evaluateVariable(image_var_name);
    status = eval_passed;
    
    
  end %getVars
 
  %-----------------------------------------------
  function disableCmapForRGBVar(set_enable_off)
    % disables the colormap panel contents ("choose colormap" button and edit box)
    % if the input variable name qualifies as an RGB or Binary image.              
    
    if set_enable_off
      set(hCMapPanelObjs,'Enable','off');
      set(hCMapPanelObjs(1),'BackgroundColor',[0.72 0.72 0.72]);
    else
      set(hCMapPanelObjs,'Enable','on');
      set(hCMapPanelObjs(1),'BackgroundColor','white');
    end
    
  end %disableCmapForRGBAndBinaryVar
  
  %------------------------------------------------  
  function hOut = createCMapPanel(hParent)
  
    cmap_panel_pos = [0, 0, default_panel_width,...
                      button_size(2)+spacing*3];
                      
    hCMapPanel = uipanel('parent',hParent,...
                         'Tag','cmapPanel',...
                         'Units','pixels',...
                         'Position',cmap_panel_pos,...
                         'BorderType',b_type);
    
    setChildColorToMatchParent(hCMapPanel,hParent);
    
    hEditCMap = uicontrol('parent',hCMapPanel,...
                          'Style','edit',...
                          'HorizontalAlignment','left',...
                          'Units','pixels',...
                          'Tag','editCmap');
    
    setChildColorToMatchParent(hEditCMap,hCMapPanel);
    
    
    hSelectCMap = uicontrol('parent',hCMapPanel,...
                            'Style','pushbutton',...
                            'String',getString(message('images:privateUIString:chooseColormapButton')),...
                            'Tag','chooseCmap',...
                            'Units','pixels');
    
    setChildColorToMatchParent(hSelectCMap,hCMapPanel);
    
    set(hSelectCMap,'Callback',{@callImChoosecmap hImportFig hEditCMap});

    
    str_extent = get(hSelectCMap,'Extent');
    cmap_button_size = [str_extent(3)+spacing button_size(2)];
    cmap_edit_size = [cmap_panel_pos(3)-left_margin*2-spacing-cmap_button_size(1),...
                     cmap_button_size(2)];
    cmap_edit_posX = left_margin;
    cmap_button_posX = cmap_panel_pos(3)-right_margin-cmap_button_size(1);
    cmap_button_posY = spacing*2;
    cmap_edit_posY = spacing*2;
    cmap_button_position = [cmap_button_posX cmap_button_posY cmap_button_size];
    cmap_edit_position = [cmap_edit_posX cmap_edit_posY cmap_edit_size];
    
    set(hEditCMap,'Position',cmap_edit_position);
    set(hSelectCMap,'Position',cmap_button_position);
    
    set(hEditCMap,'BackgroundColor','white');
    hOut = [hEditCMap, hSelectCMap];
      
  end % createCMapPanel

  
  %---------------------------------------------------------
  function callImChoosecmap(src,evt,hImportFig,hEditCMap) %#ok<INUSL>
      
      % create dummy hg objects
      set(hImportFig,'Colormap',gray(256));
      dummy_ax = axes('parent',hImportFig,...
          'visible','off',...
          'position',[-10 -10 1 1]);
      dummy_im = image(dummy_ax,...
          'cdata',1,...
          'cdatamapping','direct'); %#ok<NASGU>
      
      % create colormap chooser tool and scrub parts of it
      cmap_tool = imcolormaptool(hImportFig);
      % remove toolbar
      delete(findall(cmap_tool,'tag','uitoolbar'));
      % remove the (original) string
      fcn_list = findall(cmap_tool,'tag','cmapFcnList');
      fcn_strs = get(fcn_list,'String');
      
      % Replace the string "gray (original)" with "gray"
      gray_orig_str = strcat(getString(message('images:privateUIString:getVarGray')),...
          ' (',getString(message('images:imcolormaptoolUIStrings:original')),')');
      
      fcn_strs = strrep(fcn_strs,...
                        gray_orig_str,...
                        getString(message('images:privateUIString:getVarGray')));
                    
      set(fcn_list,'String',fcn_strs);
      edit_box = findall(cmap_tool,'tag','cmapEvalEdit');
      edit_link = linkprop([edit_box hEditCMap],'String');
      % block until tool is dismissed (make tool modal)
      set(cmap_tool,'WindowStyle','modal');
      uiwait(cmap_tool);
      
      % clean up tool
      delete(edit_link);
      clear edit_link
      delete(dummy_ax);
      
  end
  
end %imgetvar


%----------------------------------------
function [out, eval_passed] = evaluateVariable(var_name)
  
  eval_passed = true;
  out = [];
  try
    out = evalin('base',sprintf('%s;',var_name));
  catch ME
    errordlg(ME.message)
    eval_passed = false;
  end
  
end %evaluateVariable
