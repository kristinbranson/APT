function [hPropsPane,parameters] = propertiesGUI(hParent, parameters, filename, selectedBranch)
% propertiesGUI displays formatted editable list of properties
%
% Syntax:
%
%   Initialization:
%    [hPropsPane,parameters] = propertiesGUI(hParent, parameters)
%
%   Run-time interaction:
%    propertiesGUI(hPropsPane, mode, filename, branch)
%
% Description:
%    propertiesGUI processes a list of data properties and displays
%    them in a GUI table, where each parameter value has a unique
%    associated editor.
%
%    propertiesGUI by itself, with no input parameters, displays a demo
%
%    By default, propertiesGUI identifies and processes the following
%    field types: signed, unsigned, float, file, folder, text or string,
%    color, IPAddress, password, date, boolean, cell-array, numeric array,
%    font, struct and class object.
%
% Inputs:
%  Initialization (up to 2 inputs)
%  
%    hParent - optional handle of a parent GUI container (figure/uipanel
%              /uitab) in which the properties table will appear.
%              If missing or empty or 0, the table will be shown in a
%              new modal dialog window; otherwise it will be embedded
%              in the parent container.
%
%    parameters - struct or object with data fields. The fields are
%              processed separately to determine their corresponding cell
%              editor. If parameters is not specified, then the global
%              test_data will be used. If test_data is also empty, then
%              a demo of several different data types will be used.
%
%  Run-time interactions (4 inputs)
%
%   hPropsPane - handle to the properties panel width (see output below)
%
%   mode - char string, 'save' or 'load' to indicate the operation mode
%          to save data or load
%
%   filename - char string of the filename to be saved.  Can be full path
%              or relative.
%
%   branch - char string to the branch to be saved or loaded.  If it is a 
%            sub branch then the items are to be delimited by '.'
%            If branch is empty -> then the full tree data is saved/loaded.
%
% Outputs:
%    hPropsPane - handle of the properties panel widget, which can be
%              customized to display field descriptions, toolbar, etc.
%
%    parameters - the resulting (possibly-updated) parameters struct.
%              Naturally, this is only relevant in case of a modal dialog.
%
%    (global test_data) - this global variable is updated internally when
%              the <OK> button is clicked. It is meant to enable easy data
%              passing between the properties GUI and other application
%              component. Using global vars is generally discouraged as
%              bad programming, but it simplifies component interaction.
%
% Customization:
%    This utility is meant to be used either as stand-alone, or as a
%    template for customization. For example, you can attach a unique
%    description to each property that will be shown in an internal
%    sub-panel: see the customizePropertyPane() and preparePropsList()
%    sub-functions.
%
%    When passing the properties in an input parameters struct, the
%    utility automatically inspects each struct field and assigns a
%    corresponding cell-editor with no description and a field label
%    that reflects the field name. The properties are automatically
%    set as modifiable (editable) and assigned a default callback
%    function (propUpdatedCallback() sub-function).
%    See the demoParameters() sub-function for some examples.
%
%    You can have specific control over each property's description,
%    label,  editability, cell-editor and callback function. See the
%    preparePropsList() sub-functions for some examples. You can add
%    additional cell-editors/renderers in the newProperty() sub-function.
%
%    You can place specific control over the acceptable property values
%    by entering custom code into the checkProp() sub-function.
%
% Future development:
%    1. Improve the editor for time, numeric and cell arrays
%    2. Enable more control over appearance and functionality via 
%       propertiesGUI's input parameters
%    3. Add additional built-in cell editors/renderers: slider, point,
%       rectangle (=position), ...
%
% Example:
%    propertiesGUI;   % displays the demo
%
%    params.name   = 'Yair';
%    params.age    = uint8(41);
%    params.folder = pwd;
%    params.date   = now;
%    params.size.width  = 10;
%    params.size.height = 20;
%    [hPropsPane, params] = propertiesGUI(params);
%
%    % runtime interation:
%    propertiesGUI(hPropsPane, 'save', 'width.mat', 'size.width');
%    propertiesGUI(hPropsPane, 'load', 'width.mat', 'size.width');
%
% Bugs and suggestions:
%    Please send to Yair Altman (altmany at gmail dot com)
%
% Warning:
%    This code heavily relies on undocumented and unsupported Matlab
%    functionality. It works on Matlab 7+, but use at your own risk!
%
%    A technical description of the implementation can be found at:
%    http://undocumentedmatlab.com/blog/propertiesGUI
%    http://undocumentedmatlab.com/blog/jide-property-grids
%    http://undocumentedmatlab.com/blog/advanced-jide-property-grids
%
% Change log:
%    2015-03-12: Fixes for R2014b; added support for matrix data, data save/load, feedback links
%    2013-12-24: Fixes for R2013b & R2014a; added support for Font property
%    2013-04-23: Handled multi-dimensional arrays
%    2013-04-23: Fixed case of empty ([]) data, handled class objects & numeric/cell arrays, fixed error reported by Andrew Ness
%    2013-01-26: Updated help section
%    2012-11-07: Minor fix for file/folder properties
%    2012-11-07: Accept any object having properties/fields as input parameter; support multi-level properties
%    2012-10-31: First version posted on <a href="http://www.mathworks.com/matlabcentral/fileexchange/authors/27420">MathWorks File Exchange</a>
%
% See also:
%    inspect, uiinspect (#17935 on the MathWorks File Exchange)

% License to use and modify this code is granted freely to all interested, as long as the original author is
% referenced and attributed as such. The original author maintains the right to be solely associated with this work.

% Programmed and Copyright by Yair M. Altman: altmany(at)gmail.com
% $Revision: 1.15 $  $Date: 2015/03/12 12:37:46 $
% Minor modifications for new Matlab version compatibility by Allen Lee. 

  % Get the initial data
  global test_data
  isDemo = false;
  if nargin < 2
      try
          isObj = nargin==1;
          [hasProps,isHG] = hasProperties(hParent);
          isObj = isObj && hasProps && ~isHG;
      catch
          % ignore - maybe nargin==0, so no hParent is available
      end
      if isObj
          parameters = hParent;
          hParent = [];
      else
          parameters = test_data;  % comment this if you do not want persistent parameters
          if isempty(parameters)
              % demo mode
              parameters = demoParameters;
          end
          isDemo = true;
      end
  elseif nargin == 4  % check if load or save mode
      mode = parameters;
      if ischar(mode) && ischar(filename) && (ischar(selectedBranch) || iscell(selectedBranch))
          hParent = handle(hParent);
          hFig = ancestor(hParent,'figure');
          %mirrorData = getappdata(hFig, 'mirror');
          hgrid = getappdata(hFig, 'hgrid');
          if strcmp(mode,'load') && isempty(selectedBranch)
              parameters = load(filename, '-mat');
              container = hParent.Parent;
              delete(hParent)
              hPropsPane = propertiesGUI(container,parameters);
          else
              fileIO_Callback(hgrid, selectedBranch, mode, hFig, filename)
          end
      else
          error('propertiesGUI:incorrectUsage', 'Incorrect input parameters in input');
      end
      return
  end

  % Accept any object having data fields/properties
  try
      parameters = get(parameters);
  catch
      oldWarn = warning('off','MATLAB:structOnObject');
      parameters = struct(parameters);
      warning(oldWarn);
  end
  
  % Init JIDE
  com.mathworks.mwswing.MJUtilities.initJIDE;

  % Prepare the list of properties
  oldWarn = warning('off','MATLAB:hg:JavaSetHGProperty');
  warning off MATLAB:hg:PossibleDeprecatedJavaSetHGProperty
  isEditable = true; %=nargin < 1;
  propsList = preparePropsList(parameters, isEditable);
  
  % Create a mapping propName => prop
  propsHash = java.util.Hashtable;
  propsArray = propsList.toArray();
  for propsIdx = 1 : length(propsArray)
      thisProp = propsArray(propsIdx);
      propName = getPropName(thisProp);
      propsHash.put(propName, thisProp);
  end
  warning(oldWarn);

  % Prepare a properties table that contains the list of properties
  model = javaObjectEDT(com.jidesoft.grid.PropertyTableModel(propsList));
  model.expandAll();

  % Prepare the properties table (grid)
  grid = javaObjectEDT(com.jidesoft.grid.PropertyTable(model));
  grid.setShowNonEditable(grid.SHOW_NONEDITABLE_BOTH_NAME_VALUE);
  %set(handle(grid.getSelectionModel,'CallbackProperties'), 'ValueChangedCallback', @propSelectedCallback);
  com.jidesoft.grid.TableUtils.autoResizeAllColumns(grid);
  %com.jidesoft.grid.TableUtils.autoResizeAllRows(grid);
  grid.setRowHeight(19);  % default=16; autoResizeAllRows=20 - we need something in between

  % Auto-end editing upon focus loss
  grid.putClientProperty('terminateEditOnFocusLost',true);

  % If no parent (or the root) was specified
  if nargin < 1 || isempty(hParent) || isequal(hParent,0)
      % Create a new figure window
      delete(findall(0, '-depth',1, 'Tag','fpropertiesGUI'));
      hFig = figure('NumberTitle','off', 'Name','Application properties', 'Units','pixel', 'Pos',[300,200,500,500], 'Menu','none', 'Toolbar','none', 'Tag','fpropertiesGUI', 'Visible','off');
      hParent = hFig;
      setappdata(0,'isParamsGUIApproved',false)

      % Add the bottom action buttons
      btOK     = uicontrol('String','OK',     'Units','pixel', 'Pos',[ 20,5,60,30], 'Tag','btOK',     'Callback',@btOK_Callback);
      btCancel = uicontrol('String','Cancel', 'Units','pixel', 'Pos',[100,5,60,30], 'Tag','btCancel', 'Callback',@(h,e)close(hFig)); %#ok<NASGU>

      % Add the rating link (demo mode only)
      if isDemo
          cursor = java.awt.Cursor(java.awt.Cursor.HAND_CURSOR);
          blogLabel = javax.swing.JLabel('<html><center>Additional interesting stuff at<br/><b><a href="">UndocumentedMatlab.com');
          set(handle(blogLabel,'CallbackProperties'), 'MouseClickedCallback', 'web(''http://UndocumentedMatlab.com'',''-browser'');');
          blogLabel.setCursor(cursor);
          javacomponent(blogLabel, [200,5,170,30], hFig);

          url = 'http://www.mathworks.com/matlabcentral/fileexchange/38864-propertiesgui';
          rateLabel = javax.swing.JLabel('<html><center><b><a href="">Feedback / rating for this utility');
          set(handle(rateLabel,'CallbackProperties'), 'MouseClickedCallback', ['web(''' url ''',''-browser'');']);
          rateLabel.setCursor(cursor);
          javacomponent(rateLabel, [380,5,110,30], hFig);
      end

      % Check the property values to determine whether the <OK> button should be enabled or not
      checkProps(propsList, btOK, true);
  
      % Set the figure icon & make visible
      jFrame = get(handle(hFig),'JavaFrame');
      icon = javax.swing.ImageIcon(fullfile(matlabroot, '/toolbox/matlab/icons/tool_legend.gif'));
      jFrame.setFigureIcon(icon);
      set(hFig, 'WindowStyle','modal', 'Visible','on');
      
      % Set the component's position
      %pos = [5,40,490,440];
      hFigPos = getpixelposition(hFig);
      pos = [5,40,hFigPos(3)-10,hFigPos(4)-50];

      wasFigCreated = true;
  else
      % Set the component's position
      drawnow nocallbacks;
      pos = getpixelposition(hParent);
      pos(1:2) = 5;
      pos = pos - [0,0,10,10];
      hFig = ancestor(hParent,'figure');
      wasFigCreated = false;

      % Clear the parent container
      if isequal(hFig,hParent)
          clf(hFig);
      else
          delete(allchild(hParent));
      end
  end

  %drawnow; pause(0.05);
  pane = javaObjectEDT(com.jidesoft.grid.PropertyPane(grid));
  customizePropertyPane(pane);
  warnst = warning('off','MATLAB:ui:javacomponent:FunctionToBeRemoved');
  [jPropsPane, hPropsPane_] = javacomponent(pane, pos, hParent);
  warning(warnst);
    
  % A callback for touching the mouse
  hgrid = handle(grid, 'CallbackProperties');
  set(hgrid, 'MousePressedCallback', {@MousePressedCallback, hFig});

  setappdata(hFig, 'jPropsPane',jPropsPane);
  setappdata(hFig, 'propsList',propsList);
  setappdata(hFig, 'propsHash',propsHash);
  setappdata(hFig, 'mirror',parameters);
  setappdata(hFig, 'hgrid',hgrid);
  set(hPropsPane_,'tag','hpropertiesGUI');

  set(hPropsPane_, 'Units','norm');

  % Align the background colors
  bgcolor = pane.getBackground.getComponents([]);
  try set(hParent, 'Color', bgcolor(1:3)); catch, end  % this fails in uitabs - never mind (works ok in stand-alone figures)
  try pane.setBorderColor(pane.getBackground); catch, end  % error reported by Andrew Ness
  
  % If a new figure was created, make it modal and wait for user to close it
  if wasFigCreated
      uiwait(hFig);
      if getappdata(0,'isParamsGUIApproved')
          parameters = test_data; %=getappdata(hFig, 'mirror');
      end
  end
  
  if nargout, hPropsPane = hPropsPane_; end  % prevent unintentional printouts to the command window
end  % propertiesGUI

% Mouse-click callback function
function MousePressedCallback(grid, eventdata, hFig)
    % Get the clicked location
    %grid = eventdata.getSource;
    %columnModel = grid.getColumnModel;
    %leftColumn = columnModel.getColumn(0);
    clickX = eventdata.getX;
    clickY = eventdata.getY;
    %rowIdx = grid.getSelectedRow + 1;

    if clickX <= 20  %leftColumn.getWidth % clicked the side-bar
        return
    %elseif grid.getSelectedColumn==0 % didn't press on the value (second column)
    %    return
    end

    % bail-out if right-click
    if eventdata.isMetaDown
        showGridContextMenu(hFig, grid, clickX, clickY);
    else
        % bail-out if the grid is disabled
        if ~grid.isEnabled,  return;  end

        %data = getappdata(hFig, 'mirror');
        selectedProp = grid.getSelectedProperty; % which property (java object) was selected
        if ~isempty(selectedProp)
            if ismember('arrayData',fieldnames(get(selectedProp)))
                % Get the current data and update it
                actualData = get(selectedProp,'ArrayData');
                updateDataInPopupTable(selectedProp.getName, actualData, hFig, selectedProp);
            end
        end
    end
end %Mouse pressed

% Update data in a popup table
function updateDataInPopupTable(titleStr, data, hGridFig, selectedProp)
    figTitleStr = [char(titleStr) ' data'];
    hFig = findall(0, '-depth',1, 'Name',figTitleStr);
    if isempty(hFig)
        hFig = figure('NumberTitle','off', 'Name',figTitleStr, 'Menubar','none', 'Toolbar','none');
    else
        figure(hFig);  % bring into focus
    end
    try
        mtable = createTable(hFig, [], data);
        set(mtable,'DataChangedCallback',{@tableDataUpdatedCallback,hGridFig,selectedProp});
        %uiwait(hFig)  % modality
    catch
        delete(hFig);
        errMsg = {'Editing this data requires Yair Altman''s Java-based data table (createTable) utility from the Matlab File Exchange.', ...
                  ' ', 'If you have already downloaded and unzipped createTable, then please ensure that it is on the Matlab path.'};
        uiwait(msgbox(errMsg,'Error','warn'));
        web('http://www.mathworks.com/matlabcentral/fileexchange/14225-java-based-data-table');
    end
end  % updateDataInPopupTable

% User updated the data in the popup table
function tableDataUpdatedCallback(mtable,eventData,hFig,selectedProp) %#ok<INUSL>
    % Get the latest data
    updatedData = cell(mtable.Data);
    try
        if ~iscellstr(updatedData)
            updatedData = cell2mat(updatedData);
        end
    catch
        % probably hetrogeneous data
    end

    propName = getRecursivePropName(selectedProp); % get the property name
    set(selectedProp,'ArrayData',updatedData); % update the appdata of the
    % specific property containing the actual information of the array

    %% Update the displayed value in the properties GUI
    dataClass = class(updatedData);
    value = regexprep(sprintf('%dx',size(updatedData)),{'^(.)','x$'},{'<$1',['> ' dataClass ' array']});
    % set(selectProp,'value',value);
    selectedProp.setValue(value); % update the table

    % Update the display
    propsList = getappdata(hFig, 'propsList');
    checkProps(propsList, hFig);

    % Refresh the GUI
    propsPane = getappdata(hFig, 'jPropsPane');
    try propsPane.repaint; catch; end    

    % Update the local mirror
    data = getappdata(hFig, 'mirror');
    eval(['data.' propName ' = updatedData;']);
    setappdata(hFig, 'mirror',data);
end  % tableDataUpdatedCallback

% Determine whether a specified object should be considered as having fields/properties
% Note: HG handles must be processed seperately for the main logic to work
function [hasProps,isHG] = hasProperties(object)
    % A bunch of tests, some of which may croak depending on the Matlab release, platform
    try isHG  = ishghandle(object); catch, isHG  = ishandle(object);  end
    try isst  = isstruct(object);   catch, isst  = false; end
    try isjav = isjava(object);     catch, isjav = false; end
    try isobj = isobject(object);   catch, isobj = false; end
    try isco  = iscom(object);      catch, isco  = false; end
    hasProps = ~isempty(object) && (isst || isjav || isobj || isco);
end

% Customize the property-pane's appearance
function customizePropertyPane(pane)
  pane.setShowDescription(false);  % YMA: we don't currently have textual descriptions of the parameters, so no use showing an empty box that just takes up GUI space...
  pane.setShowToolBar(false);
  pane.setOrder(2);  % uncategorized, unsorted - see http://undocumentedmatlab.com/blog/advanced-jide-property-grids/#comment-42057
end

% Prepare a list of some parameters for demo mode
function parameters = demoParameters
    parameters.floating_point_property = pi;
    parameters.signed_integer_property = int16(12);
    parameters.unsigned_integer_property = uint16(12);
    parameters.flag_property = true;
    parameters.file_property = mfilename('fullpath');
    parameters.folder_property = pwd;
    parameters.text_property = 'Sample text';
    parameters.fixed_choice_property = {'Yes','No','Maybe', 'No'};
    parameters.editable_choice_property = {'Yes','No','Maybe','', {3}};  % editable if the last cell element is ''
    parameters.date_property = java.util.Date;  % today's date
    parameters.another_date_property = now-365;  % last year
    parameters.time_property = datestr(now,'HH:MM:SS');
    parameters.password_property = '*****';
    parameters.IP_address_property = '10.20.30.40';
    parameters.my_category.width = 4;
    parameters.my_category.height = 3;
    parameters.my_category.and_a_subcategory.is_OK = true;
    parameters.numeric_array_property = [11,12,13,14];
    parameters.numeric_matrix = magic(5);
    parameters.logical_matrix = true(2,5);
    parameters.mixed_data_matrix = {true,'abc',pi,uint8(123); false,'def',-pi,uint8(64)};
    parameters.cell_array_property  = {1,magic(3),'text',-4};
    parameters.color_property = [0.4,0.5,0.6];
    parameters.another_color_property = java.awt.Color.red;
    parameters.font_property = java.awt.Font('Arial', java.awt.Font.BOLD, 12);
    if ~isdeployed()
      %#exclude matlab.desktop.editor.getActive
      try parameters.class_object_property = matlab.desktop.editor.getActive; catch, end
    end
end  % demoParameters

% Prepare a list of properties
function propsList = preparePropsList(parameters, isEditable)
  propsList = java.util.ArrayList();

  % Convert a class object into a struct
  if isobject(parameters)
      parameters = struct(parameters);
  end

  % Check for an array of inputs (currently unsupported)
  %if numel(parameters) > 1,  error('YMA:propertiesGUI:ArrayParameters','Non-scalar inputs are currently unsupported');  end

  % Prepare a dynamic list of properties, based on the struct fields
  if isstruct(parameters) && ~isempty(parameters)
      %allParameters = parameters(:);  % convert ND array => 3D array
      allParameters = reshape(parameters, size(parameters,1),size(parameters,2),[]);
      numParameters = numel(allParameters);
      if numParameters > 1
          for zIdx = 1 : size(allParameters,3)
              for colIdx = 1 : size(allParameters,2)
                  for rowIdx = 1 : size(allParameters,1)
                      parameters = allParameters(rowIdx,colIdx,zIdx);
                      field_name = '';
                      field_label = sprintf('(%d,%d,%d)',rowIdx,colIdx,zIdx);
                      field_label = regexprep(field_label,',1\)',')');  % remove 3D if unnecesary
                      newProp = newProperty(parameters, field_name, field_label, isEditable, '', '', @propUpdatedCallback);
                      propsList.add(newProp);
                  end
              end
          end
      else
          % Dynamically (generically) inspect all the fields and assign corresponding props
          field_names = fieldnames(parameters);
          for field_idx = 1 : length(field_names)
              arrayData = [];
              field_name = field_names{field_idx};
              value = parameters.(field_name);
              field_label = getFieldLabel(field_name);
              %if numParameters > 1,  field_label = [field_label '(' num2str(parametersIdx) ')'];  end
              field_description = '';  % TODO
              type = 'string';
              if isempty(value)
                  type = 'string';  % not really needed, but for consistency
              elseif isa(value,'java.awt.Color')
                  type = 'color';
              elseif isa(value,'java.awt.Font')
                  type = 'font';
              elseif isnumeric(value)
                  try %if length(value)==3
                      colorComponents = num2cell(value);
                      if numel(colorComponents) ~= 3
                          error(' ');  % bail out if definitely not a color
                      end
                      try
                          value = java.awt.Color(colorComponents{:});  % value between 0-1
                      catch
                          colorComponents = num2cell(value/255);
                          value = java.awt.Color(colorComponents{:});  % value between 0-255
                      end
                      type = 'color';
                  catch %else
                      if numel(value)==1
                          %value = value(1);
                          if value > now-3650 && value < now+3650
                              type = 'date';
                              value = java.util.Date(datestr(value));
                          elseif isa(value,'uint') || isa(value,'uint8') || isa(value,'uint16') || isa(value,'uint32') || isa(value,'uint64')
                              type = 'unsigned';
                          elseif isinteger(value)
                              type = 'signed';
                          else
                              type = 'float';
                          end
                      else % a vector or a matrix
                          arrayData = value;
                          value = regexprep(sprintf('%dx',size(value)),{'^(.)','x$'},{'<$1','> numeric array'});
                          %{
                          value = num2str(value);
                          if size(value,1) > size(value,2)
                              value = value';
                          end
                          if size(squeeze(value),2) > 1
                              % Convert multi-row string into a single-row string
                              value = [value'; repmat(' ',1,size(value,1))];
                              value = value(:)';
                          end
                          value = strtrim(regexprep(value,' +',' '));
                          if length(value) > 50
                              value(51:end) = '';
                              value = [value '...']; %#ok<AGROW>
                          end
                          value = ['[ ' value ' ]']; %#ok<AGROW>
                          %}
                      end
                  end
              elseif islogical(value)
                  if numel(value)==1
                      % a single value
                      type = 'boolean';
                  else % an array of boolean values
                      arrayData = value;
                      value = regexprep(sprintf('%dx',size(value)),{'^(.)','x$'},{'<$1','> logical array'});
                  end
              elseif ischar(value)
                  if exist(value,'dir')
                      type = 'folder';
                      value = java.io.File(value);
                  elseif exist(value,'file')
                      type = 'file';
                      value = java.io.File(value);
                  elseif value(1)=='*'
                      type = 'password';
                  elseif sum(value=='.')==3
                      type = 'IPAddress';
                  else
                      type = 'string';
                      if length(value) > 50
                          value(51:end) = '';
                          value = [value '...']; %#ok<AGROW>
                      end
                  end
              elseif iscell(value) 
                  type = value;  % editable if the last cell element is ''
                  if size(value,1)==1 || size(value,2)==1
                      % vector - treat as a drop-down (combo-box/popup) of values
                      if ~iscellstr(value)
                          type = value;
                          for ii=1:length(value)
                              if isnumeric(value{ii})  % if item is numeric -> change to string for display.
                                  type{ii} = num2str(value{ii});
                              else
                                  type{ii} = value{ii};
                              end
                          end
                      end
                  else  % Matrix - use table popup
                      %value = ['{ ' strtrim(regexprep(evalc('disp(value)'),' +',' ')) ' }'];
                      arrayData = value;
                      value = regexprep(sprintf('%dx',size(value)),{'^(.)','x$'},{'<$1','> cell array'});
                  end                  
              elseif isa(value,'java.util.Date')
                  type = 'date';
              elseif isa(value,'java.io.File')
                  if value.isFile
                      type = 'file';
                  else  % value.isDirectory
                      type = 'folder';
                  end
              elseif isobject(value)
                  oldWarn = warning('off','MATLAB:structOnObject');
                  value = struct(value);
                  warning(oldWarn);
              elseif ~isstruct(value)
                  value = strtrim(regexprep(evalc('disp(value)'),' +',' '));
              end
              parameters.(field_name) = value;  % possibly updated above
              newProp = newProperty(parameters, field_name, field_label, isEditable, type, field_description, @propUpdatedCallback);
              propsList.add(newProp);

              % Save the array as a new property of the object
              if ~isempty(arrayData)
                  try
                      set(newProp,'arrayData',arrayData)
                  catch
                      %setappdata(hProp,'UserData',propName)
                      hp = schema.prop(handle(newProp),'arrayData','mxArray'); %#ok<NASGU>
                      set(handle(newProp),'arrayData',arrayData)
                  end
                  newProp.setEditable(false);
              end
          end
      end
  else
      % You can also use direct assignments, instead of the generic code above. For example:
      % (Possible property types: signed, unsigned, float, file, folder, text or string, color, IPAddress, password, date, boolean, cell-array of strings)
      propsList.add(newProperty(parameters, 'flag_prop_name',   'Flag value:',     isEditable, 'boolean',            'Turn this on if you want to make extra plots', @propUpdatedCallback));
      propsList.add(newProperty(parameters, 'float_prop_name',  'Boolean prop',    isEditable, 'float',              'description 123...',   @propUpdatedCallback));
      propsList.add(newProperty(parameters, 'string_prop_name', 'My text msg:',    isEditable, 'string',             'Yaba daba doo',        @propUpdatedCallback));
      propsList.add(newProperty(parameters, 'int_prop_name',    'Now an integer',  isEditable, 'unsigned',           '123 456...',           @propUpdatedCallback));
      propsList.add(newProperty(parameters, 'choice_prop_name', 'And a drop-down', isEditable, {'Yes','No','Maybe'}, 'no description here!', @propUpdatedCallback));
  end
end  % preparePropsList

% Get a normalized field label (see also checkFieldName() below)
function field_label = getFieldLabel(field_name)
    field_label = regexprep(field_name, '__(.*)', ' ($1)');
    field_label = strrep(field_label,'_',' ');
    field_label(1) = upper(field_label(1));
end
    
% Prepare a data property
function prop = newProperty(dataStruct, propName, label, isEditable, dataType, description, propUpdatedCallback)
    % Auto-generate the label from the property name, if the label was not specified
    if isempty(label)
        label = getFieldLabel(propName);
    end

    % Create a new property with the chosen label
    prop = javaObjectEDT(com.jidesoft.grid.DefaultProperty);  % UNDOCUMENTED internal MATLAB component
    prop.setName(label);
    prop.setExpanded(true);

    % Set the property to the current patient's data value
    try
        thisProp = dataStruct.(propName);
    catch
        thisProp = dataStruct;
    end
    origProp = thisProp;
    if isstruct(thisProp)  %hasProperties(thisProp)
        % Accept any object having data fields/properties
        try
            thisProp = get(thisProp);
        catch
            oldWarn = warning('off','MATLAB:structOnObject');
            thisProp = struct(thisProp);
            warning(oldWarn);
        end

        % Parse the children props and add them to this property
        %summary = regexprep(evalc('disp(thisProp)'),' +',' ');
        %prop.setValue(summary);  % TODO: display summary dynamically
        if numel(thisProp) < 2
            prop.setValue('');
        else
            sz = size(thisProp);
            szStr = regexprep(num2str(sz),' +','x');
            prop.setValue(['[' szStr ' struct array]']);
        end
        prop.setEditable(false);
        children = toArray(preparePropsList(thisProp, isEditable));
        for childIdx = 1 : length(children)
            prop.addChild(children(childIdx));
        end
    else
        prop.setValue(thisProp);
        prop.setEditable(isEditable);
    end

    % Set property editor, renderer and alignment
    if iscell(dataType)
        % treat this as drop-down values
        % Set the defaults
        firstIndex = 1;
        cbIsEditable = false;
        % Extract out the number of items in the user list
        nItems = length(dataType);
        % Check for any empty items
        emptyItem = find(cellfun('isempty', dataType) == 1);
        % If only 1 empty item found check editable rules
        if length(emptyItem) == 1
            % If at the end - then remove it and set editable flag
            if emptyItem == nItems
                cbIsEditable = true;
                dataType(end) = []; % remove from the drop-down list
            elseif emptyItem == nItems - 1
                cbIsEditable = true;
                dataType(end-1) = []; % remove from the drop-down list
            end
        end

        % Try to find the initial (default) drop-down index
        if ~isempty(dataType)
            if iscell(dataType{end})
                if isnumeric(dataType{end}{1})
                    firstIndex = dataType{end}{1};
                    dataType(end) = []; % remove the [] from drop-down list
                end
            else
                try
                    if ismember(dataType{end}, dataType(1:end-1))
                        firstIndex = find(strcmp(dataType(1:end-1),dataType{end}));
                        dataType(end) = [];
                    end
                catch
                    % ignore - possibly mixed data
                end
            end

            % Build the editor
            editor = com.jidesoft.grid.ListComboBoxCellEditor(dataType);
            try editor.getComboBox.setEditable(cbIsEditable); catch, end % #ok<NOCOM>
            %set(editor,'EditingStoppedCallback',{@propUpdatedCallback,tagName,propName});
            alignProp(prop, editor);
            try prop.setValue(origProp{firstIndex}); catch, end
        end
    else
        switch lower(dataType)
            case 'signed',    %alignProp(prop, com.jidesoft.grid.IntegerCellEditor,    'int32');
                model = javax.swing.SpinnerNumberModel(prop.getValue, -intmax, intmax, 1);
                editor = com.jidesoft.grid.SpinnerCellEditor(model);
                alignProp(prop, editor, 'int32');
            case 'unsigned',  %alignProp(prop, com.jidesoft.grid.IntegerCellEditor,    'uint32');
                val = max(0, min(prop.getValue, intmax));
                model = javax.swing.SpinnerNumberModel(val, 0, intmax, 1);
                editor = com.jidesoft.grid.SpinnerCellEditor(model);
                alignProp(prop, editor, 'uint32');
            case 'float',     %alignProp(prop, com.jidesoft.grid.CalculatorCellEditor, 'double');  % DoubleCellEditor
                alignProp(prop, com.jidesoft.grid.DoubleCellEditor, 'double');
            case 'boolean',   alignProp(prop, com.jidesoft.grid.BooleanCheckBoxCellEditor, 'logical');
            case 'folder',    alignProp(prop, com.jidesoft.grid.FolderCellEditor);
            case 'file',      alignProp(prop, com.jidesoft.grid.FileCellEditor);
            case 'ipaddress', alignProp(prop, com.jidesoft.grid.IPAddressCellEditor);
            case 'password',  alignProp(prop, com.jidesoft.grid.PasswordCellEditor);
            case 'color',     alignProp(prop, com.jidesoft.grid.ColorCellEditor);
            case 'font',      alignProp(prop, com.jidesoft.grid.FontCellEditor);
            case 'text',      alignProp(prop);
            case 'time',      alignProp(prop);  % maybe use com.jidesoft.grid.FormattedTextFieldCellEditor ?

            case 'date',      dateModel = com.jidesoft.combobox.DefaultDateModel;
                dateFormat = java.text.SimpleDateFormat('dd/MM/yyyy');
                dateModel.setDateFormat(dateFormat);
                editor = com.jidesoft.grid.DateCellEditor(dateModel, 1);
                alignProp(prop, editor, 'java.util.Date');
                try
                    prop.setValue(dateFormat.parse(prop.getValue));  % convert string => Date
                catch
                    % ignore
                end

            otherwise,        alignProp(prop);  % treat as a simple text field
        end
    end  % for all possible data types

    prop.setDescription(description);
    if ~isempty(description)
        renderer = com.jidesoft.grid.CellRendererManager.getRenderer(prop.getType, prop.getEditorContext);
        renderer.setToolTipText(description);
    end

    % Set the property's editability state
    if prop.isEditable
        % Set the property's label to be black
        prop.setDisplayName(['<html><font color="black">' label]);

        % Add callbacks for property-change events
        hprop = handle(prop, 'CallbackProperties');
        set(hprop,'PropertyChangeCallback',{propUpdatedCallback,propName});
    else
        % Set the property's label to be gray
        prop.setDisplayName(['<html><font color="gray">' label]);
    end

    setPropName(prop,propName);
end  % newProperty

% Set property name in the Java property reference
function setPropName(hProp,propName)
    try
        set(hProp,'UserData',propName)
    catch
        %setappdata(hProp,'UserData',propName)
        hp = schema.prop(handle(hProp),'UserData','mxArray'); %#ok<NASGU>
        set(handle(hProp),'UserData',propName)
    end
end  % setPropName

% Get property name from the Java property reference
function propName = getPropName(hProp)
    try
        propName = get(hProp,'UserData');
    catch
        %propName = char(getappdata(hProp,'UserData'));
        propName = get(handle(hProp),'UserData');
    end
end  % getPropName

% Get recursive property name
function propName = getRecursivePropName(prop, propBaseName)
    try
        oldWarn = warning('off','MATLAB:hg:JavaSetHGProperty');
        try prop = java(prop); catch, end
        if nargin < 2
            propName = getPropName(prop);
        else
            propName = propBaseName;
        end
        while isa(prop,'com.jidesoft.grid.Property')
            prop = get(prop,'Parent');
            newName = getPropName(prop);
            if isempty(newName)
                %% check to see if it's a (1,1)
                displayName = char(prop.getName);
                [flag, index] = CheckStringForBrackets(displayName);
                if flag
                    propName = sprintf('(%i).%s',index,propName); 
                else                
                    break; 
                end
            else
                propName = [newName '.' propName]; %#ok<AGROW>
            end
        end
    catch
        % Reached the top of the property's heirarchy - bail out
        warning(oldWarn);
    end
end  % getRecursivePropName

% Align a text property to right/left
function alignProp(prop, editor, propTypeStr, direction)
    persistent propTypeCache
    if isempty(propTypeCache),  propTypeCache = java.util.Hashtable;  end

    if nargin < 2 || isempty(editor),      editor = com.jidesoft.grid.StringCellEditor;  end  %(javaclass('char',1));
    if nargin < 3 || isempty(propTypeStr), propTypeStr = 'cellstr';  end  % => javaclass('char',1)
    if nargin < 4 || isempty(direction),   direction = javax.swing.SwingConstants.RIGHT;  end

    % Set this property's data type
    propType = propTypeCache.get(propTypeStr);
    if isempty(propType)
        propType = javaclass(propTypeStr);
        propTypeCache.put(propTypeStr,propType);
    end
    prop.setType(propType);

    % Prepare a specific context object for this property
    if strcmpi(propTypeStr,'logical')
        %TODO - FIXME
        context = editor.CONTEXT;
        prop.setEditorContext(context);
        %renderer = CheckBoxRenderer;
        %renderer.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        %com.jidesoft.grid.CellRendererManager.registerRenderer(propType, renderer, context);
    else
        context = com.jidesoft.grid.EditorContext(prop.getName);
        prop.setEditorContext(context);

        % Register a unique cell renderer so that each property can be modified seperately
        %renderer = com.jidesoft.grid.CellRendererManager.getRenderer(propType, prop.getEditorContext);
        renderer = com.jidesoft.grid.ContextSensitiveCellRenderer;
        com.jidesoft.grid.CellRendererManager.registerRenderer(propType, renderer, context);
        renderer.setBackground(java.awt.Color.white);
        renderer.setHorizontalAlignment(direction);
        %renderer.setHorizontalTextPosition(direction);
    end

    % Update the property's cell editor
    try editor.setHorizontalAlignment(direction); catch, end
    try editor.getTextField.setHorizontalAlignment(direction); catch, end
    try editor.getComboBox.setHorizontalAlignment(direction); catch, end

    % Set limits on unsigned int values
    try
        if strcmpi(propTypeStr,'uint32')
            %pause(0.01);
            editor.setMinInclusive(java.lang.Integer(0));
            editor.setMinExclusive(java.lang.Integer(-1));
            editor.setMaxExclusive(java.lang.Integer(intmax));
            editor.setMaxInclusive(java.lang.Integer(intmax));
        end
    catch
        % ignore
    end
    com.jidesoft.grid.CellEditorManager.registerEditor(propType, editor, context);
end  % alignProp

% Property updated callback function
function propUpdatedCallback(prop, eventData, propName, fileData)
    try if strcmpi(char(eventData.getPropertyName),'parent'),  return;  end;  catch, end

    % Retrieve the containing figure handle
    %hFig = findall(0, '-depth',1, 'Tag','fpropertiesGUI');
    hFig = get(0,'CurrentFigure'); %gcf;
    if isempty(hFig)
        hPropsPane = findall(0,'Tag','hpropertiesGUI');
        if isempty(hPropsPane),  return;  end
        hFig = ancestor(hPropsPane,'figure'); %=get(hPropsPane,'Parent');
    end
    if isempty(hFig),  return;  end

    % Get the props data from the figure's ApplicationData
    propsList = getappdata(hFig, 'propsList');
    propsPane = getappdata(hFig, 'jPropsPane');
    data = getappdata(hFig, 'mirror');

    % Bail out if arriving from tableDataUpdatedCallback
    try
        s = dbstack;
        if strcmpi(s(2).name, 'tableDataUpdatedCallback')
            return;
        end
    catch
        % ignore
    end

    % Get the updated property value
    propValue = get(prop,'Value');
    if isjava(propValue)
        if isa(propValue,'java.util.Date')
            sdf = java.text.SimpleDateFormat('MM-dd-yyyy');
            propValue = datenum(sdf.format(propValue).char);  %#ok<NASGU>
        elseif isa(propValue,'java.awt.Color')
            propValue = propValue.getColorComponents([])';  %#ok<NASGU>
        else
            propValue = char(propValue);  %#ok<NASGU>
        end
    end

    % Get the actual recursive propName
    propName = getRecursivePropName(prop, propName);

    % Find if the original item was a cell array and the mirror accordingly
    items = strread(propName,'%s','delimiter','.');
    if ~isempty(data)
        cpy = data;
        for idx = 1 : length(items)
            % This is for dealing with structs with multiple levels...
            [flag, index] = CheckStringForBrackets(items{idx});
            if flag
                cpy = cpy(index);
            else
                if isfield(cpy,items{idx})
                    cpy = cpy.(items{idx});
                else
                    return
                end
            end
        end
        if nargin == 4
            if iscell(cpy) && iscell(fileData) %%&& length(fileData)==1 % if mirror and filedata are cells then update the data -> otherwise overright.
                propValue=UpdateCellArray(cpy,fileData);
            end
        else
            if iscell(cpy)
                propValue = UpdateCellArray(cpy, propValue);
            end
        end
    end

    % Check for loading from file and long string which has been truncated
    if nargin == 4
        propValue = checkCharFieldForAbreviation(propValue,fileData);
        if ~isempty(propValue) && strcmp(propValue(1),'[') && ~isempty(strfind(propValue,' struct array]'))
            propValue = fileData;
        end
        if isempty(propValue) % a struct
            propValue = fileData;
        end
    end
    
    % For items with .(N) in the struct -> remove from path for eval
    propName = regexprep(propName,'\.(','(');
    
    % Update the mirror with the updated field value
    %data.(propName) = propValue;  % croaks on multiple sub-fields
    eval(['data.' propName ' = propValue;']);

    % Update the local mirror
    setappdata(hFig, 'mirror',data);

    % Update the display
    checkProps(propsList, hFig);
    try propsPane.repaint; catch; end
end  % propUpdatedCallback

function selectedValue = UpdateCellArray(originalData,selectedValue)
    if length(originalData)==length(selectedValue) || ~iscell(selectedValue)
        index=find(strcmp(originalData,selectedValue)==1);
        if iscell(originalData{end})
            originalData{end}={index};
        else
            if index~=1 % If it's not first index then we can save it
                originalData{end+1} = {index};
            end
        end
        selectedValue=originalData;
    else
        selectedValue=originalData;
    end
end  % UpdateCellArray

% <OK> button callback function
function btOK_Callback(btOK, eventData) %#ok<INUSD>
  global test_data

  % Store the current data-info struct mirror in the global struct
  hFig = ancestor(btOK, 'figure');
  test_data = getappdata(hFig, 'mirror');
  setappdata(0,'isParamsGUIApproved',true);

  % Close the window
  try
      close(hFig);
  catch
      delete(hFig);  % force-close
  end
end  % btOK_Callback

% Check whether all mandatory fields have been filled, update background color accordingly
function checkProps(propsList, hContainer, isInit)
    if nargin < 3,  isInit = false;  end
    okEnabled = 'on';
    try propsArray = propsList.toArray(); catch, return; end
    for propsIdx = 1 : length(propsArray)
        isOk = checkProp(propsArray(propsIdx));
        if ~isOk || isInit,  okEnabled = 'off';  end
    end
    
    % Update the <OK> button's editability state accordingly
    btOK = findall(hContainer, 'Tag','btOK');
    set(btOK, 'Enable',okEnabled);
    set(findall(get(hContainer,'parent'), 'tag','btApply'),  'Enable',okEnabled);
    set(findall(get(hContainer,'parent'), 'tag','btRevert'), 'Enable',okEnabled);
    try; drawnow nocallbacks; pause(0.01); end

    % Update the figure title to indicate dirty-ness (if the figure is visible)
    hFig = ancestor(hContainer,'figure');
    if strcmpi(get(hFig,'Visible'),'on')
        sTitle = regexprep(get(hFig,'Name'), '\*$','');
        set(hFig,'Name',[sTitle '*']);
    end
end  % checkProps

function isOk = checkProp(prop)
  isOk = true;
  oldWarn = warning('off','MATLAB:hg:JavaSetHGProperty');
  warning off MATLAB:hg:PossibleDeprecatedJavaSetHGProperty
  propName = getPropName(prop);
  renderer = com.jidesoft.grid.CellRendererManager.getRenderer(get(prop,'Type'), get(prop,'EditorContext'));
  warning(oldWarn);
  mandatoryFields = {};  % TODO - add the mandatory field-names here
  if any(strcmpi(propName, mandatoryFields)) && isempty(get(prop,'Value'))
      propColor = java.awt.Color.yellow;
      isOk = false;
  elseif ~prop.isEditable
      %propColor = java.awt.Color.gray;
      %propColor = renderer.getBackground();
      propColor = java.awt.Color.white;
  else
      propColor = java.awt.Color.white;
  end
  renderer.setBackground(propColor);
end  % checkProp

% Return java.lang.Class instance corresponding to the Matlab type
function jclass = javaclass(mtype, ndims)
    % Input arguments:
    % mtype:
    %    the MatLab name of the type for which to return the java.lang.Class
    %    instance
    % ndims:
    %    the number of dimensions of the MatLab data type
    %
    % See also: class
    
    % Copyright 2009-2010 Levente Hunyadi
    % Downloaded from: http://www.UndocumentedMatlab.com/files/javaclass.m
    
    validateattributes(mtype, {'char'}, {'nonempty','row'});
    if nargin < 2
        ndims = 0;
    else
        validateattributes(ndims, {'numeric'}, {'nonnegative','integer','scalar'});
    end
    
    if ndims == 1 && strcmp(mtype, 'char');  % a character vector converts into a string
        jclassname = 'java.lang.String';
    elseif ndims > 0
        jclassname = javaarrayclass(mtype, ndims);
    else
        % The static property .class applied to a Java type returns a string in
        % MatLab rather than an instance of java.lang.Class. For this reason,
        % use a string and java.lang.Class.forName to instantiate a
        % java.lang.Class object; the syntax java.lang.Boolean.class will not do so
        switch mtype
            case 'logical'  % logical vaule (true or false)
                jclassname = 'java.lang.Boolean';
            case 'char'  % a singe character
                jclassname = 'java.lang.Character';
            case {'int8','uint8'}  % 8-bit signed and unsigned integer
                jclassname = 'java.lang.Byte';
            case {'int16','uint16'}  % 16-bit signed and unsigned integer
                jclassname = 'java.lang.Short';
            case {'int32','uint32'}  % 32-bit signed and unsigned integer
                jclassname = 'java.lang.Integer';
            case {'int64','uint64'}  % 64-bit signed and unsigned integer
                jclassname = 'java.lang.Long';
            case 'single'  % single-precision floating-point number
                jclassname = 'java.lang.Float';
            case 'double'  % double-precision floating-point number
                jclassname = 'java.lang.Double';
            case 'cellstr'  % a single cell or a character array
                jclassname = 'java.lang.String';
            otherwise
                jclassname = mtype;
                %error('java:javaclass:InvalidArgumentValue', ...
                %    'MatLab type "%s" is not recognized or supported in Java.', mtype);
        end
    end
    % Note: When querying a java.lang.Class object by name with the method
    % jclass = java.lang.Class.forName(jclassname);
    % MatLab generates an error. For the Class.forName method to work, MatLab
    % requires class loader to be specified explicitly.
    jclass = java.lang.Class.forName(jclassname, true, java.lang.Thread.currentThread().getContextClassLoader());
end  % javaclass
    
% Return the type qualifier for a multidimensional Java array
function jclassname = javaarrayclass(mtype, ndims)
    switch mtype
        case 'logical'  % logical array of true and false values
            jclassid = 'Z';
        case 'char'  % character array
            jclassid = 'C';
        case {'int8','uint8'}  % 8-bit signed and unsigned integer array
            jclassid = 'B';
        case {'int16','uint16'}  % 16-bit signed and unsigned integer array
            jclassid = 'S';
        case {'int32','uint32'}  % 32-bit signed and unsigned integer array
            jclassid = 'I';
        case {'int64','uint64'}  % 64-bit signed and unsigned integer array
            jclassid = 'J';
        case 'single'  % single-precision floating-point number array
            jclassid = 'F';
        case 'double'  % double-precision floating-point number array
            jclassid = 'D';
        case 'cellstr'  % cell array of strings
            jclassid = 'Ljava.lang.String;';
        otherwise
            jclassid = ['L' mtype ';'];
            %error('java:javaclass:InvalidArgumentValue', ...
            %    'MatLab type "%s" is not recognized or supported in Java.', mtype);
    end
    jclassname = [repmat('[',1,ndims), jclassid];
end  % javaarrayclass

% Set up the uitree context (right-click) menu
function showGridContextMenu(hFig, grid, clickX, clickY)
    % Prepare the context menu (note the use of HTML labels)
    import javax.swing.*
    row = grid.rowAtPoint(java.awt.Point(clickX, clickY));
    selectedProp = grid.getPropertyTableModel.getPropertyAt(row);
    if ~isempty(selectedProp)
        branchName = char(selectedProp.getName);
    else
        branchName = 'branch';
    end
    menuItem1 = JMenuItem(['Save ' branchName '...']);
    menuItem2 = JMenuItem(['Load ' branchName '...']);

    % Set the menu items' callbacks
    set(handle(menuItem1,'CallbackProperties'),'ActionPerformedCallback',@(obj,event)fileIO_Callback(grid,selectedProp,'save',hFig));
    set(handle(menuItem2,'CallbackProperties'),'ActionPerformedCallback',@(obj,event)fileIO_Callback(grid,selectedProp,'load',hFig));

    % Add all menu items to the context menu (with internal separator)
    jmenu = JPopupMenu;
    jmenu.add(menuItem1);
    jmenu.addSeparator;
    jmenu.add(menuItem2);

    % Display the context-menu
    jmenu.show(grid, clickX, clickY);
    jmenu.repaint;
end  % setGridContextMenu

function fileIO_Callback(grid, selectedProp, mode, hFig, filename)
    persistent lastdir
    mirrorData = getappdata(hFig, 'mirror');
    if nargin == 4 % interactive
        filename = [];
    end

    % Initialize the persistent variable with the current Dir if not populated.
    if isempty(lastdir); lastdir = pwd; end
    switch mode
        case 'save'
            filename = saveBranch_Callback(grid, selectedProp, lastdir, mirrorData, hFig, filename);
        case 'load'
            filename = loadBranchCallback(grid, selectedProp, lastdir, mirrorData, filename);
        case {'update', 'select'} % hidden calling method
            runtimeUpdateBranch(grid,filename,mirrorData,selectedProp);
            return
        otherwise
            error('propertiesGUI:fileIOCallback:invalidMethod', 'invalid calling method to propertiesGUI');            
%             setappdata(hFig, 'mirror',mirrorData);
    end

    % Check that the save/load wsa processed
    if ischar(filename)
        filePath = fileparts(filename);
        if ~isempty(filePath)
            lastdir = filePath;
        end
    end
end  % fileIO_Callback

function filename = loadBranchCallback(grid, selectedProp, lastdir, mirrorData, filename)
    if isempty(filename)
        [filename, pathname] = uigetfile({'*.branch','Branch files (*.branch)'}, 'Load a file', lastdir);
        if filename == 0
            return
        end
        filename = fullfile(pathname, filename);
    else
        selectedProp = findUserProvidedProp(grid, selectedProp);
    end
    propName = char(selectedProp.getName);
    propName = checkFieldName(propName);
    data = load(filename, '-mat');
    fnames = fieldnames(data);
    index = strcmpi(fnames,propName);

    % If a match was found then it's okay to proceed
    if any(index)
        % Remove any children
        selectedProp.removeAllChildren();

        % Make a new list
        newList = preparePropsList(data, true);

        % Conver the list to an array
        newArray = newList.toArray();
        updatedProp = newArray(1);
        
        isStruct = false;
        propValue = selectedProp.getValue;
        if ~isempty(propValue) && strcmp(propValue(1),'[') && ~isempty(strfind(propValue,' struct array]'))
            isStruct = true;
        end
        
        % If individual value update it.  TODO: Bug when it is a cell array....
            
        if isStruct == false && ~isempty(propValue)
            selectedProp.setValue (updatedProp.getValue)
            propName = checkFieldName(char(updatedProp.getName));
            if iscell(data.(fnames{index})) && ischar(data.(fnames{index}){end}) && ismember(data.(fnames{index})(end),data.(fnames{index})(1:end-1))
                data.(fnames{index})(end) = [];
            end
            propUpdatedCallback(selectedProp, [], propName, data.(fnames{index}));
        else
            % Add children to the original property.
            for ii=1:updatedProp.getChildrenCount
                childProp = updatedProp.getChildAt(ii-1);
                propName = checkFieldName(char(childProp.getName));
                [flag, sIndex] = CheckStringForBrackets(propName);
                if flag
%                     propUpdatedCallback(childProp, [], propName, data.(fnames{index}).(propName));
                else
                    propUpdatedCallback(childProp, [], propName, data.(fnames{index}).(propName));
                end                
                selectedProp.addChild(childProp);
            end
        end
    else
        errMsg = 'The selected branch does not match the data in the data file';
        %error('propertieGUI:load:branchName', errMsg);
        errordlg(errMsg, 'Load error');
    end
end
% runtime update item in branch (undocumented - for easier testing)
function runtimeUpdateBranch(grid, selectedProp, mirrorData, newData)
    userStr = strread(selectedProp,'%s','delimiter','.');
    if length(userStr)~= 1
        mirrorData = findMirrorDataLevel(mirrorData, userStr);
    end    
    selectedProp = findUserProvidedProp(grid, selectedProp);
    if ~isempty(selectedProp.getValue)
        propName = checkFieldName(char(selectedProp.getName));
        if iscell(newData) && length(newData)==1 && isnumeric(newData{1}) % user specifying index to select.
            propData = mirrorData.(propName);
            if iscell(mirrorData.(propName))
                userSelection = propData{newData{1}};
            else
                userSelection = newData;
            end
            if any(ismember(propData,userSelection))
                selectedProp.setValue (userSelection);
                propUpdatedCallback(selectedProp, [], propName, propData);
            end
        end
    end
end  % runtimeUpdateBranch

% Save callback and subfunctions
function filename = saveBranch_Callback(grid, selectedProp, lastdir, mirrorData, hFig, filename)
    % Interactive use
    runtimeCall = isempty(filename);
    if runtimeCall
        [filename, pathname] = uiputfile ({'*.branch','Branch files (*.branch)'}, 'Save as', lastdir);
        if filename == 0
            return
        end
        filename = fullfile(pathname, filename);
    else % from commandline
        if isempty(selectedProp) % user wants to save everything.
            selectedProp = grid;            
        else
            userStr = strread(selectedProp,'%s','delimiter','.');
            if length(userStr)~= 1
                mirrorData = findMirrorDataLevel(mirrorData, userStr);
            end
            selectedProp = findUserProvidedProp(grid, selectedProp);
        end
    end
    
    if ~isempty(selectedProp.getName)
        fieldname = checkFieldName(selectedProp.getName);
        data.(fieldname) = selfBuildStruct(selectedProp);
        fieldname = {fieldname};
    else
        [rootProps, data] = buildFullStruct(hFig);  % (grid,mirrorData)
        fieldname = fieldnames(data);
        selectedProp = rootProps{1};
    end
    
    % option to save combo boxes as well...
    if nargin >= 4
        for fieldIdx = 1 : length(fieldname)
            if fieldIdx>1 % This only happens when loading to replace the full data
                 selectedProp = rootProps{fieldIdx};
            end
            dataNames = fieldnames(mirrorData);
            match = strcmpi(dataNames,fieldname{fieldIdx});

            % This sub function will add all the extra items
            if any(match)
                % This looks in the mirrorData to update the output with cell array items.
                data.(fieldname{fieldIdx}) = addOptionsToOutput(data.(fieldname{fieldIdx}), mirrorData.(dataNames{match}), selectedProp);
                % Update the original var names (case sensitive)
                data = updateOriginalVarNames(data, mirrorData); %data is used in the save command.
            else
                propName = getRecursivePropName(selectedProp, fieldname{fieldIdx});
                items = strread(propName,'%s','delimiter','.');
                for idx = 1 : length(items)-1
                    if strcmp(items{idx}(1),'(') && strcmp(items{idx}(end),')')
                        index = str2double(items{idx}(2:end-1));
                        mirrorData = mirrorData(index);
                    else
                        mirrorData = mirrorData.(items{idx});
                    end
                end
                data.(fieldname{fieldIdx}) = addOptionsToOutput(data.(fieldname{fieldIdx}), mirrorData.(items{end}), selectedProp);
                % Update the original var names (case sensitive)
                data = updateOriginalVarNames(data, mirrorData); %data is used in the save command.
            end
        end
    end
    
    % Save the data to file
    save(filename, '-struct', 'data')
end 

% Descent through the mirror data to find the matching variable for the user requested data
function mirrorData = findMirrorDataLevel(mirrorData, userStr)
    if length(userStr)==1
        return
    else
        [flag, index] = CheckStringForBrackets(userStr{1});
        if flag
            mirrorData = findMirrorDataLevel(mirrorData(index), userStr(2:end));
        else
            mirrorData = mirrorData.(userStr{1});
            mirrorData = findMirrorDataLevel(mirrorData, userStr(2:end));
        end
    end
end  % findMirrorDataLevel

% Search for the user specified property to load or to save
function selectedProp = findUserProvidedProp(grid, selectedProp)
        index = 0;
        % Loop through the properties to find the matching branch
        strItems = strread(selectedProp, '%s', 'delimiter', '.');
        while true
            incProp = grid.getPropertyTableModel.getPropertyAt(index);
            if isempty(incProp)
                error('propertiesGUI:InvalidBranch', 'User provied property name which was invalid')
            end
            % Search the full user defined string for the item to be saved.
            selectedProp = searchForPropName(incProp, strItems);
            if ~isempty(selectedProp); break; end
            index = index + 1;
        end
end  % findUserProvidedProp

% Sub function for searching down through the user provided string when A.B.C provided.
function selectedProp = searchForPropName(parentNode, userString)
    selectedProp = [];
    nodeName = char(parentNode.getName);
%     if strcmp(nodeName(1),'(') && strcmp(nodeName(end),')')
    if strcmpi(userString{1},checkFieldName(nodeName)) % ? shoudl this be case sensitive?
        if length(userString) == 1
            selectedProp = parentNode;
        else
            for jj=1:parentNode.getChildrenCount
                selectedProp = searchForPropName(parentNode.getChildAt(jj-1), userString(2:end));
                if ~isempty(selectedProp)
                    break
                end
            end
        end
    end
end  % searchForPropName

% Build full struct
function [rootProps, output] = buildFullStruct(hFig)  % (grid,mirrorData)
    %{
    % This fails if some of the top-level props are expanded (open)
    index = 0;
    rootProps = {};
    while true
        incProp = grid.getPropertyTableModel.getPropertyAt(index);
        if isempty(incProp); break; end
        % Search the full user defined string for the item to be saved.
        propName = checkFieldName(incProp.getName);
        if isfield(mirrorData,propName)
          output.(propName) = selfBuildStruct(incProp);
          rootProps{end+1} = incProp;
        end
        index = index + 1;
    end
    %}
    propsList = getappdata(hFig, 'propsList');
    rootProps = cell(propsList.toArray)';
    for propIdx = 1 : numel(rootProps)
        thisProp = rootProps{propIdx};
        propName = checkFieldName(thisProp.getName);
        output.(propName) = selfBuildStruct(thisProp);
    end
end  % buildFullStruct

% Build the structure for saving from the selected Prop
function output = selfBuildStruct(selectedProp)
    % Self calling loop to build the output structure.
    propValue = selectedProp.getValue;
    % If property empty then the selectedProp is a struct.
    
    isStruct = isempty(propValue);
    nStructs = 1;
    
    % Check if it's an array of structs
    M = 1;
    if ~isempty(propValue) && strcmp(propValue(1),'[') && ~isempty(strfind(propValue,' struct array]'))
        isStruct = true;
        nStructs = selectedProp.getChildrenCount;
        xIndex = strfind(propValue,'x');
        %spIndex = strfind(propValue,' ');
        M=str2double(propValue(2:xIndex-1));
        %N=str2double(propValue(xIndex+1:spIndex(1)-1));
    end
        
    if isStruct
        output=struct;
        % Extract out each child
        for ii=1:nStructs;
            if nStructs>1
                structLoopProp = selectedProp.getChildAt(ii-1);
            else
                structLoopProp = selectedProp;
            end
            
            for jj=1:structLoopProp.getChildrenCount
                child = structLoopProp.getChildAt(jj-1);
                fieldname = checkFieldName(child.getName);
                if M==1
                    output(1,ii).(fieldname) = selfBuildStruct(child);
                else
                    output(ii,1).(fieldname) = selfBuildStruct(child);
                end
            end
        end
    else
        switch class(propValue)
            case 'java.io.File'
                output = char(propValue);
            otherwise
                output = propValue;
        end
    end
end  % selfBuildStruct

% Replace any ' ' with an '_' in the output fieldname (see also getFieldLabel() above)
function fieldname = checkFieldName(fieldname)
    fieldname = char(fieldname);
    fieldname = regexprep(fieldname, ' \((.*)\)', '__$1');
    fieldname = strrep(fieldname, ' ', '_');
    fieldname(1) = upper(fieldname(1));
end  % checkFieldName

% Function to add the extra options (when popupmenu) to the output
function output = addOptionsToOutput(output, mirrorData, selectedProp)
    if isstruct(output) && isstruct(mirrorData)
        outputFields = fieldnames(output);
        mirrorFields = fieldnames(mirrorData);
        for ii=1:length(output)
            if length(output)>1
                structLoopProp = selectedProp.getChildAt(ii-1);
            else
                structLoopProp = selectedProp;
            end
            for jj=1:numel(outputFields)
                childProp = structLoopProp.getChildAt(jj-1);
                % sanity check this??????childProp.getName
                mirrorIndex = strcmpi(mirrorFields,outputFields{jj});
                if any(mirrorIndex)
                    mirrorField = mirrorFields{mirrorIndex};
                    if isfield(mirrorData(ii), mirrorField)
                        if ismember('arrayData',fieldnames(get(childProp)))
                            arrayData = get(childProp,'ArrayData');
                            output(ii).(outputFields{jj}) = arrayData;
                        else
                            if iscell(mirrorData(ii).(mirrorField))
                                % If original was a cell -> save originals as extra items in the cell array.

                                output(ii).(outputFields{jj}) = UpdateCellArray(mirrorData(ii).(outputFields{jj}),output(ii).(outputFields{jj}));
        %                         selectedIndex = find(strcmp(mirrorData.(mirrorField),output.(outputFields{ii})))==1;
        %                         
        %                         output.(outputFields{ii}) = {mirrorData.(mirrorField){:} {selectedIndex}};
                            elseif isstruct(mirrorData(ii).(mirrorField))
                                output(ii).(outputFields{jj}) = addOptionsToOutput(output(ii).(outputFields{jj}),mirrorData(ii).(mirrorField), childProp);
                            else
                                output(ii).(outputFields{jj}) = checkCharFieldForAbreviation(output(ii).(outputFields{jj}),mirrorData(ii).(mirrorField));
                            end
                        end
                    end                    
                end
            end
        end
    else
        if ismember('arrayData',fieldnames(get(selectedProp)))
            arrayData = get(selectedProp,'ArrayData');
            output = arrayData;            
        else
            if iscell(mirrorData)
                output = UpdateCellArray(mirrorData,output);
            else
                output = checkCharFieldForAbreviation(output,mirrorData);
            end
        end
    end
end  % addOptionsToOutput

% Check to see if a char was truncated on GUI building (>50)
function output = checkCharFieldForAbreviation(output,mirrorData)
    % This is to replace the ... with the original data
    if ischar(output) && ...       % Is it a char which has been truncated?
            length(output) == 53 && ...
            length(mirrorData) > 50 && ...
            strcmp(output(end-2:end),'...') && ...
            strcmp(output(1:50),mirrorData(1:50))
        output = mirrorData;
        
    end
end  % checkCharFieldForAbreviation

% Loop through the structure and replace any in case sensitive names
function output = updateOriginalVarNames(output, mirrorData)
    outputFields = fieldnames(output);
    for jj=1:length(output)
        if isempty(outputFields)
            output = mirrorData;
        else
            mirrorFields = fieldnames(mirrorData);
            for ii=1:numel(outputFields)
                mirrorIndex = strcmpi(mirrorFields,outputFields{ii});
                if any(mirrorIndex)
                    mirrorField = mirrorFields{mirrorIndex};
                    if ~strcmp(mirrorField, outputFields{ii})
                        output(jj).(mirrorField) = output(jj).(outputFields{ii});
                        if jj==length(output)
                            output = rmfield(output,outputFields{ii});
                        end
                    end
                    if isstruct(output(jj).(mirrorField))
                        output(jj).(mirrorField) = updateOriginalVarNames(output(jj).(mirrorField), mirrorData(jj).(mirrorField));
                    end
                end
            end
        end
    end
end  % updateOriginalVarNames
function [flag, index] = CheckStringForBrackets(str)
    index = [];
    flag = strcmp(str(1),'(') && strcmp(str(end),')');
    if flag
        index = max(str2num(regexprep(str,'[()]','')));  % this assumes it's always (1,N) or (N,1)
    end
end  % CheckStringForBrackets
