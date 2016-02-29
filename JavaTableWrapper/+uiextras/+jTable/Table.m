classdef Table < hgsetget
    %  uiextras.jTable.Table - Class definition for Table
    %   The Table object places a Java table control within a figure or
    %   container.
    %
    % Syntax:
    %   t = uiextras.jTable.Table
    %   t = uiextras.jTable.Table('Property','Value',...)
    %
    %  uiextras.jTable.Table Properties:
    %
    %   BackgroundColor - controls background color of the table
    %
    %   ButtonUpFcn - callback when the mouse button goes up over the table
    %
    %   ButtonDownFcn - callback when the mouse button goes down over the
    %   table
    %
    %   CellEditCallback - callback for edits to the cell
    %
    %   CellSelectionCallback - callback for change in selection
    %
    %   ColumnEditable - boolean array the same size as number of columns,
    %   indicating whether each column is editable or not
    %
    %   ColumnFormat - cellstr array defining the data format for each
    %       column. Valid values are:
    %
    %           '' (default format)
    %           'boolean' (checkbox)
    %           'integer'
    %           'float'
    %           'bank'
    %           'date'
    %           'char' (single line text)
    %           'longchar' (multi-line text)
    %           'popup' (popup/dropdown single selection)
    %           'popupcheckbox' (popup/dropdown multi-selection)
    %
    %   ColumnFormatData - cell array the same size as ColumnFormat, and
    %       containing a cellstr list of choices for any column that has a
    %       popup list.
    %
    %   ColumnName - name of each column
    %
    %   ColumnMinWidth - minimum width of each column
    %
    %   ColumnMaxWidth - maximum width of each column
    %
    %   ColumnPreferredWidth - preferred width of each column
    %
    %   ColumnWidth - width of each column
    %
    %   ColumnResizable - whether each column is resizable (true/false)
    %
    %   ColumnResizePolicy - resize policy for columns. Valid values:
    %       'AUTO_RESIZE_OFF', 'AUTO_RESIZE_NEXT_COLUMN',
    %       'AUTO_RESIZE_SUBSEQUENT_COLUMNS', 'AUTO_RESIZE_LAST_COLUMN'
    %       'AUTO_RESIZE_ALL_COLUMNS'
    %       Doc: http://docs.oracle.com/javase/7/docs/api/javax/swing/JTable.html
    %
    %   Data - data in the table. Must be entered as a cell array.
    %
    %   Editable - controls whether the table node text is editable
    %
    %   Enabled - controls whether the table is enabled or disabled
    %
    %   FontAngle - font angle [normal|italic]
    %
    %   FontName - font name [string]
    %
    %   FontSize - font size [numeric]
    %
    %   FontWeight - font weight [normal|bold]
    %
    %   MouseClickedCallback - callback when the mouse is clicked on the
    %   table
    %
    %   MouseDraggedCallback - callback while the mouse is being dragged
    %   over the table
    %
    %   MouseMotionFcn - callback while the mouse is being moved over the
    %   table
    %
    %   Parent - handle graphics parent for the table, which should be
    %   a valid container including figure, uipanel, or uitab
    %
    %   Position - position of the table within the parent container
    %
    %   RowHeight - height of the rows in the table
    %
    %   SelectedRows - table rows that are currently selected
    %
    %   Tag - tag assigned to the table container
    %
    %   UIContextMenu - context menu for the table [uicontextmenu]
    %
    %   Units - units of the table container, used for determining the
    %   position
    %
    %   UserData - User data to store in the table node
    %
    %   Visible - controls visibility of the control
    %
    %  uiextras.jTable.Table Methods:
    %
    %   getCell - get a value from a single cell
    %
    %   setCell - set a value to a single cell
    %
    %   sizeColumnsToData - autosize column widths to best fit the data
    %
    %  uiextras.jTable.Table Example:
    %
    %   % Create the figure and display the table
    %   f = figure;
    %   t = uiextras.jTable.Table('Parent',f);
    %   t.Data = num2cell(magic(5));
    %
    % Known Issues:
    %   1. Changing the number of columns will reset certain column
    %   properties. To work around this issue, the number of columns should
    %   be specified by at least one column property-value pair upon
    %   construction. A future fix can be made by retaining and reapplying
    %   all column properties automatically when the number of columns
    %   changes.
    %
    %   2. Resizing a figure from the top or sometimes from a corner will
    %   cause the java component to not be positioned correctly. Resizing
    %   from the bottom will fix it. This will be fixed in a future
    %   version.
    %
    %   3. Setting per-column properties like ColumnName, ColumnWidth, or
    %   Data will increase the number of columns as needed. To decrease
    %   columns in the table, set the ColumnName property.
    %
    %   4. This table is set up for row-based selection only. Cell-based
    %   selection requires additional improvements.
    %
    
    %   Copyright 2013-2015 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: rjackey $
    %   $Revision: 1078 $  $Date: 2015-02-20 09:13:35 -0500 (Fri, 20 Feb 2015) $
    % ---------------------------------------------------------------------
    %#ok<*PROP>
    
    % DEVELOPER NOTE: Java objects that may be used in a callback must be
    % put on the EDT to make them thread-safe in MATLAB. Otherwise, they
    % could execute along side a MATLAB command and get into a thread-lock
    % situation. Methods of the objects put on the EDT will be executed on
    % the thread-safe EDT.
    
    
    %% User Properties
    properties(Dependent = true)
        BackgroundColor
    end
    properties
        ButtonDownFcn = ''
        ButtonUpFcn = ''
        CellEditCallback = ''
        CellSelectionCallback = ''
    end
    properties(Dependent = true)
        ColumnEditable
    end
    properties
        ColumnFormat = cell(1,0)
        ColumnFormatData = cell(1,0)
    end
    properties(Dependent = true)
        ColumnName
        ColumnResizable
        ColumnResizePolicy
        ColumnWidth
        ColumnMaxWidth
        ColumnMinWidth
        ColumnPreferredWidth
        Data
        Editable
        Enabled
        FontAngle
        FontName
        FontSize
        FontWeight
    end
    properties
        MouseClickedCallback = ''
        MouseDraggedCallback = ''
        MouseMotionFcn = ''
    end
    properties(Dependent = true)
        Parent
        Position
        RowHeight
        %RowName %RAJ - not implemented
        SelectedRows
        SelectionMode
    end
    properties
        Tag = ''
        %TooltipString %RAJ - not implemented
        UIContextMenu = []
    end
    properties(Dependent = true)
        Units
    end
    properties
        UserData = []
    end
    properties(Dependent = true)
        Visible = 'on'
    end
    
    %% Internal properties
    
    properties(GetAccess = protected, SetAccess = protected)
        hPanel % Main panel
        JTable % Java table
        JTableModel % Java table model
        JColumnModel
        JSelectionModel
        JScrollPane % Java scroll pane
        JEditor % alternate Java cell editor
        Container % HG container
        IsConstructed = false; %true when the constructor is complete (internal)
        CBEnabled = false; %callbacks enabled state (internal)
    end
    
    properties(Dependent = true, SetAccess = protected, GetAccess = protected)
        JColumn
    end
    
    properties (Hidden = true) 
        Debug = false;
    end
    
    properties(Constant = true, GetAccess = protected)
        ValidResizeModes = {
            'off'
            'next'
            'subsequent'
            'last'
            'all'
            };
        ValidSelectionModes = {
            'single'
            'contiguous'
            'discontiguous'
            };
    end
    
    
    %% Events
    events(NotifyAccess = private)
        DataChanged % data changed
        SelectionChanged % selection changed
    end % events
    
    
    %% Constructor and Destructor
    methods
        
        function obj = Table(varargin)
            %  uiextras.jTable.Table - Constructor for Table
            % -------------------------------------------------------------------------
            % Abstract: Constructs a new Table object.
            %
            % Syntax:
            %           obj = uiextras.jTable.Table('p1',v1,...)
            %
            % Inputs:
            %           Property-value pairs
            %
            % Outputs:
            %           obj - uiextras.jTable.Table object
            %
            % Examples:
            %           hFig = figure;
            %           obj =  uiextras.jTable.Table('Parent',hFig)
            %
            
            %----- Parse Inputs -----%
            p = inputParser;
            p.KeepUnmatched = false;
            
            % Define defaults and requirements for each parameter
            p.addParamValue('BackgroundColor',[1 1 1]); %#ok<*NVREPL>
            p.addParamValue('ButtonDownFcn','');
            p.addParamValue('ButtonUpFcn','');
            p.addParamValue('CellEditCallback','');
            p.addParamValue('CellSelectionCallback','');
            p.addParamValue('ColumnEditable',false(1,0));
            p.addParamValue('ColumnFormat',cell(1,0));
            p.addParamValue('ColumnFormatData',cell(1,0));
            p.addParamValue('ColumnName',cell(1,0));
            p.addParamValue('ColumnResizable',false(1,0));
            p.addParamValue('ColumnResizePolicy','subsequent');
            p.addParamValue('ColumnWidth',zeros(1,0));
            p.addParamValue('ColumnPreferredWidth',zeros(1,0));
            p.addParamValue('ColumnMaxWidth',zeros(1,0));
            p.addParamValue('ColumnMinWidth',zeros(1,0));
            p.addParamValue('Data',cell(0,0));
            p.addParamValue('Editable','on');
            p.addParamValue('Enabled','on');
            p.addParamValue('FontAngle','normal');
            p.addParamValue('FontName','MS Sans Serif');
            p.addParamValue('FontSize',10);
            p.addParamValue('FontWeight','normal');
            p.addParamValue('MouseClickedCallback','');
            p.addParamValue('MouseDraggedCallback','');
            p.addParamValue('MouseMotionFcn','');
            p.addParamValue('Parent',[]);
            p.addParamValue('Position',[0 0 1 1]);
            p.addParamValue('RowHeight',20);
            p.addParamValue('SelectedRows',zeros(0,1));
            p.addParamValue('SelectionMode','single');
            p.addParamValue('Tag','');
            p.addParamValue('UserData',[]);
            p.addParamValue('Units','normalized');
            p.addParamValue('UIContextMenu',[]);
            p.addParamValue('Visible','on');
            p.addParamValue('Debug',false);
            p.parse(varargin{:});
            
            % Add customizations to Java path
            uiextras.jTable.loadJavaCustomizations();
            
            % Which parameters are not at defaults and need setting?
            ParamsToSet = rmfield(p.Results, p.UsingDefaults);
            
            % Create the table
            obj.create(ParamsToSet);
            
            % Force drawing updates
            drawnow
            
            % Indicate construction is complete
            obj.IsConstructed = true;
            obj.CBEnabled = true;
            
        end % constructor
        
        function delete(obj)
            %delete  Destructor.
            
            % Disable callbacks
            obj.CBEnabled = false;
            
            % Check if container is already being deleted
            if strcmp(get(obj.Container, 'BeingDeleted'), 'off')
                delete(obj.Container)
            end
            
            % Remove references to the java objects
            obj.JTable = [];
            obj.JTableModel = [];
            obj.JColumnModel = [];
            obj.JSelectionModel = [];
            obj.JScrollPane = [];
            drawnow() % force repaint
            
        end % destructor
        
    end % structors
    
    
    %% Creation methods
    methods (Access = private)
        function create(obj, ParamsToSet)
            
            % Ensure all drawing is caught up before creating the table
            drawnow
            
            % Create table model
            %jTableModel = javaObjectEDT('javax.swing.table.DefaultTableModel');
            jTableModel = javaObjectEDT('com.mathworks.consulting.swing.table.TableModel');
            jTableModel = handle(jTableModel, 'callbackproperties');
            jTableModel.TableChangedCallback = @obj.onTableModelChanged;
            
            % Create table
            %jTable = javaObjectEDT('javax.swing.JTable', jTableModel);
            jTable = javaObjectEDT('com.mathworks.consulting.swing.table.Table', jTableModel);
            jTable.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION)
            jTable.getTableHeader().setReorderingAllowed(false)
            jTable = handle(jTable, 'callbackproperties');
            
            % Get selection model
            jSelectionModel = jTable.getSelectionModel();
            jSelectionModel = javaObjectEDT(jSelectionModel);
            jSelectionModel = handle(jSelectionModel, 'callbackproperties');
            jSelectionModel.ValueChangedCallback = @obj.onSelectionChanged;
            jSelectionModel.setSelectionMode(0); %default to single selection
            
            % Ensure there is a valid parent
            if ~isfield(ParamsToSet,'Parent') || isempty(ParamsToSet.Parent)
                Parent = gcf;
            else
                Parent = ParamsToSet.Parent;
            end
            ParamsToSet = rmfield(ParamsToSet,'Parent');
            
            % Create the base panel
            hPanel = uipanel(...
                'Parent',Parent,...
                'BorderType','none',...
                'Clipping','on',...
                'DeleteFcn',@(h,e)onDeleted(obj,h,e),...
                'ResizeFcn',@(h,e)onResize(obj,h,e),...
                'UserData',obj);
            
            % Draw table in scroll pane
            jScrollPane = javaObjectEDT('javax.swing.JScrollPane', jTable);
            [jScrollPane, hContainer] = javacomponent(...
                jScrollPane, [], hPanel);
            set(hContainer,'Units','normalized','Position',[0 0 1 1])
            
            % Draw table in scroll pane
            jScrollPane.setViewportView(jTable);
            
            % Set the java callbacks for the table and the blank scrollpane area
            CbProps = handle(jTable,'CallbackProperties');
            set(CbProps,'MouseClickedCallback',@(src,e)onMouseClick(obj,e,'table'))
            set(CbProps,'MousePressedCallback',@(src,e)onButtonDown(obj,e,'table'))
            set(CbProps,'MouseReleasedCallback',@(src,e)onButtonUp(obj,e,'table'))
            set(CbProps,'MouseDraggedCallback',@(src,e)onMouseDrag(obj,e,'table'))
            set(CbProps,'MouseMovedCallback',@(src,e)onMouseMotion(obj,e,'table'))
            CbProps = handle(jScrollPane,'CallbackProperties');
            set(CbProps,'MouseClickedCallback',@(src,e)onMouseClick(obj,e,'scrollpane'))
            set(CbProps,'MousePressedCallback',@(src,e)onButtonDown(obj,e,'scrollpane'))
            set(CbProps,'MouseReleasedCallback',@(src,e)onButtonUp(obj,e,'scrollpane'))
            set(CbProps,'MouseDraggedCallback',@(src,e)onMouseDrag(obj,e,'scrollpane'))
            set(CbProps,'MouseMovedCallback',@(src,e)onMouseMotion(obj,e,'scrollpane'))
            
            % Get the column model
            jColumnModel = jTable.getColumnModel();
            javaObjectEDT(jColumnModel);
            
            % Store
            obj.hPanel = hPanel;
            obj.JTable = jTable;
            obj.JTableModel = jTableModel;
            obj.JColumnModel = jColumnModel;
            obj.JSelectionModel = jSelectionModel;
            obj.JScrollPane = jScrollPane;
            obj.Container = hContainer;
            
            % Add properties to the java object for MATLAB data
            schema.prop(jTable,'Tag','MATLAB array');
            schema.prop(jTable,'UserData','MATLAB array');
            
            %RAJ - turn off sorting, as it's not fully implemented
            obj.JTable.setSortingEnabled(false);
           
            % Column Name must be handled first, if specified
            if isfield(ParamsToSet,'ColumnName') && ~isempty(ParamsToSet.ColumnName)
                obj.ColumnName = ParamsToSet.ColumnName;
                ParamsToSet = rmfield(ParamsToSet,'ColumnName');
            end
            
            % Set the batch of user-supplied property values
            set(obj,ParamsToSet);
            % Fields = fieldnames(ParamsToSet);
            % for fIdx = 1:numel(Fields)
            %     ThisField = Fields{fIdx};
            %     obj.(ThisField) = ParamsToSet.(ThisField);
            % end
            
            % Now set remaining properties that need to be handled last
            
            
            % Force resize
            obj.onResize();
            
        end %function
    end %methods
    
    
    %% User methods
    methods
        
        function value = getCell(obj, row, col)
            % getCell - Get a cell to the specified value
            % -------------------------------------------------------------------------
            % Abstract: Get a cell to the specified value
            %
            % Syntax:
            %           value = obj.getCell(row,col)
            %           value = getCell(obj,row,col)
            %
            % Inputs:
            %           obj - Table object
            %           row - row index to get (scalar)
            %           col - column index to get (scalar)
            %
            % Outputs:
            %           value - value from the cell
            %
            
            % Retrieve table model
            JTableModel = obj.JTableModel;
            
            % Get row and column counts
            cv = JTableModel.getColumnCount();
            rv = JTableModel.getRowCount();
            
            % Validate inputs
            validateattributes(row,{'numeric'},{'>',0,'<=',rv,'integer','finite','nonnan','scalar'});
            validateattributes(col,{'numeric'},{'>',0,'<=',cv,'integer','finite','nonnan','scalar'});
            
            % Read from table model
            value = JTableModel.getValueAt(row-1,col-1);
            
            % Cast cells containing an Object to a cell array
            if isa(value,'java.lang.Object[]')
                value = cell(value);
            end
            
        end
        
        
        function setCell(obj, row, col, value)
            % setCell - Set a cell to the specified value
            % -------------------------------------------------------------------------
            % Abstract: Set a cell to the specified value
            %
            % Syntax:
            %           obj.setCell(row,col,value)
            %           setCell(obj,row,col,value)
            %
            % Inputs:
            %           obj - Table object
            %           row - row index to set (scalar)
            %           col - column index to set (scalar)
            %           value - value to set
            %
            % Outputs:
            %           none
            %
            
            % Retrieve table model
            JTableModel = obj.JTableModel;
            
            % Get row and column counts
            cv = JTableModel.getColumnCount();
            rv = JTableModel.getRowCount();
            
            % Validate inputs
            validateattributes(row,{'numeric'},{'>',0,'<=',rv,'integer','finite','nonnan','scalar'});
            validateattributes(col,{'numeric'},{'>',0,'<=',cv,'integer','finite','nonnan','scalar'});
            validateattributes(value,{'cell','char','numeric','logical'},{});
            
            % Disable listener
            obj.CBEnabled = false;
            
            % Convert/cast the input to the correct format
            if iscell(value) && ~isempty(ThisValue)
                if isscalar(value)
                    value = value{:};
                else
                    value = obj.castToJavaArray(value);
                end
            end
            
            % Set the value
            JTableModel.setValueAt(value,row-1,col-1);
            
            % Enable listener
            obj.CBEnabled = true;
            
            % Raise event
            notify(obj, 'DataChanged')
            
        end
        
        
        function redraw(obj)
            %redraw  Redraw table.
            %
            %  t.redraw() requests a redraw of the table t.
            
            jScrollPane = obj.JScrollPane;
            jScrollPane.repaint(jScrollPane.getBounds())
            
        end % redraw
        
        
        function refreshRenderers(obj)
            
            % Get the table renderers object, which contains all the java
            % renderers.
            rObj = uiextras.jTable.TableRenderers.getRenderers();
            
            % Get the column formats and associated data
            Formats = obj.ColumnFormat;
            FormatData = obj.ColumnFormatData;
            
            % Loop on each column
            NumCol = obj.JTableModel.getColumnCount();
            for idx = 1:NumCol
                
                % Get the format data (if any) for the column
                if numel(Formats) >= idx
                    ThisFormat = Formats{idx};
                else
                    ThisFormat = '';
                    
                end
                if numel(obj.ColumnFormatData) >= idx
                    ThisFormatData = FormatData{idx};
                else
                    ThisFormatData = {};
                end
                
                % Get the Java renderer and editor
                [r,e] = rObj.getRenderer(ThisFormat, ThisFormatData);
                
                % Set the renderer and editor to the column
                jColumn = obj.JTable.getColumnModel().getColumn(idx-1);
                jColumn.setCellRenderer(r);
                jColumn.setCellEditor(e);
                
            end
            
            % Redraw the table to show the new renderers
            obj.redraw();
            
        end
        
        
        function sizeColumnsToData(obj)
            % sizeColumnsToData - Set column sizes automatically
            % -------------------------------------------------------------------------
            % Abstract: Set column sizes automatically to fit the contents
            %
            % Syntax:
            %           obj.sizeColumnsToData()
            %           sizeColumnsToData(obj)
            %
            % Inputs:
            %           obj - Table object
            %
            % Outputs:
            %           none
            %
            
            com.mathworks.mwswing.MJUtilities.initJIDE;
            com.jidesoft.grid.TableUtils.autoResizeAllColumns(obj.JTable);
            
        end %function
        
    end %methods
    
    
    %% Internal Static Methods (Helper Functions)
    methods(Access = protected, Static = true)
        
        function jArray = castToJavaArray(value)
            % Cast cell array to a Java Object
            
            sz = numel(value);
            jArray = javaArray('java.lang.Object',sz);
            for aIdx = 1:sz
                if ischar(value{aIdx})
                    jArray(aIdx) = javaObject('java.lang.String',value{aIdx});
                elseif isnumeric(value{aIdx})
                    jArray(aIdx) = javaObject('java.lang.Double',value{aIdx});
                else
                    jArray(aIdx) = '<Unsupported Type>';
                end
            end
            
        end %function
        
        
        function loadJavaCustomizations()
            % Loads the required custom Java .jar file
            
            % Define the jar file
            JarFile = 'MathWorksConsultingCustomJTable.jar';
            JarPath = fullfile(fileparts(mfilename('fullpath')), JarFile);
            
            % Check if the jar is loaded
            JavaInMem = javaclasspath('-dynamic');
            PathIsLoaded = any(strcmp(JavaInMem,JarPath));
            
            % Load the .jar file
            if ~PathIsLoaded
                javaaddpath(JarPath);
            end
            
        end %function
        
    end %methods
    
    %% Get/Set methods
    
    methods
        
        % BackgroundColor
        function value = get.BackgroundColor(obj)
            JColor = obj.JTable.getBackground();
            value = [JColor.getRed() JColor.getGreen() JColor.getBlue()] / 255;
        end % get.BackgroundColor
        
        function set.BackgroundColor(obj, value)
            validateattributes(value,{'numeric'},...
                {'vector','numel',3,'>=',0,'<=',1})
            JColor = javaObjectEDT('java.awt.Color',value(1),value(2),value(3));
            obj.JTable.setBackground(JColor);
        end % set.BackgroundColor
        
        
        % CBEnabled
        function set.CBEnabled(obj,value)
            drawnow;
            if isvalid(obj)
                obj.CBEnabled = value;
            end
            drawnow;
        end
        
        
        % ColumnEditable
        function value = get.ColumnEditable(obj)
            
            try
                jTableModel = obj.JTableModel;
                jEditable = jTableModel.isEditable().toArray();
            catch e
                jTableModel = java(obj.JTableModel); % remove UDD wrapper
                jEditable = jTableModel.isEditable().toArray();
                warning('uiextras:Table:UnknownError', e.message)
            end
            value = com.mathworks.consulting.swing.table.Utilities.object2logical(jEditable)';
            
        end % get.ColumnEditable
        
        
        function set.ColumnEditable(obj, value)
            
            % Check
            validateattributes(value,{'logical'},{'vector'});
            
            % Increase the number of columns if needed
            NumCol = obj.JColumnModel.getColumnCount;
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol)
            end
            
            % Get the editable vector
            jTableModel = obj.JTableModel;
            jEditable = jTableModel.isEditable();
            
            % Set
            for idx = 1:min(numel(value),jEditable.size())
                jEditable.set(idx-1,value(idx));
            end
            jTableModel.setEditable(jEditable);
            
            % Redraw the table to update the view
            obj.redraw();
            
        end % set.ColumnEditable
        
        
        % ColumnFormat
        function set.ColumnFormat(obj, value)
            
            % Validate the input
            [StatusOk, Message] = uiextras.jTable.TableRenderers.validateColumnFormat(value);
            if ~StatusOk
                error(Message);
            end
            
            % Increase the number of columns if needed
            NumCol = obj.JColumnModel.getColumnCount; %#ok<MCSUP>
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol) %#ok<MCSUP>
            end
            
            % Update the value
            obj.ColumnFormat = value;
            
            % Update the renderers and editors
            obj.refreshRenderers();
            
        end % set.ColumnFormat
        
        
        % ColumnFormatData
        function set.ColumnFormatData(obj, value)
            
            % Validate the input
            validateattributes(value,{'cell'},{'vector'})
            
            % Check that all cells are empty or cellstr arrays
            idxEmpty = cellfun(@isempty,value);
            idxCellStr = cellfun(@iscellstr,value(~idxEmpty));
            if ~all(idxCellStr)
                error('Invalid ColumnFormatData. A cell array of empty values or cellstr arrays is required.');
            end
            
            % Update the value
            obj.ColumnFormatData = value;
            
            % Update the renderers and editors
            obj.refreshRenderers();
            
        end % set.ColumnFormatData
        
        
        % ColumnName
        function value = get.ColumnName(obj)
            
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            value = cell(1,NumCol);
            for idx = 1:NumCol
                value{idx} = jColumns(idx).getHeaderValue();
            end
            
            %RAJ - removed below because setting the value in the model
            %overwrites the ColumnEditable values
            %jTableModel = obj.JTableModel;
            %ct = jTableModel.getColumnCount();
            %value = cell([1 ct]);
            %for idx = 1:ct
            %    value{idx} = char(jTableModel.getColumnName(idx-1));
            %end
            
        end % get.ColumnName
        
        function set.ColumnName(obj, value)
            
            % Increase the number of columns if needed
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            if numel(value) ~= NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol);
                jColumns = obj.JColumn;
            end
            
            % Create the column names
            NumNames = numel(value);
            identifiers = java.util.Vector();
            for idx = 1:NumNames
                identifiers.addElement(value{idx})
                jColumns(idx).setHeaderValue(value{idx});
            end
            %RAJ - removed below because setting the value in the model
            %overwrites the ColumnEditable values
            %obj.JTableModel.setColumnIdentifiers(identifiers)
            obj.redraw;
            
        end % set.ColumnName
        
        
        % ColumnMinWidth
        function value = get.ColumnMinWidth(obj)
            value = zeros(1,0);
            jColumn = obj.JColumn;
            for idx=numel(jColumn):-1:1
                value(1,idx) = jColumn(idx).getMinWidth();
            end
        end % get.ColumnMinWidth
        
        function set.ColumnMinWidth(obj, value)
            
            % Validate
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            validateattributes(value,{'numeric'},...
                {'nonnegative','integer','finite','nonnan','vector'});
            
            % Increase the number of columns if needed
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol)
                jColumns = obj.JColumn;
            end
            
            % Set
            for idx = 1:min(NumCol,numel(value))
                jColumns(idx).setMinWidth(value(idx));
            end
            
        end % set.ColumnMinWidth
        
        
        % ColumnMaxWidth
        function value = get.ColumnMaxWidth(obj)
            value = zeros(1,0);
            jColumn = obj.JColumn;
            for idx=numel(jColumn):-1:1
                value(1,idx) = jColumn(idx).getMaxWidth();
            end
        end
        
        function set.ColumnMaxWidth(obj, value)
            
            % Validate
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            validateattributes(value,{'numeric'},...
                {'positive','integer','finite','nonnan','vector'});
            
            % Increase the number of columns if needed
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol)
                jColumns = obj.JColumn;
            end
            
            % Set
            for idx = 1:min(NumCol,numel(value))
                jColumns(idx).setMaxWidth(value(idx));
            end
            
        end
        
        
        % ColumnPreferredWidth
        function value = get.ColumnPreferredWidth(obj)
            value = zeros(1,0);
            jColumns = obj.JColumn;
            for idx=numel(jColumns):-1:1
                value(1,idx) = jColumns(idx).getPreferredWidth();
            end
        end
        
        function set.ColumnPreferredWidth(obj, value)
            
            % Validate
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            validateattributes(value,{'numeric'},...
                {'nonnegative','integer','finite','nonnan','vector'});
            
            % Increase the number of columns if needed
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol)
                jColumns = obj.JColumn;
            end
            
            % Set
            for idx = 1:min(NumCol,numel(value))
                jColumns(idx).setPreferredWidth(value(idx));
            end
            
        end
        
        function value = get.ColumnWidth(obj)
            value = zeros(1,0);
            jColumns = obj.JColumn;
            for idx=numel(jColumns):-1:1
                value(1,idx) = jColumns(idx).getWidth();
            end
        end % get.ColumnWidth
        
        function set.ColumnWidth(obj, value)
            
            % Validate
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            validateattributes(value,{'numeric'},...
                {'nonnegative','integer','finite','nonnan','vector'});
            
            % Increase the number of columns if needed
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol)
                jColumns = obj.JColumn;
            end
            
            % Set
            for idx = 1:min(NumCol,numel(value))
                jColumns(idx).setWidth(value(idx));
            end
            
        end % set.ColumnWidth
        
        
        % ColumnResizable
        function value = get.ColumnResizable(obj)
            value = true(0,0);
            jColumns = obj.JColumn;
            for idx=numel(jColumns):-1:1
                value(1,idx) = jColumns(idx).getResizable();
            end
        end
        
        function set.ColumnResizable(obj, value)
            
            % Validate
            jColumns = obj.JColumn;
            NumCol = numel(jColumns);
            validateattributes(value,{'logical'},{'vector'});
            
            % Increase the number of columns if needed
            if numel(value) > NumCol
                NumCol = numel(value);
                obj.JTableModel.setColumnCount(NumCol)
                jColumns = obj.JColumn;
            end
            
            % Set
            for idx = 1:min(NumCol,numel(value))
                jColumns(idx).setResizable(value(idx));
            end
            
        end
        
        
        % ColumnResizePolicy
        function value = get.ColumnResizePolicy(obj)
            
            % Get policy index
            ModeIdx = obj.JTable.getAutoResizeMode();
            
            % Lookup and provide result
            value = obj.ValidResizeModes{ModeIdx+1};
            
        end
        
        function set.ColumnResizePolicy(obj, value)
            
            % Validate
            value = validatestring(value,obj.ValidResizeModes);
            
            % Get policy index
            ModeIdx = find(strcmp(value,obj.ValidResizeModes), 1) - 1;
            
            % Set
            obj.JTable.setAutoResizeMode(ModeIdx);
            
        end
        
        
        % Data
        function value = get.Data(obj)
            JTableModel = obj.JTableModel;
            NumRow = JTableModel.getRowCount();
            NumCol = JTableModel.getColumnCount();
            value = cell([NumRow NumCol]);
            for idx = 1:NumRow
                for jj = 1:NumCol
                    value{idx,jj} = JTableModel.getValueAt(idx-1, jj-1);
                    % Cast cells containing an Object to a cell array
                    if isa(value{idx,jj},'java.lang.Object[]')
                        value{idx,jj} = cell(value{idx,jj});
                    end
                end
            end
        end % get.Data
        
        function set.Data(obj, value)

            % Check
            assert(iscell(value) && ndims(value) == 2, ...
                'uiextras:Table:InvalidArgument', ...
                'Property ''Data'' must be a cell array.') %#ok<ISMAT>
            
            % Retrieve table model
            if isvalid(obj)
                JTableModel = obj.JTableModel;
                SelRows = obj.SelectedRows;
            end
            
            % Disable listener
            if isvalid(obj)
                obj.CBEnabled = false;
            end
            
            % Increase the number of columns if needed
            NumCol = size(value, 2);
            if isvalid(obj)
                if NumCol > JTableModel.getColumnCount()
                    JTableModel.setColumnCount(NumCol);
                end
            end
            
            % Add/remove rows if needed
            NumRow = size(value, 1);
            if isvalid(obj)
                if NumRow ~= JTableModel.getRowCount()
                    JTableModel.setRowCount(NumRow)
                end
            end
            
            % Populate table model
            if isvalid(obj)
                for idx = 1:NumRow
                    for jj = 1:NumCol
                        % Cast cells containing a cell array to an Object
                        ThisValue = value{idx,jj};
                        if iscell(ThisValue) && ~isempty(ThisValue)
                            ThisValue = obj.castToJavaArray(ThisValue);
                        end
                        JTableModel.setValueAt(ThisValue, idx-1, jj-1)
                    end
                end
            end
            
            % Enable listener
            if isvalid(obj)
                obj.CBEnabled = true;
            end
            
            % Raise events
            notify(obj,'DataChanged')
            if isvalid(obj) && ~isequal(SelRows, obj.SelectedRows)
                notify(obj,'SelectionChanged');
            end
            
        end % set.Data
        
        
        % Editable
        function value = get.Editable(obj)
            
            jEditor = obj.JTable.getDefaultEditor(...
                java.lang.Class.forName('java.lang.Object'));
            if isempty(jEditor)
                value = 'off';
            else
                value = 'on';
            end
            
        end % get.Editable
        
        
        function set.Editable(obj, value)
            
            % Check
            assert(ischar(value) && any(strcmp(value, {'on','off'})), ...
                'uiextras:Table:InvalidArgument', ...
                'Property ''Editable'' must be ''on'' or ''off''.')
            
            % Set
            jEditor = obj.JTable.getDefaultEditor(...
                java.lang.Class.forName('java.lang.Object'));
            if ~isempty(jEditor) && strcmp(value, 'off') % on to off
                obj.JEditor = jEditor;
                obj.JTable.setDefaultEditor(...
                    java.lang.Class.forName('java.lang.Object'), []);
            elseif isempty(jEditor) && strcmp(value, 'on') % off to on
                obj.JTable.setDefaultEditor(...
                    java.lang.Class.forName('java.lang.Object'), obj.JEditor);
                obj.JEditor = [];
            end
            
        end % set.Editable
        
        
        % Enabled
        function value = get.Enabled(obj)
            Enabled = get(obj.JTable,'Enabled');
            if Enabled
                value = 'on';
            else
                value = 'off';
            end
        end
        
        function set.Enabled(obj, value)
            value = validatestring(value,{'on','off'});
            Enabled = strcmp(value,'on');
            setEnabled(obj.JTable,Enabled);
            sb1 = get(obj.JScrollPane,'VerticalScrollBar');
            sb2 = get(obj.JScrollPane,'HorizontalScrollBar');
            setEnabled(sb1,Enabled);
            setEnabled(sb2,Enabled);
        end
        
        
        % FontAngle
        function value = get.FontAngle(obj)
            
            switch obj.JTable.getFont().isItalic()
                case true
                    value = 'italic';
                case false
                    value = 'normal';
            end
            
        end % get.FontAngle
        
        function set.FontAngle(obj, value)
            
            jTable = obj.JTable;
            jFont = jTable.getFont();
            switch value
                case 'normal'
                    jStyle = java.awt.Font.BOLD * jFont.isBold();
                case 'italic'
                    jStyle = java.awt.Font.BOLD * jFont.isBold() + ...
                        java.awt.Font.ITALIC;
                case 'oblique'
                    error('uiextras:Table:InvalidArgument', ...
                        'Value ''%s'' is not supported for property ''%s''.', ...
                        value, 'FontAngle')
                otherwise
                    error('uiextras:Table:InvalidArgument', ...
                        'Property ''FontAngle'' must be %s.', ...
                        '''normal'' or ''italic''')
            end
            jTable.setFont(javax.swing.plaf.FontUIResource(...
                jFont.getName(), jStyle, jFont.getSize()));
            
        end % set.FontAngle
        
        
        % FontName
        function value = get.FontName(obj)
            
            value = char(obj.JTable.getFont().getName());
            
        end % get.FontName
        
        function set.FontName(obj, value)
            
            jTable = obj.JTable;
            jFont = jTable.getFont();
            jTable.setFont(javax.swing.plaf.FontUIResource(...
                value, jFont.getStyle(), jFont.getSize()));
            
        end % set.FontName
        
        
        % FontSize
        function value = get.FontSize(obj)
            
            value = obj.JTable.getFont().getSize();
            
            % Convert value from pixels to points
            %http://stackoverflow.com/questions/6257784/java-font-size-vs-html-font-size
            % Java font is in pixels, and assumes 72dpi. Windows is
            % typically 96 and up, depending on display settings.
            dpi = java.awt.Toolkit.getDefaultToolkit().getScreenResolution();
            value = (value * 72 / dpi);
            
        end % get.FontSize
        
        function set.FontSize(obj, value)
            
            % Get the current font
            jFont = obj.JTable.getFont();
            
            % Convert value from points to pixels
            dpi = java.awt.Toolkit.getDefaultToolkit().getScreenResolution();
            value = round(value * dpi / 72);
            
            % Create a new Java font
            jFont = javax.swing.plaf.FontUIResource(jFont.getName(),...
                jFont.getStyle(), value);
            
            % Set
            obj.JTable.setFont(jFont);
            
            % Refresh
            obj.redraw();
            
        end % set.FontSize
        
        
        %FontWeight
        function value = get.FontWeight(obj)
            
            switch obj.JTable.getFont().isBold()
                case true
                    value = 'bold';
                case false
                    value = 'normal';
            end
            
        end % get.FontWeight
        
        function set.FontWeight(obj, value)
            
            jTable = obj.JTable;
            jFont = jTable.getFont();
            switch value
                case 'normal'
                    jStyle = jFont.isItalic() * java.awt.Font.ITALIC;
                case 'bold'
                    jStyle = jFont.isItalic() * java.awt.Font.ITALIC + ...
                        java.awt.Font.BOLD;
                case {'light','demi'}
                    error('uiextras:Table:InvalidArgument', ...
                        'Value ''%s'' is not supported for property ''%s''.', ...
                        value, 'FontWeight')
                otherwise
                    error('uiextras:Table:InvalidArgument', ...
                        'Property ''FontWeight'' must be %s.', ...
                        '''normal'' or ''bold''')
            end
            jTable.setFont(javax.swing.plaf.FontUIResource(...
                jFont.getName(), jStyle, jFont.getSize()));
            
        end % set.FontWeight
        
        
        % JColumn
        function value = get.JColumn(obj)
            
            NumCol = obj.JColumnModel.getColumnCount;
            for idx = 1:NumCol
                value(idx) = obj.JColumnModel.getColumn(idx-1); %#ok<AGROW>
            end
            if ~NumCol
                value = zeros(1,0);
            end
            
        end
        
        
        % Parent
        function value = get.Parent(obj)
            
            value = get(obj.hPanel, 'Parent');
            
        end % get.Parent
        
        function set.Parent(obj, value)
            
            set(obj.hPanel, 'Parent', double(value));
            
        end % set.Parent
        
        
        % Position
        function value = get.Position(obj)
            
            value = get(obj.hPanel, 'Position');
            
        end % get.Position
        
        function set.Position(obj, value)
            
            set(obj.hPanel, 'Position', value);
            
        end % set.Position
        
        
        % RowHeight
        function value = get.RowHeight(obj)
            
            value = obj.JTable.getRowHeight();
            
        end % get.RowHeight
        
        function set.RowHeight(obj, value)
            
            % Check
            validateattributes(value,{'numeric'},{'real','positive','scalar'});
            
            % Set
            obj.JTable.setRowHeight(value);
            
        end % set.RowHeight
        
        
        % SelectedRows
        function value = get.SelectedRows(obj)
            
            value = obj.JTable.getSelectedRows() + 1;
            
        end % get.SelectedRows
        
        function set.SelectedRows(obj, value)
            
            % Check
            validateattributes(value,{'numeric'},{'real','integer',...
                'positive','<=',obj.JTableModel.getRowCount()});
            
            % Turn off callbacks
            if isvalid(obj)
                obj.CBEnabled = false;
            end
            
            % Set the new value
            if isvalid(obj)
                obj.JSelectionModel.clearSelection()
                for idx = 1:numel(value)
                    obj.JSelectionModel.addSelectionInterval(value(idx)-1, value(idx)-1);
                end
            end
            
            % Scroll to the selection
            if isvalid(obj)
                if ~isempty(value)
                    obj.JTable.scrollRowToVisible(value(1));
                end
            end
            
            % Turn on callbacks
            if isvalid(obj)
                obj.CBEnabled = true;
            end
            
        end % set.SelectedRows
        
        
        % SelectionMode
        function value = get.SelectionMode(obj)
            
            % Get the value from JTable
            mIdx = obj.JSelectionModel.getSelectionMode();
            value = obj.ValidSelectionModes{mIdx+1};
            
        end
        
        function set.SelectionMode(obj, value)
            
            % Check
            value = validatestring(value,obj.ValidSelectionModes);
            
            % Turn off callbacks
            obj.CBEnabled = false;
            
            % Set the value in JTable
            mIdx = find(strcmp(value,obj.ValidSelectionModes),1);
            obj.JSelectionModel.setSelectionMode(mIdx-1)
            
            % Turn on callbacks
            obj.CBEnabled = true;
            
        end
        
        
        % Tag
        function set.Tag(obj, value)
            
            validateattributes(value,{'char'},{});
            obj.Tag = value;
            
        end %set.Tag
        
        
        % UIContextMenu
        function set.UIContextMenu(obj, value)
            
            % Check
            if ~isempty(value) && ( ~ishghandle(value) || ~isscalar(value) || ...
                    ~strcmp(get(value, 'Type'), 'uicontextmenu') )
                error('uiextras:Table:InvalidArgument', ...
                    'Property ''UIContextMenu'' must be a handle to a context menu.')
            end
            
            % Set
            obj.UIContextMenu = value;
            
        end % set.UIContextMenu
        
        
        % Units
        function value = get.Units(obj)
            
            value = get(obj.hPanel, 'Units');
            
        end % get.Units
        
        function set.Units(obj, value)
            
            set(obj.hPanel, 'Units', value);
            
        end % set.Units
        
        
        % Visible
        function value = get.Visible(obj)
            
            value = get(obj.hPanel, 'Visible');
            
        end % get.Visible
        
        function set.Visible(obj, value)
            
            % Check
            value = validatestring(value,{'on','off'});
            
            % Set
            set(obj.hPanel, 'Visible', value)
            
        end % set.Visible
        
    end % accessors (get/set methods)
    
    
    %% Event handlers
    methods(Access = private)
        
       
        function tf = callbacksEnabled(obj)
            % Check whether callbacks from Java should trigger a MATLAB
            % callback
            tf = isvalid(obj) && obj.CBEnabled;
            
        end
        
        
        function onDeleted(obj, ~, ~)
            %onDeleted  Event handler
            
            if obj.Debug, disp('onDeleted'); end
            
            obj.delete();
            
        end % onDeleted
        
        
        function onResize(obj, ~, ~)
            
            if obj.Debug, disp('onResize'); end
            
            if ~isempty(obj.Container) && ~isempty(obj.hPanel) &&...
                    ishghandle(obj.Container) && ishghandle(obj.hPanel)
                
                if obj.Debug, disp('onResize - entered'); end
                
                try %#ok<TRYNC>
                    pp = getpixelposition(obj.hPanel);
                    w = max(5, pp(3)-5);
                    h = max(5, pp(4)-5);
                    set(obj.Container,'Units','pixels','Position',[5 5 w h])
                end
                
            end
            
        end %onResize
        
        
        function onButtonDown(obj,e,location)
            
            if obj.Debug, disp('onButtonDown'); end
            
            if callbacksEnabled(obj) && ~isempty(obj.ButtonDownFcn)
                
                if obj.Debug, disp('onButtonDown - entered'); end
                
                % Get the position clicked
                x = e.getX;
                y = e.getY;
                
                % Get the cell clicked
                row = obj.JTable.rowAtPoint(e.getPoint) + 1;
                col = obj.JTable.columnAtPoint(e.getPoint) + 1;
                
                % Call the custom callback
                e1 = struct(...
                    'Position',[x,y],...
                    'Location',location,...
                    'Cell',[row col]);
                hgfeval(obj.ButtonDownFcn,obj,e1);
                
            end
            
        end %function
        
        function onButtonUp(obj,e,location)
            
            if obj.Debug, disp('onButtonUp'); end
            
            if callbacksEnabled(obj) && ~isempty(obj.ButtonUpFcn)
                
                if obj.Debug, disp('onButtonUp - entered'); end
                
                % Get the position clicked
                x = e.getX;
                y = e.getY;
                
                % Get the cell clicked
                row = obj.JTable.rowAtPoint(e.getPoint) + 1;
                col = obj.JTable.columnAtPoint(e.getPoint) + 1;
                
                % Call the custom callback
                e1 = struct(...
                    'Position',[x,y],...
                    'Location',location,...
                    'Cell',[row col]);
                hgfeval(obj.ButtonUpFcn,obj,e1);
                
            end
            
        end %function
        
        
        function onMouseClick(obj,e,location)
            % Occurs when the mouse is clicked within the pane
            
            if obj.Debug, disp('onMouseClick'); end
            
            if callbacksEnabled(obj)
                
                % Get the position clicked
                x = e.getX;
                y = e.getY;
                
                % Get the cell clicked
                row = obj.JTable.rowAtPoint(e.getPoint) + 1;
                col = obj.JTable.columnAtPoint(e.getPoint) + 1;
                
                % Which button was clicked?
                if e.isMetaDown
                    % Right-click
                    
                    if obj.Debug, disp('onMouseClick - context'); end
                    
                    % Without modifiers, select the row clicked, if it isn't already
                    if row>0 && ~all(obj.SelectedRows == row)
                        obj.SelectedRows = row;
                    end
                    
                    % Display the context menu
                    CMenu = obj.UIContextMenu;
                    if ~isempty(CMenu)
                        tPos = getpixelposition(obj.Container,true);
                        mPos = [x+tPos(1) tPos(2)+tPos(4)-y+obj.JScrollPane.getVerticalScrollBar().getValue()];
                        set(CMenu,'Position',mPos,'Visible','on');
                    end
                    
                elseif ~isempty(obj.MouseClickedCallback)
                    % Other click - fire any custom callback
                
                    if obj.Debug, disp('onMouseClick - entered'); end
                    
                    % Call the custom callback
                    e1 = struct(...
                        'Position',[x,y],...
                        'Location',location,...
                        'Cell',[row col]);
                    hgfeval(obj.MouseClickedCallback,obj,e1);
                    
                end %if e.isMetaDown
                
            end %if callbacksEnabled(obj)
            
        end %function onMouseClick
        
        
        function onMouseDrag(obj,e,location)
            
            if obj.Debug, disp('onMouseDrag'); end
            
            if callbacksEnabled(obj) && ~isempty(obj.MouseDraggedCallback)
                
                if obj.Debug, disp('onMouseDrag - entered'); end
                
                % Get the position clicked
                x = e.getX;
                y = e.getY;
                
                % Get the cell clicked
                row = obj.JTable.rowAtPoint(e.getPoint) + 1;
                col = obj.JTable.columnAtPoint(e.getPoint) + 1;
                
                % Call the custom callback
                e1 = struct(...
                    'Position',[x,y],...
                    'Location',location,...
                    'Cell',[row col]);
                hgfeval(obj.MouseDraggedCallback,obj,e1);
                
            end
            
        end %function
        
        function onMouseMotion(obj,e,location)
            
            %if obj.Debug, disp('onMouseMotion'); end
            
            if callbacksEnabled(obj) && ~isempty(obj.MouseMotionFcn)
                
                if obj.Debug, disp('onMouseMotion - entered'); end
                
                % Get the position clicked
                x = e.getX;
                y = e.getY;
                
                % Get the cell clicked
                row = obj.JTable.rowAtPoint(e.getPoint) + 1;
                col = obj.JTable.columnAtPoint(e.getPoint) + 1;
                
                % Call the custom callback
                e1 = struct(...
                    'Position',[x,y],...
                    'Location',location,...
                    'Cell',[row col]);
                hgfeval(obj.MouseMotionFcn,obj,e1);
                
            end
            
        end %function
        
        
        function onSelectionChanged(obj, ~, eventData)
            %onSelectionChanged  Event handler
            
            if obj.Debug, disp('onSelectionChanged'); end
            
            % This method may be triggered multiple times from a selection.
            % Don't process notifications until the time when changes are
            % complete.            
            if callbacksEnabled(obj) && ~eventData.getValueIsAdjusting()
                
                if obj.Debug, disp('onSelectionChanged - entered'); end
                
                % Trigger the event
                notify(obj, 'SelectionChanged')
                
                % Is there a callback to fire?
                if ~isempty(obj.CellSelectionCallback)
                    
                    % What was selected?
                    index = obj.SelectedRows;
                    
                    % Prepare event data
                    e1 = struct('Indices',index);
                    
                    % Call the callback
                    hgfeval(obj.CellSelectionCallback, obj, e1);
                    
                end
                
            end %if callbacksEnabled(obj) && ~eventData.getValueIsAdjusting()
            
        end % onSelectionChanged
        
        
        function onTableModelChanged(obj, ~, eventData)
            
            if obj.Debug, disp('onTableModelChanged'); end
            
            if callbacksEnabled(obj)
                
                if obj.Debug, disp('onTableModelChanged - entered'); end
                
                % What rows/columns were changed
                rv = eventData.getFirstRow() : eventData.getLastRow();
                cv = eventData.getColumn();
                
                % What happened?
                if all(rv == eventData.HEADER_ROW)
                    % column(s) added, removed or changed
                    
                    % Do nothing
                    
                elseif cv == eventData.ALL_COLUMNS
                    % row(s) added or removed
                    
                    % Do nothing
                    
                else
                    
                    % Prepare event data
                    indices(:,1) = rv'+1;
                    indices(:,2) = cv+1;
                    e1 = struct('Indices',indices);
                    
                    % Trigger the event
                    notify(obj, 'DataChanged')
                    
                    % Call the callback
                    hgfeval(obj.CellEditCallback, obj, e1);
                    
                end
                
            end %if callbacksEnabled(obj)
            
        end % onTableModelChanged
        
    end % event handlers
    
    
end % classdef