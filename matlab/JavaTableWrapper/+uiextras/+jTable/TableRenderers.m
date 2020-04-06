classdef TableRenderers < handle
    %  uiextras.jTable.TableRenderers - Class definition for TableRenderers
    %   The TableRenderers object sets up reusable column renderers for the
    %   Java-based table
    %   
    
    %   Copyright 2013-2015 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: rjackey $
    %   $Revision: 1078 $  $Date: 2015-02-20 09:13:35 -0500 (Fri, 20 Feb 2015) $
    % ---------------------------------------------------------------------
    
    properties (Transient = true)
        Formats
    end
    
    
    %% Constructor
    % This constructor is set to private to create a singleton class. This
    % is because we want to only allow one instance that
    % can be retrieved from anywhere in the MATLAB session.
    methods (Access = private)
        function obj = TableRenderers()
            
            % Define and set up the formats
            obj.defineFormats();
            
        end
    end %methods
    
    
    %% Public Methods
    
    methods
        
        function defineFormats(obj)
            
            % Formats columns: [name  renderer  editor]
            % See http://www.jidesoft.com/javadoc/com/jidesoft/grid/package-summary.html
            obj.Formats = {
                ''          javax.swing.table.DefaultTableCellRenderer                          []
                'boolean'   com.jidesoft.grid.BooleanCheckBoxCellRenderer                       com.jidesoft.grid.BooleanCheckBoxCellEditor
                'integer'   com.mathworks.consulting.swing.table.NumberCellRenderer('#,##0')    com.jidesoft.grid.DoubleCellEditor
                'float'     com.jidesoft.grid.NumberCellRenderer                                com.jidesoft.grid.DoubleCellEditor
                'bank'      com.mathworks.consulting.swing.table.NumberCellRenderer('#,##0.00') com.jidesoft.grid.DoubleCellEditor
                'date'      com.jidesoft.grid.NumberCellRenderer                                com.jidesoft.grid.DateCellEditor
                'char'      javax.swing.table.DefaultTableCellRenderer                          []
                'longchar'  javax.swing.table.DefaultTableCellRenderer                          com.jidesoft.grid.MultilineStringCellEditor
                'popup'     javax.swing.table.DefaultTableCellRenderer                          []
                'popupcheckbox' com.jidesoft.grid.MultilineTableCellRenderer                    []
                };
            
        end %function
        
        
    end %methods
    
    
    %% Static Methods
    methods (Static = true)
        
        
        function obj = getRenderers()
            % This method returns the singleton TableRenderers object. Only one
            % TableRenderers is allowed in a MATLAB session, and this method will
            % retrieve it.
            
            persistent sObj
            
            % Does the object need to be instantiated?
            if isempty(sObj) || ~isvalid(sObj)
                sObj = uiextras.jTable.TableRenderers;
            end
            
            % Output the object
            obj = sObj;
            
        end %function
        
        
        function [StatusOk, Message] = validateColumnFormat(name)
            % Check the column format names are supported
            
            % Default outputs
            StatusOk = true;
            Message = '';
            
            % Get the singleton object
            obj = uiextras.jTable.TableRenderers.getRenderers();
            
            % Check the validity
            ValidNames = obj.Formats(:,1);
            if ~all(ismember(name, ValidNames))
                ValidNamesStr = sprintf('''%s'', ',ValidNames{:});
                Message = sprintf(...
                    'Invalid ColumnFormat specified. Valid values are %s',...
                    ValidNamesStr(1:end-2));
                StatusOk = false;
            end
            
        end %function
        
        
        function [Renderer, Editor] = getRenderer(type,data)
            
 
            % Get the singleton object
            obj = uiextras.jTable.TableRenderers.getRenderers();
            
            % Which renderer is selected?
            idx = strcmp(type,obj.Formats(:,1));
            
            % If bad selection use default (1st in list)
            if ~any(idx)
                idx(1) = true;
            end
            
            % Get the renderer and editor
            Renderer = obj.Formats{idx,2};
            Editor = obj.Formats{idx,3};
            
            % Recreate the editor with a list of possible values
            if ~isempty(data)
                switch type
                    
                    case 'popup'
                        try
                            % Works on newer releases, fails on R2011b
                            EditorType = 'com.jidesoft.grid.ListComboBoxCellEditor';
                            Editor = javaObject(EditorType, data);
                        catch %#ok<CTCH>
                            EditorType = 'javax.swing.JComboBox';
                            jComboBox = javaObject(EditorType, data);
                            EditorType = 'javax.swing.DefaultCellEditor';
                            Editor = javaObject(EditorType, jComboBox);
                        end
                        
                    case 'popupcheckbox'
                        EditorType = 'com.jidesoft.grid.CheckBoxListComboBoxCellEditor';
                        Editor = javaObject(EditorType, data);
                        
                    otherwise
                        % Do nothing - leave editor as-is
                        
                end %switch
            end %if ~isempty(data)
            
        end %function
        
    end %methods
    
end %classdef