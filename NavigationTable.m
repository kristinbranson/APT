classdef NavigationTable < handle
  % Generalization of MMTableSingle
  
  properties
    jtable
    fcnRowSelected % fcn handle, fcnRowSelected(row)
    navOnSingleClick % If true, navigate on single click; otherwise require double-click
  end
  properties (Dependent)
    height % height of jtable
  end
  methods
    function v = get.height(obj)
      jt = obj.jtable;
      v = size(jt.Data,1);
    end
  end
  
  methods
    function obj = NavigationTable(hParent,posn,cbkSelectRow,varargin)
      % varargin: eg {'ColumnName',{'col1' 'col2'},...
      %   'ColumnPreferredWidth',[100 200]}
      assert(isgraphics(hParent) && isscalar(hParent));
      szassert(posn,[1 4]);
      assert(isa(cbkSelectRow,'function_handle'));
      
      jt = uiextras.jTable.Table(...
        'parent',hParent,...
        'Position',posn,...
        'SelectionMode','discontiguous',...
        'Editable','off',... %        'MouseClickedCallback',@(src,evt)obj.cbkTableClick(src,evt),...
        'CellSelectionCallback',@(src,evt)obj.cbkCellSelection(src,evt),...
        varargin{:});
      obj.jtable = jt;
      
      obj.fcnRowSelected = cbkSelectRow;
      obj.navOnSingleClick = false;
    end
  end
  methods
    
    % tbl: [nxnFld] table
    function setData(obj,tbl)
      jt = obj.jtable;
      assert(isequal(jt.ColumnName,tbl.Properties.VariableNames));
      %newdat = tbl{:,:};
      newdat = table2cell(tbl);
      if ~isequal(jt.Data,newdat)
        jt.Data = newdat;
      end
    end
    
    % Update currently selected row
    %
    % row: 1-based row indices into table. Currently, tables are expected 
    % never to re-sort by row.
    function setSelectedRows(obj,rows)
      jt = obj.jtable;
      tblnrows = size(jt.Data,1);
      rows = rows(:);
      if all(0<rows & rows<=tblnrows)
        jt.SelectedRows = rows;
      else
        jt.SelectedRows = [];
      end
    end
    
    function rows = getSelectedRows(obj)
      % IMPORTANT: currently CANNOT sort table by columns
      jt = obj.jtable;
      rows = sort(jt.SelectedRows);
    end
        
  end
  
  methods

    function navSelected(obj)
      rows = obj.getSelectedRows();
      if isempty(rows)
        % none
      else
        if numel(rows)>1
          warningNoTrace('NavigationTable:rows',...
            'Multiple rows selected. Using first row selected.');
        end
        obj.fcnRowSelected(rows(1));
      end
    end
    
    function cbkCellSelection(obj,src,evt)
      if isfield(evt,'Indices')
        rows = evt.Indices;
        if ~isempty(rows)
          obj.fcnRowSelected(rows(1));
        end
      end
    end
    
    function cbkTableClick(obj,src,evt)
      persistent chk
      PAUSE_DURATION_CHECK = 0.25;
      
      if obj.navOnSingleClick
        obj.navSelected();
        return;
      end
      
      if isempty(chk)
        chk = 1;
        pause(PAUSE_DURATION_CHECK); %Add a delay to distinguish single click from a double click
        if chk==1          
          % single-click
          chk = [];
        end
      else
        % double-click
        chk = [];
        obj.navSelected();
      end      
    end
    
  end
  
end