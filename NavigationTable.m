classdef NavigationTable < handle
  % Generalization of MMTableSingle
  
  properties
    jtable
    fcnRowSelected % fcn handle with sig: fcnRowSelected(row,rowdata) 
      % row is 1-based index into .data; rowdata is .data(row,:)
    navOnSingleClick % If true, navigate on single click; otherwise require double-click
    data % data in table form. jtable has it in cell form as well
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
      % cbkSelectRow: function handle with sig as in .fcnRowSelected
      % 
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
      jt.ColumnEditable(:) = false;
      obj.jtable = jt;
      
      obj.fcnRowSelected = cbkSelectRow;
      obj.navOnSingleClick = false;
    end
    function initColFormatAPTJava(obj,colfmt)
      % Initialize column cellrenderers based on colfmt. Optionally call 
      % this immediately after construction and before setting data.
      
      jt = obj.jtable.JTable;
      jcm = obj.jtable.JColumnModel;
      ncol = jt.ColumnCount;
      assert(iscellstr(colfmt) && numel(colfmt)==ncol,...
        'Invalid column format specification.');
      for icol=0:ncol-1
        switch colfmt{icol+1}
          case 'integer'
            cr = aptjava.StripedIntegerTableCellRenderer;
          case 'float'
            cr = aptjava.StripedFloatTableCellRenderer('%.3f');
          otherwise
            cr = javax.swing.table.DefaultTableCellRenderer;
        end
        jcm.getColumn(icol).setCellRenderer(cr);
      end
      jt.Foreground = java.awt.Color.WHITE;
      jt.repaint;
    end
    function delete(obj)
      delete(obj.jtable);
      obj.jtable = [];
      obj.fcnRowSelected = [];
      obj.navOnSingleClick = [];
      obj.data = [];
    end
  end
  methods
    
    % tbl: [nxnFld] table
    function setData(obj,tbl)
      jt = obj.jtable;
      %assert(isequal(jt.ColumnName,tbl.Properties.VariableNames));
      %newdat = tbl{:,:};
      newdat = table2cell(tbl);
      if ~isequal(jt.Data,newdat)
        jt.Data = newdat;
        obj.data = tbl;
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

%     function navSelected(obj)
%       rows = obj.getSelectedRows();
%       if isempty(rows)
%         % none
%       else
%         if numel(rows)>1
%           warningNoTrace('NavigationTable:rows',...
%             'Multiple rows selected. Using first row selected.');
%         end
%         obj.fcnRowSelected(rows(1));
%       end
%     end
    
    function cbkCellSelection(obj,src,evt)
      if isfield(evt,'Indices')
        rows = evt.Indices;
        if ~isempty(rows)
          r = rows(1);
          obj.fcnRowSelected(r,obj.data(r,:));
        end
      end
    end
    
%     function cbkTableClick(obj,src,evt)
%       persistent chk
%       PAUSE_DURATION_CHECK = 0.25;
%       
%       if obj.navOnSingleClick
%         obj.navSelected();
%         return;
%       end
%       
%       if isempty(chk)
%         chk = 1;
%         pause(PAUSE_DURATION_CHECK); %Add a delay to distinguish single click from a double click
%         if chk==1          
%           % single-click
%           chk = [];
%         end
%       else
%         % double-click
%         chk = [];
%         obj.navSelected();
%       end      
%     end
    
  end
  
end