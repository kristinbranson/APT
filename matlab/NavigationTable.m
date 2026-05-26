classdef NavigationTable < handle
  % A row-selectable table that fires a callback when the user clicks a
  % row.  Wraps a uitable; the parent passed to the constructor should
  % ultimately be rooted in a uifigure, since the modern uitable
  % properties used here (SelectionType, Selection, SelectionChangedFcn)
  % require that.

  properties
    uitable_         % the underlying uitable handle
    fcnRowSelected   % fcn handle with sig: fcnRowSelected(row, rowdata)
    data             % last data set via setData(), as a Matlab table
  end

  properties (Dependent)
    height           % row count of the displayed table
  end

  methods
    function v = get.height(obj)
      % Return the number of rows currently displayed.
      d = obj.uitable_.Data ;
      if isempty(d)
        v = 0 ;
      else
        v = size(d, 1) ;
      end
    end  % function
  end  % methods

  methods
    function obj = NavigationTable(hParent, posn, cbkSelectRow, varargin)
      % Construct a new navigation table inside hParent.
      %
      % cbkSelectRow: function handle with sig fcnRowSelected(row, rowdata)
      %   row is the 1-based row index into .data; rowdata is .data(row,:).
      %
      % varargin: optional name/value pairs.  Recognized:
      %   'ColumnName'           - cellstr of header labels (passed through)
      %   'ColumnPreferredWidth' - numeric vector of column widths in pixels
      % 'ColumnFormat' is accepted but ignored; uitable formats columns
      % by data type.  Other name/value pairs are passed through to
      % uitable.

      assert(isgraphics(hParent) && isscalar(hParent)) ;
      szassert(posn, [1 4]) ;
      assert(isa(cbkSelectRow, 'function_handle')) ;

      [columnWidths, extra] = ...
        myparse_nocheck(varargin, 'ColumnPreferredWidth', []) ;
      [~, extra] = ...
        myparse_nocheck(extra, 'ColumnFormat', {}) ;

      ut = uitable('Parent', hParent, ...
                   'Units', 'normalized', ...
                   'Position', posn, ...
                   'ColumnEditable', false, ...
                   'SelectionType', 'row', ...
                   'Multiselect', 'on', ...
                   extra{:}) ;
      if ~isempty(columnWidths)
        % Use uitable's weighted '<N>x' widths so columns share the
        % available width proportionally to the requested values, the
        % way uiextras.jTable.Table's ColumnPreferredWidth did.  If we
        % set ColumnWidth to a cell of numerics they become exact pixel
        % widths and the columns no longer fill the table.
        ut.ColumnWidth = arrayfun(@(w)(sprintf('%gx', w)), ...
                                  columnWidths, ...
                                  'UniformOutput', false) ;
      end

      obj.uitable_ = ut ;
      obj.fcnRowSelected = cbkSelectRow ;
      % Wire the selection callback last so it can't fire before
      % fcnRowSelected and uitable_ are populated.
      ut.SelectionChangedFcn = @(src,evt)(obj.cbkSelectionChanged_(src, evt)) ;
    end  % function

    function delete(obj)
      % Destructor.  Tears down the underlying uitable.
      delete(obj.uitable_) ;
      obj.uitable_ = [] ;
      obj.fcnRowSelected = [] ;
      obj.data = [] ;
    end  % function

    function setData(obj, tbl)
      % Set the displayed table data.  tbl should be a Matlab table.
      if ~isequal(obj.data, tbl)
        obj.uitable_.Data = tbl ;
        obj.data = tbl ;
      end
    end  % function

    function setSelectedRows(obj, rows)
      % Set the currently selected rows by 1-based row index.
      d = obj.uitable_.Data ;
      tableRowCount = size(d, 1) ;
      rows = rows(:) ;
      if ~isempty(rows) && all(0 < rows & rows <= tableRowCount)
        obj.uitable_.Selection = rows ;
      else
        obj.uitable_.Selection = [] ;
      end
    end  % function

    function rows = getSelectedRows(obj)
      % Return the currently selected row indices, sorted ascending.
      rows = sort(obj.uitable_.Selection) ;
    end  % function

    function cbkSelectionChanged_(obj, src, evt)  %#ok<INUSD>
      % Handler for uitable SelectionChangedFcn.
      sel = evt.Selection ;
      if ~isempty(sel)
        r = sel(1) ;
        obj.fcnRowSelected(r, obj.data(r,:)) ;
      end
    end  % function
  end  % methods
end  % classdef
