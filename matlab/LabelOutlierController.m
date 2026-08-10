classdef LabelOutlierController < handle
  % Controller for the "Suspicious Labels" dialog, which lists outlier labels
  % (by angle / distance / reprojection error) and lets the user jump to them.
  %
  % This replaces the bare uifigure function label_outlier_gui.m: state and
  % widget handles move from guidata into instance properties.  The heavy
  % outlier computation lives in the static fromLabeler() factory; the
  % constructor just builds the view from already-computed error tables (which
  % keeps it testable).

  properties (Transient)
    hFig  % the dialog figure
    labeler_  % a Labeler
    err_tables_  % 1xN cell of tables, one per outlier type
    buttonGroup_  % the outlier-type radiobutton group
    table_  % the results uitable
  end  % properties

  methods
    function obj = LabelOutlierController(labeler, errTables)
      % Build the dialog from already-computed error tables.  Use fromLabeler()
      % to compute the tables from a Labeler and construct in one step.
      obj.labeler_ = labeler ;
      obj.err_tables_ = errTables ;
      obj.createGui_() ;
    end  % function

    function delete(obj)
      deleteValidGraphicsHandles(obj.hFig) ;
      obj.hFig = [] ;
    end  % function

    function createGui_(obj)
      % Build the dialog figure and its widgets programmatically.
      fig = uifigure('Name', 'Suspicious Labels', 'Position', [100, 100, 400, 600]) ;
      fig.Units = 'Normalized' ;
      obj.hFig = fig ;

      % Radiobutton group selecting the outlier type shown in the table.
      bg = uibuttongroup(fig, 'Units', 'Normalized', 'Position', [0.05, 0.85, 0.9, 0.12]) ;
      obj.buttonGroup_ = bg ;
      bh = 20 ;
      uiradiobutton(bg, 'Text', 'Angle Outliers', 'Position', [5, 2*bh+2*5, 200, bh], 'UserData', 1) ;
      uiradiobutton(bg, 'Text', 'Distance Outliers', 'Position', [5, bh+5, 200, bh], 'UserData', 2) ;
      if numel(obj.err_tables_) > 2
        uiradiobutton(bg, 'Text', 'Reprojection Outliers', 'Position', [5, 2, 200, bh], 'UserData', 3) ;
      end

      % Results table.
      t = uitable(fig, 'Data', obj.err_tables_{1}, 'Units', 'Normalized', 'Position', [0.05, 0.05, 0.9, 0.8]) ;
      obj.table_ = t ;
      t.ColumnWidth = {'1x', '1x', '1x', '3x', '1x'} ;
      t.ColumnSortable = true ;

      bg.SelectionChangedFcn = @(s,e)(obj.radioActuated_(e)) ;
      t.CellSelectionCallback = @(s,e)(obj.cellClickTblActuated_(e)) ;
    end  % function

    function radioActuated_(obj, event)
      % Show the error table for the selected outlier type.
      obj.table_.Data = obj.err_tables_{event.NewValue.UserData} ;
    end  % function

    function cellClickTblActuated_(obj, event)
      % On a double-click, jump the Labeler to the selected outlier's
      % movie/frame/target.
      pause(0.5) ;  % allow the user time to add a second click
      if ~strcmpi(obj.hFig.SelectionType, 'open')
        return ;
      end
      lobj = obj.labeler_ ;
      id = event.Indices(1) ;
      tdat = obj.table_.Data ;
      newmov = tdat.('Mov')(id) ;
      if lobj.currMovie ~= newmov
        qstr = sprintf('Switch to movie %d?', newmov) ;
        res = questdlg(qstr, 'Switch Movie') ;
        if strcmp(res, 'Yes')
          lobj.movieSet(newmov) ;
        else
          return ;
        end
      end
      lobj.setFrame(tdat.('Frm')(id)) ;
      lobj.setTarget(tdat.('Lbl')(id)) ;
    end  % function
  end  % methods

  methods (Static)
    function controller = fromLabeler(labeler)
      % Compute the outlier tables from the Labeler and build the dialog.
      if ~labeler.hasMovie
        error('Need to have at least one movie in order to label outliers') ;
      end
      labeler.pushBusyStatus('Finding outliers in labels...') ;
      oc = onCleanup(@()(labeler.popBusyStatus())) ;  %#ok<NASGU>
      [errTables, labels] = label_outliers(labeler) ;
      if size(labels, 3) < 20
        error('No Labels or too few labels to detect outliers') ;
      end
      controller = LabelOutlierController(labeler, errTables) ;
    end  % function
  end  % methods (Static)
end  % classdef
