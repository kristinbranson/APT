function label_outlier_gui(lobj)

  if ~lobj.hasMovie ,
    error('Need to have at least one movie in order to label outliers')
  end    
  lobj.pushStatus('Finding outliers in labels...');
  oc = onCleanup(@()(lobj.popStatus())) ;
  
  [err_tables,labels] = label_outliers(lobj);
  if size(labels,3)<20 
    error('No Labels or too few labels to detect outliers')
  end
%%
  % Create a figure window
 %
  fig = uifigure('Name', 'Suspicious Labels', 'Position', [100, 100, 400, 600]);
  fig.Units = 'Normalized';
  h = guidata(fig);
  h.lobj = lobj;
  h.fig = fig;
  h.err_tables = err_tables;


  % Create a button group for radio buttons
  bg = uibuttongroup(fig, 'Units','Normalized','Position', [0.05, 0.85, 0.9, 0.12]);
  
  bh = 20;
   
  rb1 = uiradiobutton(bg,  'Text', 'Angle Outliers', 'Position', [5,2*bh+2*5,200,bh],'UserData',1); 
  rb2 = uiradiobutton(bg, 'Text', 'Distance Outliers', 'Position', [5,bh+5,200,bh],'UserData',2);
  if numel(err_tables)>2
    rb3 = uiradiobutton(bg, 'Text', 'Reprojection Outliers', 'Position', [5,2,200,bh],'UserData',3);  
  end
  %
  % Create a table
  t = uitable(fig, 'Data', err_tables{1},'Units','Normalized','Position', [0.05, 0.05, 0.9, 0.8]);
  h.table = t;
  h.table.ColumnWidth = {'1x','1x','1x','3x','1x'};
  guidata(fig,h);
  
  bg.SelectionChangedFcn=@radioCallback;
  h.table.CellSelectionCallback = {@cell_click_tbl,fig};  
  h.table.ColumnSortable = true;
  % Callback function to update table data based on selected radio button
  function radioCallback(src,event)
      % Option 1: Update table data with random values
      h1 = guidata(src);

      % Update the table data
      h1.table.Data = h1.err_tables{event.NewValue.UserData};
  end
%

  function cell_click_tbl(table_handle, event, fig_handle)
    pause(0.5); % to allow the user time to add a second click
    if ~strcmpi(fig_handle.SelectionType, 'open')
      return;
    end

    h1 = guidata(table_handle);
    id = event.Indices(1);
    tdat = table_handle.Data;
    newmov = tdat.('Mov')(id);
    if h1.lobj.currMovie ~= newmov;
      qstr = sprintf('Switch to movie %d?',newmov);
      res = questdlg(qstr,'Switch Movie');
      if strcmp(res, 'Yes')
        h1.lobj.movieSetGUI(newmov);
      else
        return
      end
    end
    lobj = h1.lobj;
    lobj.setFrameGUI(tdat.('Frm')(id));
    lobj.setTarget(tdat.('Lbl')(id));

  end

  
end


