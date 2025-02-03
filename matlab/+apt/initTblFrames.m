function handles = initTblFrames(handles, isMA)
  tbl0 = handles.tblFrames;
  tbl0.Units = 'pixel';
  tw = tbl0.Position(3);
  if tw<50,  tw= 50; end
  tbl0.Units = 'normalized';
  if isMA
    COLNAMES = {'Frm' 'Tgts' 'Pts' 'ROIs'};
    COLWIDTH = {min(tw/4-1,80) min(tw/4-5,40) max(tw/4-7,10) max(tw/4-7,10)};
  else
    COLNAMES = {'Frame' 'Tgts' 'Pts'};
    COLWIDTH = {100 50 'auto'};
  end
  
  % if 1 
  set(tbl0,...
    'ColumnWidth',COLWIDTH,...
    'ColumnName',COLNAMES,...
    'Data',cell(0,numel(COLNAMES)),...
    'CellSelectionCallback',@(src,evt)cbkTblFramesCellSelection(src,evt),...
    'FontUnits','points',...
    'FontSize',9.75,... % matches .tblTrx
    'BackgroundColor',[.3 .3 .3; .45 .45 .45]);
end  % function

