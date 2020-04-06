% Copyright 2015 The MathWorks, Inc.

%% Create a figure with a table inside
f = figure('Position',[1 250 560 420]);
t = uiextras.jTable.Table(...
    'Parent',f,...
    'Data',num2cell(magic(4)),...
    'CellSelectionCallback','disp(''SelectionChanged'')');
t.JTable.setCellSelectionEnabled(true)

