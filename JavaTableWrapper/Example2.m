% Copyright 2015 The MathWorks, Inc.

%% Create a figure with a table inside
f = figure('Position',[1 250 560 420]);
t = uiextras.jTable.Table(...
    'Parent',f,...
    'ColumnEditable',[false false true true],...
    'ColumnName',{'abc','def','ghi','jkl'},...
    'ColumnPreferredWidth',[40 40 100 100],...
    'CellEditCallback','disp(''DataChanged'')',...
    'CellSelectionCallback','disp(''SelectionChanged'')');

% t.ColumnName = {'abc','def','ghi','jkl'};

%% Set the data
t.Data = {
    '',     'x',    true,   'apples'
    '',     'y',    false,  'oranges'
    'abc',  'y',    true,  'oranges'
    };

%% Change column format
t.ColumnFormatData{4} = {'apples','oranges','bananas'};
t.ColumnFormat = {'','char','boolean','char'};

%% 
t.ColumnFormat{4} = 'popupcheckbox';

