% Copyright 2015 The MathWorks, Inc.

%% Create a figure with a table inside
f = figure('Position',[1 250 560 420],'NumberTitle','off',...
    'Toolbar','none','Menubar','none','Name','Java Table Wrapper - Column Format Examples');
t = uiextras.jTable.Table('Parent',f);


%% Adjust column formats
t.ColumnName = {'default','boolean','integer','float','bank','date','char','longchar','popup','popupcheckbox'};
t.ColumnFormat = {'','boolean','integer','float','bank','date','char','longchar','popup','popupcheckbox'};

% Additonal formatting info for 'popup' columns
t.ColumnFormatData{9} = {'apples','oranges','bananas'};

% Additonal formatting info for 'popupcheckbox' columns
t.ColumnFormatData{10} = {'fork','spoon','knife','spatula'};


%% Set the data programmatically

% Note: Date column format is not configured properly yet

% A really long string
LongStr = repmat('abcdefg ',1,100);

% Set the data
t.Data = {
    %def    bool    int     float   bank    date    char    longchar  popup      popupcheckbox
    20      false	20      20      20      1       'abc'	'abc'     'bananas'  'fork'
    1.5     true    1.5     1.5     1.5     1       ''      LongStr   'apples'   {'fork','knife','spatula'}
    22/7	false   10000   22/7	3.256	1       'defgh' LongStr   'oranges'  {}
    true	true	1e9     0.525	1000.2	1       LongStr 'efghi'   'apples'   {'spoon'}
    'abc'   true	50      1e9     -3      1       'lmnp'  ''        'apples'   {'knife','spoon'}
    };




