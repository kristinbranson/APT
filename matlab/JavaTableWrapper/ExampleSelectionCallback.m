function ExampleSelectionCallback(h,e)
% Copyright 2015 The MathWorks, Inc.

disp('ExampleSelectionCallback start');

% Append the data
Data = h.Data;
Data(end+1,:) = Data(end,:);
h.Data = Data;

disp('ExampleSelectionCallback end');

