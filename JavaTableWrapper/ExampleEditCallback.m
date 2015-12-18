function ExampleEditCallback(h,e)
% Copyright 2015 The MathWorks, Inc.

disp('ExampleEditCallback start');

% Put the data somewhere
ThisData = h.Data;
assignin('base','TempData',ThisData);
pause(0.1);

disp('ExampleEditCallback end');