function res = isdisplay()
% res = isdisplay()
% check if there is a display set

res = ~(usejava('jvm') && ~feature('ShowFigureWindows'));