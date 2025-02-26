function hlpRemoveFocus(h,handles)
% Hack to manage focus. As usual the uitable is causing problems. The
% tables used for Target/Frame nav cause problems with focus/Keypresses as
% follows:
% 1. A row is selected in the target table, selecting that target.
% 2. If nothing else is done, the table has focus and traps arrow
% keypresses to navigate the table, instead of doing LabelCore stuff
% (moving selected points, changing frames, etc).
% 3. The following lines of code force the focus off the uitable.
%
% Other possible solutions: 
% - Figure out how to disable arrow-key nav in uitables. Looks like need to
% drop into Java and not super simple.
% - Don't use uitables, or use them in a separate figure window.
uicontrol(handles.txStatus);

