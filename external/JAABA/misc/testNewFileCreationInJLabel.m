function works=testNewFileCreationInJLabel()

try
  jLabelFigure=JLabel();
  pause(1);
  eventData=[];
  JLabel('menu_file_new_Callback',jLabelFigure, eventData, guidata(jLabelFigure));
  pause(1);
  projectSetupFigure=findall(0,'tag','figureProjectSetup');
  behaviorNameEdit=findall(projectSetupFigure,'tag','editName');
  set(behaviorNameEdit,'String','fooing');
  ProjectSetup('editName_Callback',behaviorNameEdit, eventData, guidata(projectSetupFigure));
  pause(1);
  doneButton=findall(projectSetupFigure,'tag','pushbutton_done');
  ProjectSetup('pushbutton_done_Callback',doneButton, eventData, guidata(projectSetupFigure));
  pause(1);
  %JLabel('menu_file_close_Callback',jLabelFigure, eventData, guidata(jLabelFigure));
  %pause(1);
  % If would be nice to test the "close" functionlaity here, but that's
  % difficult b/c we use questdlg() to ask whether the user wants to save
  % the new .jab file, and questdlg() calls uiwait(), so the call to
  % JLabel('menu_file_close_Callback',... blocks.  It would be nice to
  % write a version of questdlg that doesn't block, but uses
  % callbacks/delegates to handle the button presses.  But that would take
  % a while...
  delete(jLabelFigure);  
  works=true;
catch excp
  fprintf('Exception thrown in testNewFileCreationInJLabel:\n');
  fprintf('  identifier: %s\n',excp.identifier);
  fprintf('  message: %s\n',excp.message);
  works=false;
end
  
end
