function out = ParameterSetup(hParent,yaml)
% ParameterSetup(hParent,handleObj)

assert(isscalar(hParent) && ishandle(hParent));
%assert(isscalar(handleObject) && isa(handleObject,'HandleObject'));

t = parseConfigYaml(yaml);
hFig = figure('ToolBar','none','Visible','off','menubar','none','Name','CPR Tracking Parameters');
centerOnParentFigure(hFig,hParent);

propertiesGUI(hFig,t);
h = findall(hFig,'type','hgjavacomponent');
LOFF = 0.025;
BOFF = 0.1;
pos = h.Position;
set(h,'Units','normalized','Position',[LOFF pos(2)+BOFF 1-2*LOFF pos(4)-BOFF]);
BOFF2 = 0.01;
BTNWIDTH = .2;
BTNGAP = .01;
hApply =  uicontrol('String','Apply','Units','normalized',...
  'Pos',[0.5-BTNWIDTH-BTNGAP/2 BOFF2 BTNWIDTH BOFF-2*BOFF2],...
  'FontUnits','pixels','fontsize',16,...
  'Tag','pbApply','Callback',@cbkApply);
hCncl =  uicontrol('String','Cancel','Units','normalized',...
  'Pos',[.5+BTNGAP/2 BOFF2 BTNWIDTH BOFF-2*BOFF2],...
  'FontUnits','pixels','fontsize',16,...
  'Tag','pbCncl','Callback',@cbkCncl);

%  btCancel = uicontrol('String','Cancel', 'Units','pixel', 'Pos',[100,5,60,30], 'Tag','btCancel', 'Callback',@(h,e)close(hFig)); %#ok<NASGU>


hFig.Visible = 'on';

  
hObj = HandleObj;
setappdata(hFig,'output',hObj);
%uiwait(h);
out = hObj.data;



function pbCreateProject_Callback(hObject, eventdata, handles)
cfg = genCurrentConfig(handles);
cfg.ProjectName = handles.etProjectName.String;
handles.output = cfg;
guidata(handles.figure1,handles);
close(handles.figure1);
function pbCancel_Callback(hObject, eventdata, handles)
handles.output = [];
guidata(handles.figure1,handles);
close(handles.figure1);
function pbCollapseNames_Callback(hObject, eventdata, handles)
function pbAdvanced_Callback(hObject, eventdata, handles)
handles = advModeToggle(handles);
guidata(handles.figure1,handles);
function pbCopySettingsFrom_Callback(hObject, eventdata, handles)
lastLblFile = RC.getprop('lastLblFile');
if isempty(lastLblFile)
  lastLblFile = pwd;
end
[fname,pth] = uigetfile('*.lbl','Select project file',lastLblFile);
if isequal(fname,0)
  return;
end
lbl = load(fullfile(pth,fname),'-mat');
lbl = Labeler.lblModernize(lbl);
cfg = lbl.cfg;
handles = setCurrentConfig(handles,cfg);
guidata(handles.figure1,handles);

function s = structLeavesStr2Double(s,flds)
% flds: cellstr of fieldnames
%
% Convert nonempty leaf nodes that are strs to doubles
for f=flds(:)',f=f{1}; %#ok<FXSET>
  val = s.(f);
  if isstruct(val)
    s.(f) = structLeavesStr2Double(s.(f),fieldnames(s.(f)));
  elseif ~isempty(val)
    if ischar(val)
      s.(f) = str2double(val);
    end
  else
    % none, empty
  end
end


