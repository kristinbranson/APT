function varargout = LabelerGUI(varargin)
% LARVALABELER MATLAB code for LarvaLabeler.fig
%      LARVALABELER, by itself, creates a new LARVALABELER or raises the existing
%      singleton*.
%
%      H = LARVALABELER returns the handle to a new LARVALABELER or the handle to
%      the existing singleton*.
%
%      LARVALABELER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LARVALABELER.M with the given input arguments.
%
%      LARVALABELER('Property','Value',...) creates a new LARVALABELER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LabelerGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LabelerGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LarvaLabeler

% Last Modified by GUIDE v2.5 29-Jul-2015 09:20:40

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LabelerGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @LabelerGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1}) && exist(varargin{1}),
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function LabelerGUI_OpeningFcn(hObject,eventdata,handles,varargin) %#ok<INUSL>

if verLessThan('matlab','8.4')
  error('LabelerGUI:ver','LabelerGUI requires MATLAB version R2014b or later.');
end

handles.output = hObject;

handles.labelerObj = varargin{1};
varargin = varargin(2:end);

colormap(handles.figure,gray);

handles.image_curr = imagesc(0,'Parent',handles.axes_curr);
set(handles.image_curr,'hittest','off');
axisoff(handles.axes_curr);
hold(handles.axes_curr,'on');
set(handles.axes_curr,'Color',[0 0 0]);

handles.image_prev = imagesc(0,'Parent',handles.axes_prev);
set(handles.image_prev,'hittest','off');
axisoff(handles.axes_prev);
hold(handles.axes_prev,'on');
set(handles.axes_prev,'Color',[0 0 0]);

linkaxes([handles.axes_prev,handles.axes_curr]);

%fcn = get(handles.slider_frame,'Callback');
handles.hslider_listener = addlistener(handles.slider_frame,...
  'ContinuousValueChange',@slider_frame_Callback);
%set(handles.slider_frame,'Callback','');

set(handles.output,'Toolbar','figure');

colnames = handles.labelerObj.TBLTRX_STATIC_COLSTBL;
set(handles.tblTrx,'ColumnName',colnames,'Data',cell(0,numel(colnames)));
colnames = handles.labelerObj.TBLFRAMES_COLS;
set(handles.tblFrames,'ColumnName',colnames,'Data',cell(0,numel(colnames)));

guidata(hObject, handles);

% UIWAIT makes LabelerGUI wait for user response (see UIRESUME)
% uiwait(handles.figure);

function varargout = LabelerGUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function slider_frame_Callback(hObject,~)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles = guidata(hObject);
lObj = handles.labelerObj;
v = get(hObject,'Value');
f = round(1 + v * (lObj.nframes - 1));
lObj.setFrame(f);

function slider_frame_CreateFcn(hObject,~,~)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edit_frame_Callback(hObject,~,handles)
lObj = handles.labelerObj;
f = str2double(get(hObject,'String'));
if isnan(f)
  set(hObject,'String',num2str(lObj.currFrame));
  return;
end
f = min(max(1,round(f)),lObj.nframes);
set(hObject,'String',num2str(f));
if f ~= lObj.currFrame
  lObj.setFrame(f)
end 
  
function edit_frame_CreateFcn(hObject,~,~)
if ispc && isequal(get(hObject,'BackgroundColor'), ...
                   get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end

function pbClear_Callback(hObject, eventdata, handles)
handles.labelerObj.lblCore.clearLabels();

function tbAccept_Callback(hObject, eventdata, handles)
lc = handles.labelerObj.lblCore;
switch lc.state
  case LabelState.ADJUST
    lc.acceptLabels();
  case LabelState.ACCEPTED
    lc.unacceptLabels();
  otherwise
    assert(false);
end

function tblTrx_CellSelectionCallback(hObject, eventdata, handles) %#ok<*DEFNU>
% hObject    handle to tblTrx (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
lObj = handles.labelerObj;
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  lObj.setTarget(row);
end

function tblFrames_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to tblFrames (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
lObj = handles.labelerObj;
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  dat = get(hObject,'Data');
  lObj.setFrame(dat{row,1});
end

function sldZoom_Callback(hObject, eventdata, handles)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
lObj = handles.labelerObj;
zoomFac = get(hObject,'Value');
lObj.videoZoomFac(zoomFac);

function pbResetZoom_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.videoResetView();

%% menu
function menu_file_save_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.saveLblFile();

function menu_file_load_Callback(hObject, eventdata, handles)
res = questdlg('Save before closing current video?');
if strcmpi(res,'Cancel')
  return;
elseif strcmpi(res,'Yes')
  assert(false,'TODO');
end
handles.labelerObj.loadLblFile();

function tfcontinue = hlpSave(labelerObj)
tfcontinue = true;
if labelerObj.hasMovie
  res = questdlg('Save before closing current video?');
  switch res
    case 'No'
      % none
    case 'Cancel'
      tfcontinue = false;
    case 'Yes'
      assert(false,'XXX TODO');
      handles = SaveState(handles);
  end
end

function menu_file_openmovie_Callback(hObject,~,handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  lObj.loadMovie();
  if lObj.hasMovie
    lObj.labelingInit();
    lObj.lblCore.clearLabels(); 
  end
end

function menu_file_openmovietrx_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  lObj.loadMovie([],[]);
  if lObj.hasMovie
    lObj.labelingInit();
    lObj.lblCore.clearLabels(); 
  end
end

function menu_help_keyboardshortcuts_Callback(hObject, eventdata, handles)
h = handles.labelerObj.lblCore.getKeyboardShortcutsHelp();
msgbox(h,'Keyboard shortcuts','help','modal');

function CloseImContrast(labelerObj)
labelerObj.videoSetContrastFromAxesCurr();

function menu_setup_adjustbrightness_Callback(hObject, eventdata, handles)
hConstrast = imcontrast_kb(handles.axes_curr);
addlistener(hConstrast,'ObjectBeingDestroyed',@(s,e) CloseImContrast(handles.labelerObj));

% function menu_setup_createtemplate_Callback(hObject, eventdata, handles)
% 
% res = questdlg('Changing template will result in all labels being cleared. Save before doing this?');
% if strcmpi(res,'Cancel'),
%   return;
% elseif strcmpi(res,'Yes'),
%   assert(false,'TODO');
% end
% 
% labeler = handles.labelerObj;
% labeler.labelingInit(LabelMode.TEMPLATE);
% lblCore = handles.labelerObj.lblCore;
% lblCore.createTemplate();

%%%%% BELOW IS IN PROGRESS %%%%%%

% --------------------------------------------------------------------
function menu_file_quit_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_quit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CloseGUI(handles);

function figure_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
CloseGUI(handles);

function CloseGUI(handles)

res = questdlg('Save before closing?');
if strcmpi(res,'Cancel'),
  return;
elseif strcmpi(res,'Yes'),
  SaveState(handles);
end

delete(handles.figure);

% 
% % --------------------------------------------------------------------
% function import_template_Callback(hObject, eventdata, handles)
% % hObject    handle to import_template (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% global LARVALABELERSAVEFILE;
% 
% if isempty(LARVALABELERSAVEFILE),
%   defaultfile = '';
% else
%   defaultfile = LARVALABELERSAVEFILE;
% end
% 
% [f,p] = uigetfile('*.mat','Import template from...',defaultfile);
% if ~ischar(f),
%   return;
% end
% filename = fullfile(p,f);
% 
% if ~exist(filename,'file'),
%   warndlg(sprintf('File %s does not exist',filename),'File does not exist','modal');
%   return;
% end
% 
% L = load(filename);
% assert(isfield(L,'template'),'The file does not have any template');
% 
% handles.template = L.template;
% handles.npoints = size(handles.template,1);
% handles.templatecolors = jet(handles.npoints);%*.5+.5;
% 
% for ndx = 1:size(L.template,1)
%   x = L.template(ndx,1);
%   y = L.template(ndx,2);
%   handles.hpoly(ndx) = plot(handles.axes_curr,x,y,'w+','MarkerSize',20,'LineWidth',3);
%   handles.htext(ndx) = text(x+handles.dt2p,y,num2str(numel(handles.hpoly)),'Parent',handles.axes_curr);%,...
%   set(handles.hpoly(ndx),'Color',handles.templatecolors(ndx,:),...
%     'ButtonDownFcn',@(hObject,eventdata) PointButtonDownCallback(hObject,eventdata,handles.figure,ndx));
%   %addNewPositionCallback(handles.hpoly(i),@(pos) UpdateLabels(pos,handles.figure,i));
%   set(handles.htext(ndx),'Color',handles.templatecolors(ndx,:));
% 
% end
%  
% handles.labeledpos = nan([handles.npoints,2,handles.nframes,handles.nanimals]);
% handles.labeledpos(:,:,handles.f,handles.animal) = handles.template;
% handles.islocked = false(handles.nframes,handles.nanimals);
% handles.pointselected = false(1,handles.npoints);
% 
% delete(handles.posprev(ishandle(handles.posprev)));
% handles.posprev = nan(1,handles.npoints);
% for i = 1:handles.npoints,
%   handles.posprev(i) = plot(handles.axes_prev,nan,nan,'+','Color',handles.templatecolors(i,:),'MarkerSize',8);%,...
%     %'KeyPressFcn',handles.keypressfcn);
% end
% 
% if isfield(L,'templateloc')
%   handles.templateloc = L.templateloc;
%   handles.templatetheta = L.templatetheta;
%   guidata(hObject,handles);
%   pushbutton_template_Callback(hObject,eventdata,handles);
% else
%   guidata(hObject,handles);
% end
