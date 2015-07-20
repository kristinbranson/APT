function varargout = SaveFMFSettings(varargin)
% SAVEFMFSETTINGS M-file for SaveFMFSettings.fig
%      SAVEFMFSETTINGS, by itself, creates a new SAVEFMFSETTINGS or raises the existing
%      singleton*.
%
%      H = SAVEFMFSETTINGS returns the handle to a new SAVEFMFSETTINGS or the handle to
%      the existing singleton*.
%
%      SAVEFMFSETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SAVEFMFSETTINGS.M with the given input arguments.
%
%      SAVEFMFSETTINGS('Property','Value',...) creates a new SAVEFMFSETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SaveFMFSettings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SaveFMFSettings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SaveFMFSettings

% Last Modified by GUIDE v2.5 04-Aug-2010 23:40:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SaveFMFSettings_OpeningFcn, ...
                   'gui_OutputFcn',  @SaveFMFSettings_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SaveFMFSettings is made visible.
function SaveFMFSettings_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SaveFMFSettings (see VARARGIN)

% Choose default command line output for SaveFMFSettings
handles.output = struct;
handles.archstr = computer('arch');
if length(varargin) >= 1,
  handles.previous_values = varargin{1};
else
  handles.previous_values = struct;
end

% Output FPS
if isfield(handles.previous_values,'OutputFPS'),
  handles.output.OutputFPS = handles.previous_values.OutputFPS;
else
  handles.output.OutputFPS = 30;
end
set(handles.edit_OutputFPS,'String',num2str(handles.output.OutputFPS));

% Compression
switch handles.archstr,
  case 'win32',
    handles.Compressions = {'None','MSVC','RLE','Cinepak','Indeo5'};
  case 'win64',
    handles.Compressions = {'None','MSVC','RLE','IYUV'};
  otherwise
    handles.Compressions = {'None'};
end
if isfield(handles.previous_values,'Compression') && ...
    ismember(handles.previous_values.Compression,handles.Compressions),
  handles.output.Compression = handles.previous_values.Compression;
else
  handles.output.Compression = 'None';
end

set(handles.popupmenu_Compression,'String',handles.Compressions,...
  'Value',find(strcmp(handles.Compressions,handles.output.Compression),1));

% Quality
if isfield(handles.previous_values,'Quality') && ...
    ~strcmp(handles.output.Compression,'None')
  handles.output.Quality = handles.previous_values.Quality;
else
  handles.output.Quality = 100;
end
set(handles.edit_Quality,'String',num2str(handles.output.Quality));
if strcmp(handles.output.Compression,'None'),
  set(handles.edit_Quality,'Enable','off');
end

% Start Frame
if isfield(handles.previous_values,'StartFrame'),
  handles.output.StartFrame = handles.previous_values.StartFrame;
else
  handles.output.StartFrame = 1;
end
set(handles.edit_StartFrame,'String',num2str(handles.output.StartFrame));

% End Frame
if isfield(handles.previous_values,'EndFrame'),
  handles.output.EndFrame = handles.previous_values.EndFrame;
else
  handles.output.EndFrame = inf;
end
set(handles.edit_EndFrame,'String',num2str(handles.output.EndFrame));

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SaveFMFSettings wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SaveFMFSettings_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(handles.figure1);

% --- Executes on selection change in popupmenu_Compression.
function popupmenu_Compression_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_Compression (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_Compression contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_Compression
handles.output.Compression = handles.Compressions{get(hObject,'Value')};
if strcmp(handles.output.Compression,'None'),
  handles.output.Quality = 100;
  set(handles.edit_Quality,'Enable','off','String',num2str(handles.output.Quality));
else
  set(handles.edit_Quality,'Enable','on');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function popupmenu_Compression_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_Compression (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit_Quality_Callback(hObject, eventdata, handles)
% hObject    handle to edit_Quality (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_Quality as text
%        str2double(get(hObject,'String')) returns contents of edit_Quality as a double

v = str2double(get(hObject,'String'));
if isnan(v) || v <= 0 || v > 100,
  set(hObject,'String',num2str(handles.output.Quality));
else
  handles.output.Quality = v;
  guidata(hObject,handles);
end

% --- Executes during object creation, after setting all properties.
function edit_Quality_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_Quality (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_Cancel.
function pushbutton_Cancel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.output = struct;
guidata(hObject,handles);
uiresume(handles.figure1);

% --- Executes on button press in pushbutton_OK.
function pushbutton_OK_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_OK (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

uiresume(handles.figure1);

function edit_OutputFPS_Callback(hObject, eventdata, handles)
% hObject    handle to edit_OutputFPS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_OutputFPS as text
%        str2double(get(hObject,'String')) returns contents of edit_OutputFPS as a double

v = str2double(get(hObject,'String'));
if isnan(v) || v <= 0,
  set(hObject,'String',num2str(handles.output.OutputFPS));
else
  handles.output.OutputFPS = v;
  guidata(hObject,handles);
end



function edit_StartFrame_Callback(hObject, eventdata, handles)
% hObject    handle to edit_StartFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_StartFrame as text
%        str2double(get(hObject,'String')) returns contents of edit_StartFrame as a double
s = get(hObject,'String');

v = round(str2double(s));
if v <= handles.output.EndFrame && v >= 1,
  handles.output.StartFrame = v;
else
  set(hObject,'String',num2str(handles.output.StartFrame));
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function edit_StartFrame_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_StartFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_EndFrame_Callback(hObject, eventdata, handles)
% hObject    handle to edit_EndFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_EndFrame as text
%        str2double(get(hObject,'String')) returns contents of edit_EndFrame as a double
s = strtrim(get(hObject,'String'));
if strcmpi(s,'inf') || strcmpi(s,'infty'),
  handles.output.EndFrame = inf;
  set(hObject,'String','Inf');
else
  v = round(str2double(s));
  if v >= 1 && v >= handles.output.StartFrame,
    handles.output.EndFrame = v;
  else
    set(hObject,'String',num2str(handles.output.EndFrame));
  end
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function edit_EndFrame_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_EndFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
handles.output = struct;
guidata(hObject,handles);
uiresume(handles.figure1);

