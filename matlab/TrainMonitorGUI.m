function varargout = TrainMonitorGUI(varargin)
% TRAINMONITORGUI MATLAB code for TrainMonitorGUI.fig
%      TRAINMONITORGUI, by itself, creates a new TRAINMONITORGUI or raises the existing
%      singleton*.
%
%      H = TRAINMONITORGUI returns the handle to a new TRAINMONITORGUI or the handle to
%      the existing singleton*.
%
%      TRAINMONITORGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRAINMONITORGUI.M with the given input arguments.
%
%      TRAINMONITORGUI('Property','Value',...) creates a new TRAINMONITORGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TrainMonitorGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TrainMonitorGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TrainMonitorGUI

% Last Modified by GUIDE v2.5 19-Dec-2018 13:10:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TrainMonitorGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @TrainMonitorGUI_OutputFcn, ...
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


% --- Executes just before TrainMonitorGUI is made visible.
function TrainMonitorGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TrainMonitorGUI (see VARARGIN)

% Choose default command line output for TrainMonitorGUI
handles.output = hObject;
handles.vizobj = varargin{1};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TrainMonitorGUI wait for user response (see UIRESUME)
% uiwait(handles.figure_TrainMonitor);

% --- Outputs from this function are returned to the command line.
function varargout = TrainMonitorGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu_actions.
function popupmenu_actions_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_actions (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_actions contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_actions


% --- Executes during object creation, after setting all properties.
function popupmenu_actions_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_actions (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_action.
function pushbutton_action_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_action (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.vizobj.updateClusterInfo();

% --- Executes on button press in pushbutton_startstop.
function pushbutton_startstop_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_startstop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

switch hObject.UserData
  case 'stop'
    handles.vizobj.abortTraining();
  case 'start'
    handles.vizobj.startTraining();
  otherwise
    assert(false);
end

% --- Executes when user attempts to close figure_TrainMonitor.
function figure_TrainMonitor_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure_TrainMonitor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure

tfbatch = batchStartupOptionUsed; % ci
mode = get(handles.pushbutton_startstop,'UserData');

if strcmpi(mode,'stop') && ~tfbatch,
  
  res = questdlg({'Training currently in progress. Please stop training before'
    'closing this monitor. If you have already clicked Stop training,'
    'please wait for training processes to be killed before closing'
    'this monitor.'
    'Only override this warning if you know what you are doing.'},...
    'Stop training before closing monitor','Ok','Override and close anyways','Ok');
  if ~strcmpi(res,'Ok'),
    delete(hObject);
  end
  
elseif strcmpi(mode,'start') || strcmpi(mode,'done') || tfbatch,
  
  delete(hObject);
  
else
  % sanity check
  error('Bad userdata value for pushbutton_startstop');
end

