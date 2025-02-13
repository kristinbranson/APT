function varargout = TrackMonitorGUI(varargin)
% TRACKMONITORGUI MATLAB code for TrackMonitorGUI.fig
%      TRACKMONITORGUI, by itself, creates a new TRACKMONITORGUI or raises the existing
%      singleton*.
%
%      H = TRACKMONITORGUI returns the handle to a new TRACKMONITORGUI or the handle to
%      the existing singleton*.
%
%      TRACKMONITORGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRACKMONITORGUI.M with the given input arguments.
%
%      TRACKMONITORGUI('Property','Value',...) creates a new TRACKMONITORGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TrackMonitorGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TrackMonitorGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TrackMonitorGUI

% Last Modified by GUIDE v2.5 05-Feb-2019 17:45:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TrackMonitorGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @TrackMonitorGUI_OutputFcn, ...
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


% --- Executes just before TrackMonitorGUI is made visible.
function TrackMonitorGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TrackMonitorGUI (see VARARGIN)

% Choose default command line output for TrackMonitorGUI
handles.output = hObject;
handles.vizobj = varargin{1};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TrackMonitorGUI wait for user response (see UIRESUME)

% --- Outputs from this function are returned to the command line.
function varargout = TrackMonitorGUI_OutputFcn(hObject, eventdata, handles) 
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
    handles.vizobj.abortTracking();
  case 'start'
    warning('not implemented');
  otherwise
    assert(false);
end


% --- Executes when user attempts to close figure_TrackMonitor.
function figure_TrackMonitor_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure_TrackMonitor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure

mode = get(handles.pushbutton_startstop,'UserData');
if strcmpi(mode,'stop'),
  
  res = questdlg({'Tracking currently in progress. Please stop tracking before'
    'closing this monitor. If you have already clicked Stop tracking,'
    'please wait for tracking processes to be killed before closing'
    'this monitor.'
    'Only override this warning if you know what you are doing.'},...
    'Stop tracking before closing monitor','Ok','Override and close anyways','Ok');
  if ~strcmpi(res,'Ok'),
    delete(hObject);
  end
  return;
  
elseif strcmpi(mode,'start') || strcmpi(mode,'done'),
  
  delete(hObject);
  
else
  % sanity check
  error('Bad userdata value for pushbutton_startstop');
end



function edit_trackerinfo_Callback(hObject, eventdata, handles)
% hObject    handle to edit_trackerinfo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_trackerinfo as text
%        str2double(get(hObject,'String')) returns contents of edit_trackerinfo as a double


% --- Executes during object creation, after setting all properties.
function edit_trackerinfo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_trackerinfo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
