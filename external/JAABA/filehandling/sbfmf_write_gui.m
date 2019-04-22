function varargout = sbfmf_write_gui(varargin)
% SBFMF_WRITE_GUI M-file for sbfmf_write_gui.fig
%      SBFMF_WRITE_GUI, by itself, creates a new SBFMF_WRITE_GUI or raises the existing
%      singleton*.
%
%      H = SBFMF_WRITE_GUI returns the handle to a new SBFMF_WRITE_GUI or the handle to
%      the existing singleton*.
%
%      SBFMF_WRITE_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SBFMF_WRITE_GUI.M with the given input arguments.
%
%      SBFMF_WRITE_GUI('Property','Value',...) creates a new SBFMF_WRITE_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before sbfmf_write_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to sbfmf_write_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help sbfmf_write_gui

% Last Modified by GUIDE v2.5 13-Jan-2010 23:37:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @sbfmf_write_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @sbfmf_write_gui_OutputFcn, ...
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


% --- Executes just before sbfmf_write_gui is made visible.
function sbfmf_write_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to sbfmf_write_gui (see VARARGIN)

% Choose default command line output for sbfmf_write_gui
handles.output = hObject;

handles.readframe = varargin{1};
handles.nframes = varargin{2};
handles.differencemode = varargin{3};
handles.bsthresh = varargin{4};
handles.bsalgorithm = varargin{5};
handles.bsnframes = varargin{6};
handles.display = 'Original Image';

if isnan(handles.differencemode),
  handles.differencemode = 'Other';
end
if isnan(handles.bsthresh),
  handles.bsthresh = 1;
end
handles.framenumber = 1;
hold(handles.axes1,'off');
handles.him = image(handles.im,'parent',handles.axes1);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes sbfmf_write_gui wait for user response (see UIRESUME)
uiwait(handles.figure1);

function handles = setimage(handles)

switch lower(handles.display),
  
  case 'original image',
    set(him,'cdata',handles.im);
  case 'thresholded image',
    set(him,'cdata',handles.threshim);
  case 'reconstructed image',
    set(him,'cdata',handles.reconstructim);
  case 'reconstruction error'
    set(him,'cdata',handles.errorim);
    
end
% --- Outputs from this function are returned to the command line.
function varargout = sbfmf_write_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function frameslider_Callback(hObject, eventdata, handles)
% hObject    handle to frameslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.framenumber = get(hObject,'value')*handles.nframes + 1;
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function frameslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frameslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

sliderstep = 1/handles.nframes*[1,100];
set(hObject,'sliderstep',sliderstep);
set(hObject,'value',(handles.frame-1)/handles.nframes);

function frameedit_Callback(hObject, eventdata, handles)
% hObject    handle to frameedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of frameedit as text
%        str2double(get(hObject,'String')) returns contents of frameedit as a double


% --- Executes during object creation, after setting all properties.
function frameedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frameedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

set(hObject,'string',num2str(handles.framenumber));

% --- Executes on slider movement.
function threshslider_Callback(hObject, eventdata, handles)
% hObject    handle to threshslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function threshslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to threshslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

set(hObject,'sliderstep',[1,100]/255);
set(hObject,'value',handles.bsthresh/255);
set(hObject,'visible','off');

% --- Executes on selection change in displaypopupmenu.
function displaypopupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to displaypopupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns displaypopupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from displaypopupmenu


% --- Executes during object creation, after setting all properties.
function displaypopupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to displaypopupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

i = find(strcmpi(handles.display,get(hObject,'string')));
set(hObject,'value',i);
set(hObject,'visible','off');

function threshedit_Callback(hObject, eventdata, handles)
% hObject    handle to threshedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of threshedit as text
%        str2double(get(hObject,'String')) returns contents of threshedit as a double


% --- Executes during object creation, after setting all properties.
function threshedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to threshedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

set(hObject,'string',num2str(handles.bsthresh));
set(hObject,'visible','off');

% --- Executes on selection change in differencemode_popupmenu.
function differencemode_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to differencemode_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns differencemode_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from differencemode_popupmenu


% --- Executes during object creation, after setting all properties.
function differencemode_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to differencemode_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

i = find(strcmpi(handles.differencemode,get(hObject,'string')));
set(hObject,'value',i);
set(hObject,'visible','off');

% --- Executes on selection change in algorithmpopupmenu.
function algorithmpopupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to algorithmpopupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns algorithmpopupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from algorithmpopupmenu


% --- Executes during object creation, after setting all properties.
function algorithmpopupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to algorithmpopupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

i = find(strcmpi(handles.bsalgorithm,get(hObject,'string')));
set(hObject,'value',i);

function nframesedit_Callback(hObject, eventdata, handles)
% hObject    handle to nframesedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nframesedit as text
%        str2double(get(hObject,'String')) returns contents of nframesedit as a double


% --- Executes during object creation, after setting all properties.
function nframesedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nframesedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

set(hObject,'string',num2str(handles.bsnframes));

% --- Executes on button press in donepushbutton.
function donepushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to donepushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in cancelpushbutton.
function cancelpushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to cancelpushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in updatepushbutton.
function updatepushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to updatepushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
