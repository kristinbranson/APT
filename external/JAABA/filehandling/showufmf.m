function varargout = showufmf(varargin)
% showufmf
%
% Opens a GUI to show UFMF-compressed video
% Optional inputs:
% 
% UFMFName: Name of input video.
%
%

% Edit the above text to modify the response to help showufmf

% Last Modified by GUIDE v2.5 18-Jul-2014 11:42:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @showufmf_OpeningFcn, ...
                   'gui_OutputFcn',  @showufmf_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1}) && exist(varargin{1},'file'),
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before showufmf is made visible.
function showufmf_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to showufmf (see VARARGIN)

% Choose default command line output for showufmf
handles.output = hObject;

handles.MAXSLIDERRES = 10^6;

handles.figpos = get(handles.figure1,'Position');

% set up path
if isempty(which('myparse')),
  if exist('../misc','file'),
    addpath('../misc');
  end
  while isempty(which('myparse')),
    miscdir = uigetdir('.','Choose "misc" folder to add to path');
    try
      addpath(miscdir);
    catch %#ok<CTCH>
    end
  end
end
    

% load previous values
handles.rcfilename = '.showufmfrc.mat';
handles.previous_values = struct;
if exist(handles.rcfilename,'file')
  handles.previous_values = load(handles.rcfilename);
end

if isfield(handles.previous_values,'filename'),
  handles.filename = handles.previous_values.filename;
  if ~exist(handles.filename,'file'),
    handles.filename = '';
  end
else
  handles.filename = '';
end

handles.menu_Colormaps = get(handles.menu_Colormap,'Children');
if isfield(handles.previous_values,'Colormap'),
  handles.Colormap = handles.previous_values.Colormap;
else
  handles.Colormap = 'gray';
end

for h = handles.menu_Colormaps',
  set(h,'Checked','off');
end
switch handles.Colormap,
  case 'gray',
    set(handles.menu_ColormapGray,'Checked','on');
  case 'jet',
    set(handles.menu_ColormapJet,'Checked','on');
end

if isfield(handles.previous_values,'MaxFPS'),
  handles.MaxFPS = handles.previous_values.MaxFPS;
else
  handles.MaxFPS = 0;
  handles.MinSPF = 0;
end
if isfield(handles.previous_values,'BackSubThresh'),
  handles.BackSubThresh = handles.previous_values.BackSubThresh;
else
  handles.BackSubThresh = 10;
end

[handles.filename,handles.MaxFPS,handles.FMFName,handles.BackSubThresh,figpos,handles.FirstFrame] = myparse(varargin,...
  'UFMFName',handles.filename,'MaxFPS',handles.MaxFPS,'FMFName','','BackSubThresh',handles.BackSubThresh,...
  'FigPos',[],'FirstFrame',1);
if ~isempty(figpos),
  % nan means auto-set
  figpos(isnan(figpos)) = handles.figpos(isnan(figpos));
  handles.figpos = figpos;
  set(handles.figure1,'Units','Pixels','Position',figpos);
end
if handles.MaxFPS == 0,
  handles.MinSPF = 0;
else
  handles.MinSPF = 1/handles.MaxFPS;
end
handles.UFMFNameIsInput = ismember('UFMFName',varargin(1:2:end));
handles.FMFNameIsInput = ~isempty(handles.FMFName) && exist(handles.FMFName,'file');

s1 = {'Compressed Frame'
  'Background Model'
  'Background Difference'
  'Is Foreground'};
s2 = {'Raw Frame'
  'Compression Error'
  'Thresholded Compression Error'};
if handles.FMFNameIsInput,
  set(handles.popupmenu_Show,'String',[s1;s2]);
else
  set(handles.popupmenu_Show,'String',s1);
end

% set slider listener
fcn = get(handles.slider_Frame,'Callback');
if verLessThan('matlab','8.4.0')
  handles.hslider_listener = handle.listener(handles.slider_Frame,...
    'ActionEvent',fcn);
  set(handles.slider_Frame,'Callback','');
else
  handles.hslider_listener = addlistener(handles.slider_Frame,...
    'ContinuousValueChange',fcn);
  set(handles.slider_Frame,'Callback','');
end

%handles.slider_listener = handle.listener(handles.slider_Frame,'ActionEvent',fcn);

% open video
handles = open_fmf(handles);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes showufmf wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function play(hObject)

handles = guidata(hObject);
set(hObject,'String','Stop','BackgroundColor',[.5,0,0]);
handles.isplaying = true;
guidata(hObject,handles);
tic;
t0 = now;
fnext = handles.f;

while true,
  handles = guidata(hObject);
  if ~handles.isplaying,
    break;
  end
  f = fnext;
  handles.f = f;
  handles = update_frame(handles);
  guidata(hObject,handles);
  if handles.MaxFPS > 0,
    tmp = toc;
    t1 = now;
    if tmp < handles.MinSPF
      pause(handles.MinSPF - t1);
      fnext = f + 1;
    elseif tmp > handles.MinSPF,
      fnext = max(1,round((t1-t0)*24*3600/handles.MinSPF));
      %fprintf('setting fnext to %d\n',fnext);
      if fnext > handles.nframes,
        break;
    end
      drawnow;
  else
    drawnow;
  end
  else
    fnext = f+1;
    drawnow;
  end

  
%   if handles.MaxFPS > 0,
%     tmp = toc;
%     if tmp < handles.MinSPF
%       pause(handles.MinSPF - tmp);
%     end
%   else
%     drawnow;
%   end
  tic;
end

stop(hObject);

function stop(hObject)

handles = guidata(hObject);
handles.isplaying = false;
set(hObject,'String','Play','BackgroundColor',[0,.5,0]);
guidata(hObject,handles);

% --- Outputs from this function are returned to the command line.
function varargout = showufmf_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

function handles = update_frame(handles)

try
  [handles.im,handles.timestamp,handles.cachedidx,handles.bb,handles.mu] = handles.readframe(handles.f);
catch ME
  warning([sprintf('Error reading frame %d\n',handles.f),getReport(ME)]);
  return;
end
s = get(handles.popupmenu_Show,'String');
v = get(handles.popupmenu_Show,'Value');
showType = s{v};
switch showType,
  case 'Compressed Frame',
    set(handles.himage,'CData',handles.im);
    set(handles.axes_Video,'CLim',[0,255]);
  case 'Background Model',
    set(handles.himage,'CData',handles.mu);
    set(handles.axes_Video,'CLim',[0,255]);
  case 'Background Difference',
    if handles.FMFNameIsInput,
      im = handles.readframe_raw(handles.f);
      if size(im,3) > 1,
        im = rgb2gray(im);
      end
      im = cast(im,handles.headerinfo.dataclass);
    else
      im = handles.im;
    end
    diff = max(im - handles.mu,handles.mu - im);
    set(handles.himage,'CData',diff);
    set(handles.axes_Video,'CLim',[0,max(1,max(diff(:)))]);
  case 'Is Foreground',
    if handles.FMFNameIsInput,
      im = handles.readframe_raw(handles.f);
      if size(im,3) > 1,
        im = rgb2gray(im);
      end
      im = cast(im,handles.headerinfo.dataclass);
    else
      im = handles.im;
    end
    if size(im,3) > 1,
      im = rgb2gray(im);
    end
    diff = max(im - handles.mu,handles.mu - im);
    isfore = diff >= handles.BackSubThresh;
    set(handles.himage,'CData',isfore);
    set(handles.axes_Video,'CLim',[0,1]);    
  case 'Raw Frame',
    im = handles.readframe_raw(handles.f);
    if size(im,3) > 1,
      im = rgb2gray(im);
    end
    im = cast(im,handles.headerinfo.dataclass);
    handles.rawim = im;
    set(handles.himage,'CData',handles.rawim);
    set(handles.axes_Video,'CLim',[0,255]);
  case 'Compression Error',
    im = handles.readframe_raw(handles.f);
    if size(im,3) > 1,
      im = rgb2gray(im);
    end
    im = cast(im,handles.headerinfo.dataclass);
    handles.rawim = im;
    diff = max(handles.im - handles.rawim,handles.rawim - handles.im);
    set(handles.himage,'CData',diff);
    set(handles.axes_Video,'CLim',[0,max(1,max(diff(:)))]);
  case 'Thresholded Compression Error',
    im = handles.readframe_raw(handles.f);
    if size(im,3) > 1,
      im = rgb2gray(im);
    end
    im = cast(im,handles.headerinfo.dataclass);
    handles.rawim = im;
    diff = max(handles.im - handles.rawim,handles.rawim - handles.im);
    isfore = diff >= handles.BackSubThresh;
    set(handles.himage,'CData',isfore);
    set(handles.axes_Video,'CLim',[0,1]);    
end
if(handles.headerinfo.is_fixed_size)
  y = cat(2,handles.bb(:,1),...
    handles.bb(:,1),...
    handles.bb(:,1)+handles.headerinfo.max_width,...
    handles.bb(:,1)+handles.headerinfo.max_width,...
    handles.bb(:,1),...
    nan(size(handles.bb,1),1))'-.5;
  x = cat(2,handles.bb(:,2),...
    handles.bb(:,2)+handles.headerinfo.max_height,...
    handles.bb(:,2)+handles.headerinfo.max_height,...
    handles.bb(:,2),...
    handles.bb(:,2),...
    nan(size(handles.bb,1),1))'-.5;
else
  y = cat(2,handles.bb(:,1),...
    handles.bb(:,1),...
    handles.bb(:,1)+handles.bb(:,3),...
    handles.bb(:,1)+handles.bb(:,3),...
    handles.bb(:,1),...
    nan(size(handles.bb,1),1))'-.5;
  x = cat(2,handles.bb(:,2),...
    handles.bb(:,2)+handles.bb(:,4),...
    handles.bb(:,2)+handles.bb(:,4),...
    handles.bb(:,2),...
    handles.bb(:,2),...
    nan(size(handles.bb,1),1))'-.5;
end
set(handles.hboxes,'xdata',x(:),'ydata',y(:));
set(handles.edit_Frame,'String',num2str(handles.f));
set(handles.slider_Frame,'Value',handles.f);

% --- Executes on slider movement.
function slider_Frame_Callback(hObject, eventdata, handles)
% hObject    handle to slider_Frame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
v = get(hObject,'Value');
roundv = round(v);
if roundv == handles.f,
  if v > handles.f,
    handles.f = min(handles.nframes,handles.f+1);
  elseif v < handles.f,
    handles.f = max(1,handles.f-1);
  end
else
  handles.f = roundv;
end

handles = update_frame(handles);
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function slider_Frame_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_Frame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton_PlayStop.
function pushbutton_PlayStop_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_PlayStop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~handles.isplaying,
  play(hObject);
else
  stop(hObject);
end

function edit_Frame_Callback(hObject, eventdata, handles)
% hObject    handle to edit_Frame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_Frame as text
%        str2double(get(hObject,'String')) returns contents of edit_Frame as a double
f = str2double(get(hObject,'String'));
if isnan(f),
  set(hObject,'String',num2str(handles.f));
  return;
end
handles.f = max(1,min(f,handles.nframes));
handles = update_frame(handles);
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function edit_Frame_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_Frame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function menu_File_Callback(hObject, eventdata, handles)
% hObject    handle to menu_File (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_File_Open_Callback(hObject, eventdata, handles)
% hObject    handle to menu_File_Open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = open_fmf(handles);
guidata(hObject,handles);

function handles = open_fmf(handles)

if isfield(handles,'isplaying') && handles.isplaying,
  stop(handles.figure1);
  handles = guidata(handles.figure1);
end

if ~handles.UFMFNameIsInput,
  handles.filterspec = {'*.ufmf','MicroFlyMovieFormat (*.ufmf)'};
  
  [filename, pathname] = uigetfile(handles.filterspec, 'Choose UFMF video to play',handles.filename);
  if ~ischar(filename),
    return;
  end
  handles.filename = fullfile(pathname,filename);
else
  handles.UFMFNameIsInput = false;
end

if isfield(handles,'fid') && ~isempty(fopen(handles.fid)),
  fclose(handles.fid);
end
if isfield(handles,'fid_raw') && ~isempty(fopen(handles.fid_raw)),
  fclose(handles.fid_raw);
end
if isfield(handles,'himage') && ishandle(handles.himage),
  delete(handles.himage);
end

try
  [handles.readframe,handles.nframes,handles.fid,handles.headerinfo] = ...
    get_readframe_fcn(handles.filename);
catch ME
  s = sprintf('Could not read video %s.',handles.filename);
  uiwait(errordlg(s,'Error opening video'));
  rethrow(ME);
end

if handles.FMFNameIsInput,
  [handles.readframe_raw,handles.nframes_raw,handles.fid_raw,handles.headerinfo_raw] = ...
    get_readframe_fcn(handles.FMFName);
end

% set slider steps
% this seems to be the limit to slider step resolution
handles.stepsize = ceil(handles.nframes/handles.MAXSLIDERRES);
step1 = handles.stepsize/(handles.nframes-1);
sliderstep = [step1,min(1,100*step1)];
set(handles.slider_Frame,'Value',0,'SliderStep',sliderstep,'Min',1,'Max',handles.nframes);

% show first image
if ischar(handles.FirstFrame) && strcmpi(handles.FirstFrame,'maxnboxes'),
  nboxes = ufmf_read_nboxes(handles.headerinfo,1:handles.nframes);
  [~,handles.FirstFrame] = max(nboxes);
end
handles.f = handles.FirstFrame;
[handles.im,handles.timestamp,handles.cachedidx,handles.bb,handles.mu] = handles.readframe(handles.f);
hold(handles.axes_Video,'off');
if size(handles.im,3) == 1,
  handles.himage = imagesc(handles.im,'Parent',handles.axes_Video,[0,255]);
else
  handles.himage = image(uint8(handles.im),'Parent',handles.axes_Video);
end
colormap(handles.axes_Video,handles.Colormap);
axis(handles.axes_Video,'image','off');
axis(handles.axes_Video,[0,size(handles.im,2)+1,0,size(handles.im,1)+1]);
hold(handles.axes_Video,'on');
handles.hboxes = plot(handles.axes_Video,nan(1,2),nan(1,2),'r-');

handles = update_frame(handles);

handles.isplaying = false;

% --------------------------------------------------------------------
function menu_File_Quit_Callback(hObject, eventdata, handles)
% hObject    handle to menu_File_Quit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure1_CloseRequestFcn(handles.figure1, eventdata, handles);

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
if isfield(handles,'fid') && ~isempty(fopen(handles.fid)),
  fclose(handles.fid);
end
if isfield(handles,'fid_raw') && ~isempty(fopen(handles.fid_raw)),
  fclose(handles.fid_raw);
end
savefns = {'filename','FMFName','MaxFPS','BackSubThresh','Colormap'};
save(handles.rcfilename,'-struct','handles',savefns{:});
delete(hObject);


% --- Executes when figure1 is resized.
function figure1_ResizeFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isfield(handles,'figpos'),
  return;
end

sliderpos = get(handles.slider_Frame,'Position');
editpos = get(handles.edit_Frame,'Position');
pushbuttonpos = get(handles.pushbutton_PlayStop,'Position');
axespos = get(handles.axes_Video,'Position');
newfigpos = get(handles.figure1,'Position');
popupmenupos = get(handles.popupmenu_Show,'Position');

woff = newfigpos(3) - handles.figpos(3);
hoff = newfigpos(4) - handles.figpos(4);

sliderpos(3) = sliderpos(3) + woff;
editpos(1) = editpos(1) + woff;
pushbuttonpos(1) = pushbuttonpos(1) + woff;
popupmenupos(1) = popupmenupos(1) + woff;

axespos(3) = axespos(3) + woff;
axespos(4) = axespos(4) + hoff;

set(handles.slider_Frame,'Position',sliderpos);
set(handles.edit_Frame,'Position',editpos);
set(handles.pushbutton_PlayStop,'Position',pushbuttonpos);
set(handles.axes_Video,'Position',axespos);
set(handles.popupmenu_Show,'Position',popupmenupos);

handles.figpos = newfigpos;
guidata(hObject,handles);


% --------------------------------------------------------------------
function menu_Edit_Preferences_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Edit_Preferences (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

MAXFPS_IDX = 1;
BACKSUBTHRESH_IDX = 2;
prompts = {};
defAns = {};
prompts{MAXFPS_IDX} = 'Max FPS: ';
defAns{MAXFPS_IDX} = num2str(handles.MaxFPS);
prompts{BACKSUBTHRESH_IDX} = 'BackSub Thresh: ';
defAns{BACKSUBTHRESH_IDX} = num2str(handles.BackSubThresh);
while true,
  answer = inputdlg(prompts,'showufmf Preferences',1,defAns,'on');
  if isempty(answer),
    return;
  end
  v = str2double(answer{MAXFPS_IDX});
  iserror = false;
  if isnan(v),
    iserror = true;
  else
    defAns{MAXFPS_IDX} = num2str(v);
    handles.MaxFPS = v;
    handles.MinSPF = 1 / handles.MaxFPS;
  end
  v = str2double(answer{BACKSUBTHRESH_IDX});
  if isnan(v),
    iserror = true;
  else
    defAns{BACKSUBTHRESH_IDX} = num2str(v);
    handles.BackSubThresh = v;
  end
  
  if iserror,
    uiwait(warndlg('Illegal values entered. Max FPS and BackSub Thresh must be numbers','Bad Preferences'));
  else
    break;
  end
end
guidata(hObject,handles);


% --------------------------------------------------------------------
function menu_Help_About_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Help_About (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

s = {};
s{end+1} = 'ShowUFMF';
s{end+1} = '';
s{end+1} = 'Kristin Branson';
s{end+1} = 'bransonk@janelia.hhmi.org';
s{end+1} = '';
s{end+1} = 'This is a GUI for showing UFMF compression results. The maximum frame rate can be set through the "Preferences..." menu. Set to <= 0 for no maximum frame rate. Control the frame shown with the slider or editable text box. The Play/Stop button does what you think it does.';
msgbox(s,'About ShowUFMF','help','Replace');

% --------------------------------------------------------------------
function menu_Edit_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function menu_Help_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu_Show.
function popupmenu_Show_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_Show (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_Show contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_Show
handles = update_frame(handles);
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function popupmenu_Show_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_Show (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function menu_ShowForegroundBoxes_Callback(hObject, eventdata, handles)
% hObject    handle to menu_ShowForegroundBoxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

v = get(hObject,'Checked');
if strcmpi(v,'on'),
  v = 'off';
else
  v = 'on';
end
set(hObject,'Checked',v);

if isfield(handles,'hboxes'),
  set(handles.hboxes,'Visible',v);
end


% --------------------------------------------------------------------
function menu_Colormap_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Colormap (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_ColormapGray_Callback(hObject, eventdata, handles)
% hObject    handle to menu_ColormapGray (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if strcmpi(get(hObject,'Checked'),'off'),
  for h = handles.menu_Colormaps',
    if h ~= hObject,
      set(h,'Checked','off');
    end
  end
  set(hObject,'Checked','on');
  set(handles.hboxes,'color','r');
  handles.Colormap = 'gray';
  colormap(handles.axes_Video,handles.Colormap);
  guidata(hObject,handles);
end

% --------------------------------------------------------------------
function menu_ColormapJet_Callback(hObject, eventdata, handles)
% hObject    handle to menu_ColormapJet (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if strcmpi(get(hObject,'Checked'),'off'),
  for h = handles.menu_Colormaps',
    if h ~= hObject,
      set(h,'Checked','off');
    end
  end
  set(hObject,'Checked','on');
  set(handles.hboxes,'color','k');
  handles.Colormap = 'jet';
  colormap(handles.axes_Video,handles.Colormap);
  guidata(hObject,handles);
end


% --- Executes on key press with focus on figure1 or any of its controls.
function figure1_WindowKeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)

switch eventdata.Key,
  case {'leftarrow','downarrow'},
    handles.f = max(1,handles.f-1);
    handles = update_frame(handles);
    guidata(hObject,handles);
  case {'rightarrow','uparrow'},
    handles.f = min(handles.nframes,handles.f+1);
    handles = update_frame(handles);
    guidata(hObject,handles);
end
