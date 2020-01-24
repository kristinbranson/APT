function varargout = playfmf(varargin)
% PLAYFMF M-file for playfmf.fig
%      PLAYFMF, by itself, creates a new PLAYFMF or raises the existing
%      singleton*.
%
%      H = PLAYFMF returns the handle to a new PLAYFMF or the handle to
%      the existing singleton*.
%
%      PLAYFMF('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLAYFMF.M with the given input arguments.
%
%      PLAYFMF('Property','Value',...) creates a new PLAYFMF or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before playfmf_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to playfmf_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help playfmf

% Last Modified by GUIDE v2.5 25-Aug-2010 14:04:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @playfmf_OpeningFcn, ...
                   'gui_OutputFcn',  @playfmf_OutputFcn, ...
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


% --- Executes just before playfmf is made visible.
function playfmf_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to playfmf (see VARARGIN)


handles.MAXSLIDERRES = 10^6;

% Choose default command line output for playfmf
handles.output = hObject;

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
handles.rcfilename = '.playfmfrc.mat';
handles.previous_values = struct;
if exist(handles.rcfilename,'file')
  handles.previous_values = load(handles.rcfilename);
end

if isfield(handles.previous_values,'filename'),
  handles.filename = handles.previous_values.filename;
  [handles.filedir,handles.filenamebase,handles.fileext] = ...
    fileparts(handles.previous_values.filename);
  if ~exist(handles.filename,'file'),
    handles.filename = '';
  end
else
  handles.filename = '';
end

if isfield(handles.previous_values,'CompressionSettings'),
  handles.CompressionSettings = handles.previous_values.CompressionSettings;
else
  handles.CompressionSettings = struct;
end
if ~isfield(handles.CompressionSettings,'OutputFPS'),
  handles.CompressionSettings.OutputFPS = 30;
end
if ~isfield(handles.CompressionSettings,'Compression'),
  handles.CompressionSettings.Compression = 'None';
end
if ~isfield(handles.CompressionSettings,'Quality'),
  handles.CompressionSettings.Quality = 100;
end
handles.CompressionSettings.StartFrame = 1;
handles.CompressionSettings.EndFrame = inf;

if isfield(handles.previous_values,'MaxFPS'),
  handles.MaxFPS = handles.previous_values.MaxFPS;
else
  handles.MaxFPS = 0;
  handles.MinSPF = 0;
end

% set callback for slider motion
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

% open video
if numel(varargin) >= 2 && exist(varargin{2},'file'),
  filename = varargin{2};
  try
    handles = open_fmf(handles,filename,varargin{3:end});
  catch
    handles = open_fmf(handles);
  end
else
  handles = open_fmf(handles);
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes playfmf wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function play(hObject)

global ISPLAYING;

handles = guidata(hObject);
set(hObject,'String','Stop','BackgroundColor',[.5,0,0]);
ISPLAYING = true;
tic;
t0 = now;
fnext = handles.f;
f0 = handles.f;
while true,
  handles = guidata(hObject);
  if ~ISPLAYING || fnext > handles.nframes,
    break;
  end
  f = fnext;
  handles.f = f;
  handles = update_frame(handles,hObject);
  guidata(hObject,handles);
  if handles.MaxFPS > 0,
    tmp = toc;
    t1 = now;
    if tmp < handles.MinSPF
      pause(handles.MinSPF - t1);
      fnext = f + 1;
    elseif tmp > handles.MinSPF,
      fnext = max(1,f0+round((t1-t0)*24*3600/handles.MinSPF));
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
  tic;
end

stop(hObject);

function stop(hObject)

global ISPLAYING;

handles = guidata(hObject);
ISPLAYING = false;
guidata(hObject,handles);
set(hObject,'String','Play','BackgroundColor',[0,.5,0]);

% --- Outputs from this function are returned to the command line.
function varargout = playfmf_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

function handles = update_frame(handles,hObject)

if nargin == 1,
  hObject = nan;
end
persistent update_frame_f;

if ~isempty(update_frame_f) && update_frame_f == handles.f,
  return;
end
update_frame_f = handles.f;
if handles.f ~= handles.lastfread,
  try
    [handles.im,handles.timestamp] = handles.readframe(handles.f);
    handles.lastfread = handles.f;
    guidata(handles.figure1,handles);
  catch ME
    warning('Failed to read frame %d: %s',handles.f,getReport(ME));
  end  
%   if isa(handles.im,'uint16'),
%     handles.im = im2uint8(handles.im);
%   end
  set(handles.himage,'CData',handles.im);
end
if hObject ~= handles.edit_Frame,
  set(handles.edit_Frame,'String',num2str(handles.f));
end
if hObject ~= handles.slider_Frame,
  %set(handles.slider_Frame,'Value',(handles.f-1)/(handles.nframes-1));
  set(handles.slider_Frame,'Value',handles.f);
end
update_frame_f = [];

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

% v = get(hObject,'Value');
% handles.f = round(1 + v * (handles.nframes - 1));
% handles = update_frame(handles,hObject);
% guidata(hObject,handles);

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

global ISPLAYING;

if ~ISPLAYING,
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
handles = update_frame(handles,hObject);
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

function handles = open_fmf(handles,filename,varargin)

global ISPLAYING;

if ~isempty(ISPLAYING) && ISPLAYING,
  stop(handles.figure1);
  handles = guidata(handles.figure1);
end

handles.filterspec = {  '*.ufmf','MicroFlyMovieFormat (*.ufmf)'; ...
  '*.fmf','FlyMovieFormat (*.fmf)'; ...
  '*.sbfmf','StaticBackgroundFMF (*.sbfmf)'; ...
  '*.avi','AVI (*.avi)'
  '*.mp4','MP4 (*.mp4)'
  '*.mov','MOV (*.mov)'
  '*.mmf','MMF (*.mmf)'
  '*.mjpg','MJPG (*.mjpg)'
  '*.*','*.*'};

if isfield(handles,'fileext'),
  % default ext is last chosen
  i = find(strcmpi(['*',handles.fileext],handles.filterspec(:,1)),1);
  n = size(handles.filterspec,1);
  if ~isempty(i),
    handles.filterspec = handles.filterspec([i,1:i-1,i+1:n],:);
  end
end

if nargin < 2,
[filename, pathname] = uigetfile(handles.filterspec, 'Choose FMF video to play',handles.filename);
if ~ischar(filename),
  return;
end
else
  pathname = '';
end
handles.filename = fullfile(pathname,filename);
[handles.filedir,handles.filenamebase,handles.fileext] = fileparts(handles.filename);
if strcmp(handles.fileext,'.mjpg'),
  
  res = questdlg('Is this an indexed MJPG file? If so, you will be prompted to select the corresponding index file location.');
  if strcmpi(res,'Yes'),
    
    defaultindexfile = fullfile(handles.filedir,'index.txt');
    if ~exist(defaultindexfile,'file'),
      defaultindexfile = handles.filedir;
    end
    [indexfilename,indexpath] = uigetfile('*.txt','Choose MJPG index file',defaultindexfile);
    if ~ischar(indexfilename),
      return;
    end
    handles.indexfilename = fullfile(indexpath,indexfilename);
    varargin = [varargin,{'indexfilename',handles.indexfilename}];
  end
  
end

if isfield(handles,'fid') && ~isempty(fopen(handles.fid)) && handles.fid > 1,
  fclose(handles.fid);
end
if isfield(handles,'himage') && ishandle(handles.himage),
  delete(handles.himage);
end

try
  [handles.readframe,handles.nframes,handles.fid,handles.headerinfo] = ...
    get_readframe_fcn(handles.filename,varargin{:});
catch ME
  s = sprintf('Could not read video %s.',filename);
  uiwait(errordlg(s,'Error opening video'));
  rethrow(ME);
end

% set slider steps
% this seems to be the limit to slider step resolution
handles.stepsize = ceil(handles.nframes/handles.MAXSLIDERRES);
step1 = handles.stepsize/(handles.nframes-1);
sliderstep = [step1,min(1,100*step1)];
set(handles.slider_Frame,'Value',0,'SliderStep',sliderstep,'Min',1,'Max',handles.nframes);

% % set slider steps
% sliderstep = [1/(handles.nframes-1),min(1,100/(handles.nframes-1))];
% set(handles.slider_Frame,'Value',0,'SliderStep',sliderstep);

% show first image
handles.f = 1;
[handles.im,handles.timestamp] = handles.readframe(handles.f);
handles.minv = 0;
if isa(handles.im,'uint8'),
  handles.maxv = 255;
elseif isa(handles.im,'uint16'),
  handles.maxv = 2^16-1;
elseif isa(handles.im,'double'),
  if max(handles.im(:)) > 1,
    handles.maxv = 255;
  else
    handles.maxv = 1;
  end
end

handles.lastfread = handles.f;
if size(handles.im,3) == 1,
  handles.himage = imagesc(handles.im,'Parent',handles.axes_Video,[handles.minv,handles.maxv]);
else
  handles.himage = image(im2uint8(handles.im),'Parent',handles.axes_Video);
end
colormap(handles.axes_Video,'gray');
axis(handles.axes_Video,'image','off');
handles = update_frame(handles);

ISPLAYING = false;

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

if isfield(handles,'fid') && ~isempty(fopen(handles.fid)) && handles.fid > 1,
  fclose(handles.fid);
end

savefns = {'filename','CompressionSettings'};
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

woff = newfigpos(3) - handles.figpos(3);
hoff = newfigpos(4) - handles.figpos(4);

sliderpos(3) = sliderpos(3) + woff;
editpos(1) = editpos(1) + woff;
pushbuttonpos(1) = pushbuttonpos(1) + woff;

axespos(3) = axespos(3) + woff;
axespos(4) = axespos(4) + hoff;

set(handles.slider_Frame,'Position',sliderpos);
set(handles.edit_Frame,'Position',editpos);
set(handles.pushbutton_PlayStop,'Position',pushbuttonpos);
set(handles.axes_Video,'Position',axespos);

handles.figpos = newfigpos;
guidata(hObject,handles);


% --------------------------------------------------------------------
function menu_Edit_Preferences_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Edit_Preferences (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompts = {};
defAns = {};
prompts{end+1} = 'Max FPS: ';
defAns{end+1} = num2str(handles.MaxFPS);
prompts{end+1} = 'Min intensity: ';
defAns{end+1} = num2str(handles.minv);
prompts{end+1} = 'Max intensity: ';
defAns{end+1} = num2str(handles.maxv);
while true,
  answer = inputdlg(prompts,'playfmf Preferences',1,defAns,'on');
  msgs = {};
  if isempty(answer),
    return;
  end
  v = str2double(answer{1});
  iserror = false;
  if isnan(v),
    iserror = true;
    msgs{end+1} = 'Max FPS must be a number';
  elseif v < 0,
    iserror = true;
    msgs{end+1} = 'Max FPS must be >= 0';
  else
    defAns{1} = num2str(v);
    handles.MaxFPS = v;
    handles.MinSPF = 1 / handles.MaxFPS;
  end
  
  minv = str2double(answer{2});
  if isnan(minv),
    iserror = true;
    msgs{end+1} = 'Min intensity must be a number';
  elseif v < 0,
    iserror = true;
    msgs{end+1} = 'Min intensity must be >= 0';
  end
  
  maxv = str2double(answer{3});
  if isnan(maxv),
    iserror = true;
    msgs{end+1} = 'Max intensity must be a number';
  elseif maxv < 0,
    iserror = true;
    msgs{end+1} = 'Max intensity must be >= 0';
  end
  
  if ~iserror,
    if minv > maxv,
      iserror = true;
      msgs{end+1} = 'Max intensity must be > min intensity';
    else  
      defAns{2} = num2str(minv);
      defAns{3} = num2str(maxv);
      handles.minv = minv;
      handles.maxv = maxv;
    end
  end
  
  if iserror,
    uiwait(warndlg('Illegal values entered. Max FPS, min and max intensity must be numbers','Bad Preferences'));
  else
    break;
  end
end

set(handles.axes_Video,'CLim',[handles.minv,handles.maxv]);

guidata(hObject,handles);


% --------------------------------------------------------------------
function menu_Help_About_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Help_About (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

s = {};
s{end+1} = 'PlayFMF';
s{end+1} = '';
s{end+1} = 'Kristin Branson';
s{end+1} = 'bransonk@janelia.hhmi.org';
s{end+1} = '';
s{end+1} = 'This is a GUI for playing FMF, SBFMF, UFMF, and AVI videos. The maximum frame rate can be set through the "Preferences..." menu. Set to <= 0 for no maximum frame rate. Control the frame shown with the slider or editable text box. The Play/Stop button does what you think it does.';
msgbox(s,'About PlayFMF','help','Replace');

% --------------------------------------------------------------------
function menu_Edit_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_Edit_CompressionSettings_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Edit_CompressionSettings (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CompressionSettings = SaveFMFSettings(handles.CompressionSettings);
fns = fieldnames(CompressionSettings);
for i = 1:length(fns),
  fn = fns{i};
  handles.CompressionSettings.(fn) = CompressionSettings.(fn);
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function menu_File_Compress_Callback(hObject, eventdata, handles)
% hObject    handle to menu_File_Compress (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.CompressionSettings.StartFrame > handles.nframes,
  uiwait(errordlg(sprintf('StartFrame set to %d > NFrames = %d. Please fix Compression Settings',...
    handles.CompressionSettings.StartFrame,handles.nframes)));
  return;
end

[pathstr,filestr,ext] = fileparts(handles.filename);
handles.aviname = fullfile(pathstr,[filestr,'.avi']);

[filename,pathname] = uiputfile('*.avi','Save to AVI file',handles.aviname);
if ~ischar(filename),
  return;
end
handles.aviname = fullfile(pathname,filename);

ncolors = size(handles.im,3);
isindexed = ismember(handles.CompressionSettings.Compression,{'MSVC','RLE'}) || ...
  (strcmp(handles.CompressionSettings.Compression,'None') && ...
  ncolors == 1);
if isindexed,
  if ncolors == 1,
    cmap = repmat(linspace(0,1,256)',[1,3]);
  else
    [tmp,cmap] = rgb2ind(handles.im,256,'nodither');
  end
end

params = {'compression',handles.CompressionSettings.Compression,...
  'fps',handles.CompressionSettings.OutputFPS};

if isindexed,
  params(end+1:end+2) = {'colormap',cmap};
end
if ~strcmp(handles.CompressionSettings.Compression,'None'),
  params(end+1:end+2) = {'quality',handles.CompressionSettings.Quality};
end
handles.aviobj = avifile(handles.aviname,params{:});

endframe = min(handles.nframes,handles.CompressionSettings.EndFrame);
nframescompress = endframe - handles.CompressionSettings.StartFrame + 1;

i = 0;
s = sprintf('Compressing frame %d / %d',i,nframescompress);
hwaitbar = waitbar(0,s,'CreateCancelBtn',...
  'setappdata(gcbf,''canceling'',1)');
setappdata(hwaitbar,'canceling',0);
for t = handles.CompressionSettings.StartFrame:endframe,
  if getappdata(hwaitbar,'canceling')
    break
  end
  im = uint8(handles.readframe(t));
  if isindexed,
    if ncolors == 1,
      handles.aviobj = addframe(handles.aviobj,im);
    else
      handles.aviobj = addframe(handles.aviobj,rgb2ind(im,cmap,'nodither'));
    end
  else
    if ncolors == 1,
      handles.aviobj = addframe(handles.aviobj,repmat(im,[1,1,3]));
    else
      handles.aviobj = addframe(handles.aviobj,im);
    end
  end
  i = i + 1;
  if mod(i,50) == 0,
    s = sprintf('Compressing frame %d / %d',i,nframescompress);
    waitbar(i/nframescompress,hwaitbar,s);
  end
end
handles.aviobj = close(handles.aviobj);
if ishandle(hwaitbar),
  delete(hwaitbar);
end
msgbox(sprintf('Successfully compressed %d / %d of frames in the interval [%d,%d].',i,nframescompress,handles.CompressionSettings.StartFrame,endframe),'Compression Complete','modal');

% --------------------------------------------------------------------
function menu_Help_Callback(hObject, eventdata, handles)
% hObject    handle to menu_Help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
