function varargout = ProjectSetup(varargin)
% New project creation

% Last Modified by GUIDE v2.5 03-Oct-2020 10:16:05

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ProjectSetup_OpeningFcn, ...
                   'gui_OutputFcn',  @ProjectSetup_OutputFcn, ...
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

% PROJECT CONFIGURATION/SETUP NOTES
% 20160815
% 
% A _project configuration_ is the stuff in pref.default.yaml. It is 
% comprised of:
%
% 1. Core per-project info: number/names of views, number/names of points.
% 2. More cosmetic per-project info: how lines/markers look, frame 
% increments in various situations, etc.
% 3. Tracker specification and config.
% 4. More application-level preferences, like keyboard shortcut choices. 
% Strictly speaking these could be separated out but for now just lump them 
% in. (Plus, who knows what turns out to be useful when configurable
% per-project.)
%
% If you have a project configuration, you can create/init a new, "blank"
% project.
%
% A _project_ is comprised of:
% 1. A project configuration
% 2. Moviefilenames, trxfilenames, filename macros, file metadata
% 3. (Optional) view calibration info, view calibration file
% 4. Label data (labels, timestamps, tags, flags)
% 5. UI state: current movie/frame/target, labelMode, image colormap etc
%
% The first time you need a project configuration, the stuff in
% pref.default.yaml is used. In all subsequent instances, you start off
% with your most recent configuration.
%
% Once a project is created, much/most of the configuration info is
% mutable. For instance, you can rename points, change labelModes, change
% trackers, change plot cosmetics, etc. A few things are currently 
% immutable, such as the number of labeling points. When a project is
% saved, the saved configuration is generated from the
% Labeler-state-at-that-time, which may differ from the project's initial
% configuration. 
%
% Later, if application-wide preferences are desired/added, these can
% override parts of the project configuration as appropriate.
%
% Labeler actions:
% - Create a new/blank project from a configuration.
% - Get the current configuration.
% - Load an existing project (which contains a configuration).
% - Save a project -- i) get current config, and ii) create project.

% --- Executes just before ProjectSetup is made visible.
%
% Modal dialog. Generates project configuration struct
%
% cfg = ProjectSetup(); 
% cfg = ProjectSetup(hParentFig); % centered on hParentFig
function ProjectSetup_OpeningFcn(hObject, eventdata, handles, varargin)

h1 = findall(handles.figure1,'-property','Units');
set(h1,'Units','Normalized');
set(handles.figure1,'MenuBar','None');

if numel(varargin)>=1
  hParentFig = varargin{1};
  if ~ishandle(hParentFig)
    error('ProjectSetup:arg','Expected argument to be a figure handle.');
  end
  centerOnParentFigure(hObject,hParentFig);
end  
  
handles.output = [];

% init PUMs that depend only on codebase
% lms = enumeration('LabelMode');
% tfnone = lms==LabelMode.NONE;
% lms(tfnone,:) = [];
% lmStrs = arrayfun(@(x)x.prettyString,lms,'uni',0);
% handles.pumLabelingMode.String = lmStrs;
% handles.pumLabelingMode.UserData = lms;
% trackers = LabelTracker.findAllSubclasses;
% trackers = [{'None'};trackers];
% handles.pumTracking.String = trackers;

handles.propsPane = [];

% init ui state
cfg = Labeler.cfgGetLastProjectConfigNoView;
handles = setCurrentConfig(handles,cfg);
handles.propsPane.Position(4) = handles.propsPane.Position(3); % by default table is slightly bigger than panel for some reason
handles = advModeCollapse(handles);

guidata(hObject, handles);

% UIWAIT makes ProjectSetup wait for user response (see UIRESUME)
uiwait(handles.figure1);

function varargout = ProjectSetup_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;
delete(handles.figure1);

function cfg = genCurrentConfig(handles)
% Generate config from the current UI state

ad = getappdata(handles.figure1);
cfg = ad.mirror;

assert(numel(fieldnames(cfg.ViewNames))==handles.nViews);
assert(numel(fieldnames(cfg.LabelPointNames))==handles.nPoints);
cfg.NumViews = handles.nViews;
cfg.NumLabelPoints = handles.nPoints;
cfg.ViewNames = struct2cell(cfg.ViewNames);
cfg.LabelPointNames = struct2cell(cfg.LabelPointNames);
cfg.Trx.HasTrx = handles.cbHasTrx.Value;
cfg.MultiAnimal = handles.cbMA.Value;
isMA = cfg.MultiAnimal && ~cfg.Trx.HasTrx;
if isMA
  cfg.LabelMode = LabelMode.MULTIANIMAL;
else
  cfg.LabelMode = LabelMode.SEQUENTIAL;
end
% pumLM = handles.pumLabelingMode;
% lmVal = pumLM.Value;
% cfg.LabelMode = char(pumLM.UserData(lmVal));
% pumTrk = handles.pumTracking;
% tracker = pumTrk.String{pumTrk.Value};
% cfg.Track.Enable = ~strcmpi(tracker,'none');
cfg.Track.Enable = true;
% cfg.Track.Type = tracker;
% propertiesGUI treats props with empty vals as strings even if they are
% subsequently filled with numbers
FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
for i=1:numel(cfg.View)  
  cfg.View(i) = structLeavesStr2Double(cfg.View(i),FIELDS2DOUBLIFY);
end

function handles = setCurrentConfig(handles,cfg)
% Set given config on controls

% we store these two props on handles in order to be able to revert; 
% data/model is split between i) primary UIcontrols and ii) adv panel 
handles.nViews = cfg.NumViews; 
handles.nPoints = cfg.NumLabelPoints;
set(handles.etNumberOfViews,'string',num2str(handles.nViews));
set(handles.etNumberOfPoints,'string',num2str(handles.nPoints));
set(handles.cbHasTrx,'Value',cfg.Trx.HasTrx);
set(handles.cbMA,'Value',cfg.MultiAnimal);


% pumLM = handles.pumLabelingMode;
% [tf,val] = ismember(cfg.LabelMode,arrayfun(@char,pumLM.UserData,'uni',0));
% if ~tf
%   % should never happen 
%   val = 1; % NONE
% end
% pumLM.Value = val;

% pumTrk = handles.pumTracking;
% if cfg.Track.Enable 
%   [tf,val] = ismember(cfg.Track.Type,pumTrk.String);
%   if ~tf
%     % unexpected but maybe not impossible due to path
%     val = 1; % None
%   end
% else
%   val = 1;
% end
% pumTrk.Value = val;

sMirror = Labeler.cfg2mirror(cfg);
handles = advTableRefresh(handles,sMirror);

function handles = advTableRefresh(handles,sMirror)
tfRefresh = exist('sMirror','var')==0;
if tfRefresh
  ad = getappdata(handles.figure1);
  sMirror = ad.mirror;
end
sMirror = Labeler.hlpAugmentOrTruncNameField(sMirror,'ViewNames','view',handles.nViews);
sMirror = Labeler.hlpAugmentOrTruncNameField(sMirror,'LabelPointNames','point',handles.nPoints);
sMirror = Labeler.hlpAugmentOrTruncStructField(sMirror,'View',handles.nViews);
if ~isempty(handles.propsPane) && ishandle(handles.propsPane)
  delete(handles.propsPane);
  handles.propsPane = [];
end
  
handles.propsPane = propertiesGUI(handles.pnlAdvanced,sMirror);

function handles = advModeExpand(handles)
h1 = findall(handles.figure1,'-property','Units');
set(h1,'Units','pixels');
posRight = handles.landmarkRight.Position;
posRight = posRight(1)+posRight(3);
pos = handles.figure1.Position;
pos(3) = posRight;
handles.figure1.Position = pos;
set(h1,'Units','normalized');
handles.advancedOn = true;
handles.pbAdvanced.String = '< Basic';

function handles = advModeCollapse(handles)
h1 = findall(handles.figure1,'-property','Units');
set(h1,'Units','pixels');
posMid = handles.landmarkMid.Position;
posMid = posMid(1)+posMid(3)/2;
pos = handles.figure1.Position;
pos(3) = posMid;
handles.figure1.Position = pos;
set(h1,'Units','normalized');
handles.advancedOn = false;
handles.pbAdvanced.String = 'Advanced >';

function handles = advModeToggle(handles)
if handles.advancedOn
  handles = advModeCollapse(handles);
else
  handles = advModeExpand(handles);
end

function etProjectName_Callback(hObject, eventdata, handles)
name = hObject.String;
if ~all(isstrprop(name,'alphanum')) 
  % This unfortunately invalidates _ also. Checking for it seems more work
  % than worth. MK 20220913
  warndlg('Name should have only alphanumberic characters');
  hObject.String = '';
end

function etNumberOfPoints_Callback(hObject, eventdata, handles)
%fprintf('etNOP enter');
val = str2double(hObject.String);
if floor(val)==val && val>=1
  handles.nPoints = val;
else
  hObject.String = handles.nPoints;
end
handles = advTableRefresh(handles);
guidata(hObject,handles);
%fprintf('etNOP end');
function etNumberOfViews_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if floor(val)==val && val>=1
  handles.nViews = val;
else
  hObject.String = handles.nViews;
end
switch handles.nViews
  case 1
    handles.cbHasTrx.Enable = 'on';
    handles.cbMA.Enable = 'on';    
  otherwise
    handles.cbHasTrx.Value = 0;
    handles.cbMA.Value = 0;
    handles.cbHasTrx.Enable = 'off';
    handles.cbMA.Enable = 'off';
end
handles = advTableRefresh(handles);
guidata(hObject,handles);
% function pumLabelingMode_Callback(hObject, eventdata, handles)
% function pumTracking_Callback(hObject, eventdata, handles)
function pbCreateProject_Callback(hObject, eventdata, handles)
%fprintf('pbCreate start');
cfg = genCurrentConfig(handles);
cfg.ProjectName = handles.etProjectName.String;
handles.output = cfg;
guidata(handles.figure1,handles);
close(handles.figure1);
%fprintf('pbCreate end');
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
lbl = loadLbl(fullfile(pth,fname));
lbl = Labeler.lblModernize(lbl);
cfg = lbl.cfg;
handles = setCurrentConfig(handles,cfg);
guidata(handles.figure1,handles);

function figure1_CloseRequestFcn(hObject, eventdata, handles)
if isequal(get(hObject,'waitstatus'),'waiting')
% The GUI is still in UIWAIT, us UIRESUME
  uiresume(hObject);
else  
  delete(hObject);
end

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

function cbHasTrx_Callback(hObject, eventdata, handles)
% none
function cbMA_Callback(hObject, eventdata, handles)
% none
