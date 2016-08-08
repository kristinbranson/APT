function varargout = ProjectSetup(varargin)
% PROJECTSETUP MATLAB code for ProjectSetup.fig
%      PROJECTSETUP, by itself, creates a new PROJECTSETUP or raises the existing
%      singleton*.
%
%      H = PROJECTSETUP returns the handle to a new PROJECTSETUP or the handle to
%      the existing singleton*.
%
%      PROJECTSETUP('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROJECTSETUP.M with the given input arguments.
%
%      PROJECTSETUP('Property','Value',...) creates a new PROJECTSETUP or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ProjectSetup_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ProjectSetup_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ProjectSetup

% Last Modified by GUIDE v2.5 08-Aug-2016 15:43:05

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


% --- Executes just before ProjectSetup is made visible.
function ProjectSetup_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

% init pums
lms = enumeration('LabelMode');
lms = arrayfun(@(x)x.prettyString,lms,'uni',0);
handles.pumLabelMode.String = lms;
trackers = LabelTracker.findAllSubclasses;
trackers = [{'None'};trackers];
handles.pumTracking.String = trackers;

% setup tree
sBase = ReadYaml('pref.default.yaml');
sLast = RC.getprop('lastProjectConfig');
if isempty(sLast)
  s = sBase;
else
  s = structoverlay(sBase,sLast);
end
% remove props that have explicit uicontrols
s = rmfield(s,{'NumViews' 'NumLabelPoints' 'LabelMode'}); 
s.Track = rmfield(s.Track,{'Enable' 'Type'});
handles.propsPane = propertiesGUI(handles.pnlAdvanced,s);

handles.advancedOn = false;

guidata(hObject, handles);

% UIWAIT makes ProjectSetup wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function varargout = ProjectSetup_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function advModeExpand(handles)
function advModeCollapse(handles)

function etProjectName_Callback(hObject, eventdata, handles)
function etNumberOfPoints_Callback(hObject, eventdata, handles)
function etNumberOfViews_Callback(hObject, eventdata, handles)
function pumLabelingMode_Callback(hObject, eventdata, handles)
function pumTracking_Callback(hObject, eventdata, handles)
function pbCreateProject_Callback(hObject, eventdata, handles)
function pbCancel_Callback(hObject, eventdata, handles)
function pbCollapseNames_Callback(hObject, eventdata, handles)
function pbAdvanced_Callback(hObject, eventdata, handles)
