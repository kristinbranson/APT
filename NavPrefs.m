function varargout = NavPrefs(varargin)
% NAVPREFS MATLAB code for NavPrefs.fig
%      NAVPREFS, by itself, creates a new NAVPREFS or raises the existing
%      singleton*.
%
%      H = NAVPREFS returns the handle to a new NAVPREFS or the handle to
%      the existing singleton*.
%
%      NAVPREFS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NAVPREFS.M with the given input arguments.
%
%      NAVPREFS('Property','Value',...) creates a new NAVPREFS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before NavPrefs_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to NavPrefs_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help NavPrefs

% Last Modified by GUIDE v2.5 06-Nov-2017 10:53:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @NavPrefs_OpeningFcn, ...
                   'gui_OutputFcn',  @NavPrefs_OutputFcn, ...
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

function NavPrefs_OpeningFcn(hObject, eventdata, handles, varargin)
% NavPrefs(lObj)

hObject.Visible = 'off';

lObj = varargin{1};
handles.lObj = lObj;
handles.etFrameSkip.String = num2str(lObj.movieFrameStepBig);
shiftArrowModes = enumeration('ShiftArrowMovieNavMode');
pum = handles.pumShiftArrow;
pum.String = arrayfun(@(x)x.prettyStr,shiftArrowModes,'uni',0);
pum.Value = find(lObj.movieShiftArrowNavMode==shiftArrowModes);
pum.UserData = shiftArrowModes;
guidata(hObject, handles);

centerOnParentFigure(hObject,lObj.gdata.figure);
hObject.Visible = 'on';

uiwait(handles.figure1);

function varargout = NavPrefs_OutputFcn(hObject, eventdata, handles) 
delete(hObject);

function etFrameSkip_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if isnan(val)
  hObject.String = num2str(handles.lObj.movieFrameStepBig);
end

function pbApply_Callback(hObject, eventdata, handles)
lObj = handles.lObj;
lObj.movieFrameStepBig = str2double(handles.etFrameSkip.String);
pum = handles.pumShiftArrow;
sam = pum.UserData(pum.Value);
lObj.movieShiftArrowNavMode = sam;
close(handles.figure1);

function pbCancel_Callback(hObject, eventdata, handles)
delete(handles.figure1);

function figure1_CloseRequestFcn(hObject, eventdata, handles)
if strcmp(get(hObject,'waitstatus'),'waiting')
  uiresume(hObject);
else
  delete(hObject);
end
