function varargout = CrossValidResults(varargin)
% CROSSVALIDRESULTS MATLAB code for CrossValidResults.fig
%      CROSSVALIDRESULTS, by itself, creates a new CROSSVALIDRESULTS or raises the existing
%      singleton*.
%
%      H = CROSSVALIDRESULTS returns the handle to a new CROSSVALIDRESULTS or the handle to
%      the existing singleton*.
%
%      CROSSVALIDRESULTS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CROSSVALIDRESULTS.M with the given input arguments.
%
%      CROSSVALIDRESULTS('Property','Value',...) creates a new CROSSVALIDRESULTS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CrossValidResults_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CrossValidResults_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CrossValidResults

% Last Modified by GUIDE v2.5 18-Jul-2017 09:55:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CrossValidResults_OpeningFcn, ...
                   'gui_OutputFcn',  @CrossValidResults_OutputFcn, ...
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

% hFig = CrossValidateResults(labelerObj,resultsStr,tblXVRes);
function CrossValidResults_OpeningFcn(hObject, eventdata, handles, varargin)

labelerObj = varargin{1};
resultsStr = varargin{2};
tblXVRes = varargin{3};

handles.etResults.String = resultsStr;
handles.labelerObj = labelerObj;
handles.tblXVRes = tblXVRes;

handles.output = hObject;
guidata(hObject, handles);

% UIWAIT makes CrossValidResults wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function varargout = CrossValidResults_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function pbExport_Callback(hObject, eventdata, handles)
assignin('base','aptXVRes',handles.tblXVRes);
msgbox('Wrote variable ''aptXVRes'' in base workspace.','Export results');

function pbViewTrackingResults_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
tObj = lObj.tracker;
tObj.setAllTrackResTable(handles.tblXVRes,1:lObj.nLabelPoints);
str = 'Set tracking results in APT.';
msgbox(str,'View results');
