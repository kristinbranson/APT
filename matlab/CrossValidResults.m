function varargout = CrossValidResults(varargin)
% Display/UI for cross-validation results

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

set(hObject,'MenuBar','None');
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
assignin('base','aptXVresults',handles.tblXVRes);
msgbox('Wrote variable ''aptXVresults'' in base workspace.','Export results');

function pbViewTrackingResults_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
tObj = lObj.tracker;
tObj.setAllTrackResTable(handles.tblXVRes,1:lObj.nLabelPoints);
str = 'Set tracking results in APT.';
msgbox(str,'View results');
