function varargout = MovieManager(varargin)
% Movie table GUI

% Last Modified by GUIDE v2.5 24-Aug-2017 11:16:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MovieManager_OpeningFcn, ...
                   'gui_OutputFcn',  @MovieManager_OutputFcn, ...
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

function MovieManager_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
% MovieManager(labelerObj)
%
% MovieManager is created with Visible='off'.

mmc = varargin{1};
set(hObject,'MenuBar','None');
handles.mmController = mmc;
handles.output = hObject;
hObject.Visible = 'off';
hObject.DeleteFcn = @(s,e)delete(mmc);
guidata(hObject,handles);

function varargout = MovieManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function figure1_CloseRequestFcn(hObject, eventdata, handles)
hObject.Visible = 'off';

% function menu_help_Callback(hObject, eventdata, handles)
% str = {...
%   'This window shows all movies currently in the project.'; 
%   'To change movies, double-click a table row or use the ''Switch to Movie'' button.'};
% msgbox(str,'Help');
