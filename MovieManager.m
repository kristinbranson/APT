function varargout = MovieManager(varargin)
% MOVIEMANAGER MATLAB code for MovieManager.fig
%      MOVIEMANAGER, by itself, creates a new MOVIEMANAGER or raises the existing
%      singleton*.
%
%      H = MOVIEMANAGER returns the handle to a new MOVIEMANAGER or the handle to
%      the existing singleton*.
%
%      MOVIEMANAGER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MOVIEMANAGER.M with the given input arguments.
%
%      MOVIEMANAGER('Property','Value',...) creates a new MOVIEMANAGER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MovieManager_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MovieManager_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MovieManager

% Last Modified by GUIDE v2.5 23-Sep-2015 10:40:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
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

lObj = varargin{1};
handles.labeler = lObj;
handles.output = hObject;
PROPS = {'movieFilesAll' 'trxFilesAll'};
mcls = metaclass(lObj);
mprops = mcls.PropertyList;
mprops = mprops(ismember({mprops.Name}',PROPS));
handles.listener = event.proplistener(lObj,...
  mprops,'PostSet',@(src,evt)lclUpdateTable(handles));
guidata(hObject,handles);

lclUpdateTable(handles);

function varargout = MovieManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function lclUpdateTable(handles)
lObj = handles.labeler;
movs = lObj.movieFilesAll;
trxs = lObj.trxFilesAll;
if numel(movs)~=size(trxs)
  % intermediate state, take no action
  return;
end
dat = [movs cellfun(@(x)~isempty(x),trxs,'uni',0)];
set(handles.tblMovies,'Data',dat);

function pbAdd_Callback(hObject, eventdata, handles) %#ok<*DEFNU,*INUSD>
lastmov = RC.getprop('lbl_lastmovie');
[movfile,movpath] = uigetfile('*.*','Select video',lastmov);
if ~ischar(movfile)
  return;
end
movfile = fullfile(movpath,movfile);

[trxfile,trxpath] = uigetfile('*.mat','Select trx file',movpath);
if ~ischar(trxfile)
  % user canceled; interpret this as "there is no trx file"
  trxfile = [];
else
  trxfile = fullfile(trxpath,trxfile);
end

handles.labeler.movieAdd(movfile,trxfile);

function pbRm_Callback(hObject, eventdata, handles)
if isfield(handles,'selectedRow')
  row = handles.selectedRow;
  handles.labeler.movieRm(row);
end

function tblMovies_CellSelectionCallback(hObject, eventdata, handles)
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  handles.selectedRow = row;
  guidata(hObject,handles);
  
  switch handles.figure1.SelectionType
    case 'open' % double-click
      handles.labeler.movieSet(row);
  end
end
