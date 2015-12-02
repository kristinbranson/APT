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

% Last Modified by GUIDE v2.5 02-Dec-2015 11:18:52

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
PROPS = {'movieFilesAll' 'movieFilesAllHaveLbls' 'currMovie'};
mcls = metaclass(lObj);
mprops = mcls.PropertyList;
mprops = mprops(ismember({mprops.Name}',PROPS));
handles.listener = event.proplistener(lObj,...
  mprops,'PostSet',@(src,evt)lclUpdateTable(handles));
handles.selectedRow = [];
guidata(hObject,handles);

centerfig(handles.figure1,handles.labeler.gdata.figure);

lclUpdateTable(handles);

function varargout = MovieManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function lclUpdateTable(handles)
lObj = handles.labeler;
movs = lObj.movieFilesAll;
movsHaveLbls = lObj.movieFilesAllHaveLbls;
iMov = lObj.currMovie;

if numel(movs)~=numel(movsHaveLbls)
  % intermediate state, take no action
  return;
end
dat = [movs num2cell(movsHaveLbls)];

% estimate width column1
col1MaxSz = max(cellfun(@numel,dat(:,1)));
CHARS2PIXFAC = 8;
col1Pixels = col1MaxSz*CHARS2PIXFAC;

% highlight current movie
if ~isempty(iMov) && iMov>0
  dat{iMov,1} = ['<html><font color=#0000FF>' dat{iMov,1} '</font></html>'];    
end
    
handles.tblMovies.Data = dat;
handles.tblMovies.ColumnWidth{1} = col1Pixels;

function pbAdd_Callback(hObject, eventdata, handles) %#ok<*DEFNU,*INUSD>
[tfsucc,movfile,trxfile] = promptGetMovTrxFiles();
if ~tfsucc
  return;
end
handles.labeler.movieAdd(movfile,trxfile);

function pbRm_Callback(hObject, eventdata, handles)
if isfield(handles,'selectedRow')
  row = handles.selectedRow;
  handles.labeler.movieRm(row);
end

function tblMovies_CellSelectionCallback(hObject, eventdata, handles)
row = eventdata.Indices;
if isempty(row)
  handles.selectedRow = [];
else
  row = row(1);
  handles.selectedRow = row;  
%   switch handles.figure1.SelectionType
%     case 'open' % double-click
%       handles.labeler.movieSet(row);
%   end
end
guidata(hObject,handles);

function pbSwitch_Callback(hObject, eventdata, handles)
if ~isempty(handles.selectedRow)
  handles.labeler.movieSet(handles.selectedRow);
end

function pbNextUnlabeled_Callback(hObject, eventdata, handles)
lObj = handles.labeler;
iMov = find(~lObj.movieFilesAllHaveLbls,1);
if isempty(iMov)
  msgBox('All movies are labeled!');
else
  lObj.movieSet(iMov);
end
