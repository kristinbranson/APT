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

% Last Modified by GUIDE v2.5 05-Jul-2016 09:46:13

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
  mprops,'PostSet',@(src,evt)lclUpdateTable(hObject));

% 20151218: now using JTable; uitable in .fig just used for positioning
tblOrig = handles.tblMovies;
tblOrig.Visible = 'off';
tblNew = uiextras.jTable.Table(...
  'parent',tblOrig.Parent,...
  'Position',tblOrig.Position,...
  'SelectionMode','discontiguous',...
  'ColumnName',{'Movie' 'Has Labels'},...
  'ColumnPreferredWidth',[600 250],...
  'Editable','off');
tblNew.MouseClickedCallback = @(src,evt)lclTblClicked(src,evt,tblNew,lObj);
handles.tblMoviesOrig = handles.tblMovies;
handles.tblMovies = tblNew;
handles.cbkGetSelectedMovies = @()cbkGetSelectedMovies(tblNew);

guidata(hObject,handles);
centerfig(handles.figure1,handles.labeler.gdata.figure);
lclUpdateTable(hObject);

function lclTblClicked(src,evt,tbl,lObj)
persistent chk
PAUSE_DURATION_CHECK = 0.25;
if isempty(chk)
  chk = 1;
  pause(PAUSE_DURATION_CHECK); %Add a delay to distinguish single click from a double click
  if chk==1
    % single-click; no-op    
    chk = [];
  end
else
  chk = [];
  lclSwitchMoviesBasedOnSelection(tbl,lObj);
end

function varargout = MovieManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function lclUpdateTable(hObj)
handles = guidata(hObj);

lObj = handles.labeler;
tbl = handles.tblMovies;
movs = lObj.movieFilesAll(:,1);
movsHaveLbls = lObj.movieFilesAllHaveLbls;
iMov = lObj.currMovie;

if numel(movs)~=numel(movsHaveLbls)
  % intermediate state, take no action
  return;
end
dat = [movs num2cell(movsHaveLbls)];

if ~isequal(dat,tbl.Data)
  tbl.Data = dat;
end
if iMov>0
  tbl.SelectedRows = iMov;
else
  tbl.SelectedRows = [];
end

function iMovs = cbkGetSelectedMovies(tbl)
% AL20160630: IMPORTANT: currently CANNOT sort table by columns
selRow = tbl.SelectedRows;
iMovs = sort(selRow);

function pbAdd_Callback(hObject, eventdata, handles) %#ok<*DEFNU,*INUSD>
[tfsucc,movfile,trxfile] = promptGetMovTrxFiles(true);
if ~tfsucc
  return;
end
handles.labeler.movieAdd(movfile,trxfile);

function pbRm_Callback(hObject, eventdata, handles)
tbl = handles.tblMovies;
selRow = tbl.SelectedRows;
selRow = sort(selRow);
n = numel(selRow);
lObj = handles.labeler;
for i = n:-1:1
  row = selRow(i);
  tfSucc = lObj.movieRm(row);
  if ~tfSucc
    % user stopped/canceled
    break;
  end
end

function pbSwitch_Callback(~,~,handles)
lclSwitchMoviesBasedOnSelection(handles.tblMovies,handles.labeler);

function lclSwitchMoviesBasedOnSelection(tbl,lObj)
selRow = tbl.SelectedRows;
if numel(selRow)>1
  warning('MovieManager:sel','Multiple movies selected; switching to first selection.');
  selRow = selRow(1);
end  
if ~isempty(selRow)
  lObj.movieSet(selRow);
end

function pbNextUnlabeled_Callback(hObject, eventdata, handles)
lObj = handles.labeler;
iMov = find(~lObj.movieFilesAllHaveLbls,1);
if isempty(iMov)
  msgBox('All movies are labeled!');
else
  lObj.movieSet(iMov);
end

function menu_file_add_movies_from_text_file_Callback(hObject, eventdata, handles)
lastTxtFile = RC.getprop('lastMovieBatchFile');
if ~isempty(lastTxtFile)
  [~,~,ext] = fileparts(lastTxtFile);
  ext = ['*' ext];
  file0 = lastTxtFile;
else
  ext = '*.txt';
  file0 = pwd;
end
[fname,pname] = uigetfile(ext,'Select movie batch file',file0);
if isequal(fname,0)
  return;
end
fname = fullfile(pname,fname);
handles.labeler.movieAddBatchFile(fname);
RC.saveprop('lastMovieBatchFile',fname);
