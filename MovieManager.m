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

% Last Modified by GUIDE v2.5 03-Oct-2016 13:16:20

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
%
% MovieManager is created with Visible='off'.

lObj = varargin{1};
handles.labeler = lObj;
handles.output = hObject;
hObject.Visible = 'off';
PROPS = {'movieFilesAll' 'movieFilesAllHaveLbls' 'currMovie'};
mcls = metaclass(lObj);
mprops = mcls.PropertyList;
mprops = mprops(ismember({mprops.Name}',PROPS));
handles.listener = event.proplistener(lObj,...
  mprops,'PostSet',@(src,evt)cbkUpdateTable(hObject));

% 20151218: now using JTable; uitable in .fig just used for positioning
if isa(handles.tblMovies,'matlab.ui.control.Table')
  tblOrig = handles.tblMovies;
  tblOrig.Visible = 'off';
  handles.tblMoviesOrig = tblOrig;
else
  tblOrig = handles.tblMoviesOrig;
  assert(isa(handles.tblMoviesOrig,'matlab.ui.control.Table'));
  if isvalid(handles.tblMovies)
    delete(handles.tblMovies);
    handles.tblMovies = [];
  end    
end

handles.tblMovies = MovieManagerTable.create(lObj.nview,hObject,tblOrig.Parent,...
  tblOrig.Position,@(movname)cbkSelectMovie(hObject,movname));

% For Labeler/clients to access selected movies in MovieManager
handles.cbkGetSelectedMovies = @()cbkGetSelectedMovies(hObject);

% MovieManager handles messages in two directions
% 1a. Labeler/clients can fetch current selection in Table
% 1b. Labeler prop listeners can update Table content
% 2. Table can set current movie in Labeler (based on selection/user interaction)

guidata(hObject,handles);
centerfig(handles.figure1,handles.labeler.gdata.figure);
%cbkUpdateTable(hObject);

function varargout = MovieManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function cbkUpdateTable(hMMobj)
% Update Table based on labeler movie props

handles = guidata(hMMobj);
lObj = handles.labeler;
if ~lObj.hasProject
  error('MovieManager:proj','Please open/create a project first.');
end
movs = lObj.movieFilesAll;
movsHaveLbls = lObj.movieFilesAllHaveLbls;

if size(movs,1)~=numel(movsHaveLbls)
  % intermediate state, take no action
  return;
end

tbl = handles.tblMovies;
tbl.updateMovieData(movs,movsHaveLbls);
if ~isempty(lObj.currMovie) % can occur during projload
  tbl.updateSelectedMovie(lObj.currMovie);
end

function imovs = cbkGetSelectedMovies(hMMobj)
% Get current selection in Table
handles = guidata(hMMobj);
imovs = handles.tblMovies.getSelectedMovies();

function cbkSelectMovie(hMMobj,imov)
% Set current Labeler movie to movname
handles = guidata(hMMobj);
lObj = handles.labeler;
assert(isscalar(imov));
lObj.movieSet(imov);

function pbAdd_Callback(hObject, eventdata, handles) %#ok<*DEFNU,*INUSD>
lObj = handles.labeler;
nmovieOrig = lObj.nmovies;  
if lObj.nview==1
  [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(true);
  if ~tfsucc
    return;
  end
  lObj.movieAdd(movfile,trxfile);
else
  assert(lObj.nTargets==1,'Adding trx files currently unsupported.');  
  lastmov = RC.getprop('lbl_lastmovie');
  lastmovpath = fileparts(lastmov);
  movfiles = uipickfiles(...
    'Prompt','Select movie set',...
    'FilterSpec',lastmovpath,...
    'NumFiles',lObj.nview);
  if isequal(movfiles,0)
    return;
  end
  lObj.movieSetAdd(movfiles);
end
if nmovieOrig==0 && lObj.nmovies>0
  lObj.movieSet(1);
end

function pbRm_Callback(hObject, eventdata, handles)
tbl = handles.tblMovies;
selRow = tbl.getSelectedMovies();
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
tbl = handles.tblMovies;
imov = tbl.getSelectedMovies();
if ~isempty(imov)
  imov = imov(1);
  cbkSelectMovie(handles.figure1,imov);
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
nmovieOrig = handles.labeler.nmovies;
fname = fullfile(pname,fname);
handles.labeler.movieAddBatchFile(fname);
RC.saveprop('lastMovieBatchFile',fname);

if nmovieOrig==0 && handles.labeler.nmovies>0
  handles.labeler.movieSet(1);
end

function figure1_CloseRequestFcn(hObject, eventdata, handles)
hObject.Visible = 'off';
% if isvalid(handles.listener)
%   delete(handles.listener);
%   handles.listener = [];
% end
% delete(hObject);
