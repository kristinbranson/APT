function varargout = GTManager(varargin)
% Movie table GUI

% Last Modified by GUIDE v2.5 08-Aug-2017 15:57:05

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GTManager_OpeningFcn, ...
                   'gui_OutputFcn',  @GTManager_OutputFcn, ...
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

function GTManager_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
% GTManager(labelerObj)
%
% GTManager is created with Visible='off'.

lObj = varargin{1};
handles.labeler = lObj;
handles.output = hObject;
hObject.Visible = 'off';

PROPS = {'gtSuggMFT' 'gtSuggLbled'};
mcls = metaclass(lObj);
mprops = mcls.PropertyList;
mprops = mprops(ismember({mprops.Name}',PROPS));
handles.listener{1,1} = event.proplistener(lObj,...
  mprops,'PostSet',@(src,evt)cbkUpdateTable(hObject));
%handles.listener{2,1} = addlistener(lObj,'newMovie',@(src,evt)cbkUpdateTable(hObject));

if isa(handles.tblMFT,'matlab.ui.control.Table')
  tblOrig = handles.tblMFT;
  tblOrig.Visible = 'off';
  handles.tblMoviesOrig = tblOrig;
else
  tblOrig = handles.tblMoviesOrig;
  assert(isa(handles.tblMoviesOrig,'matlab.ui.control.Table'));
  if isvalid(handles.tblMFT)
    delete(handles.tblMFT);
    handles.tblMFT = [];
  end    
end

handles.tblMFT = MovieManagerTable.create(lObj.nview,hObject,tblOrig.Parent,...
  tblOrig.Position,@(movname)cbkSelectMovie(hObject,movname));

% For Labeler/clients to access selected movies in GTManager
handles.cbkGetSelectedMovies = @()cbkGetSelectedMovies(hObject);

% GTManager handles messages in two directions
% 1a. Labeler/clients can fetch current selection in Table
% 1b. Labeler prop listeners can update Table content
% 2. Table can set current movie in Labeler (based on selection/user interaction)

guidata(hObject,handles);
centerfig(handles.figure1,handles.labeler.gdata.figure);
handles.figure1.DeleteFcn = @lclDeleteFig;

function varargout = GTManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function cbkUpdateTable(hMMobj)
% Update Table based on labeler movie props

handles = guidata(hMMobj);
lObj = handles.labeler;
if lObj.isinit
  return;
end
if ~lObj.hasProject
  error('MovieManager:proj','Please open/create a project first.');
end
movs = lObj.movieFilesAllGT;
trxs = lObj.trxFilesAllGT;
movsHaveLbls = lObj.movieFilesAllGTHaveLbls;

if ~isequal(size(movs,1),size(trxs,1),numel(movsHaveLbls))
  % intermediate state, take no action
  return;
end

tbl = handles.tblMFT;
tbl.updateMovieData(movs,trxs,movsHaveLbls);
if ~isempty(lObj.currMovie) % can occur during projload
  tbl.updateSelectedMovie(lObj.currMovie);
end

function imovs = cbkGetSelectedMovies(hMMobj)
% Get current selection in Table
handles = guidata(hMMobj);
imovs = handles.tblMFT.getSelectedMovies();

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
  if isempty(lastmov)
    lastmovpath = pwd;
  else
    lastmovpath = fileparts(lastmov);
  end
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
  lObj.movieSet(1,'isFirstMovie',true);
end

function pbComputeGT_Callback(hObject, eventdata, handles)
tbl = handles.tblMFT;
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
tbl = handles.tblMFT;
imov = tbl.getSelectedMovies();
if ~isempty(imov)
  imov = imov(1);
  cbkSelectMovie(handles.figure1,imov);
end

function pbNextUnlabeled_Callback(hObject, eventdata, handles)
lObj = handles.labeler;
iMov = find(~lObj.movieFilesAllGTHaveLbls,1);
if isempty(iMov)
  msgbox('All movies are labeled!');
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

function lclDeleteFig(src,evt)
handles = guidata(src);
listenObjs = handles.listener;
for i=1:numel(listenObjs)
  o = listenObjs{i};
  if isvalid(o)
    delete(o);
  end
end



% --- Executes when selected cell(s) is changed in tblMFT.
function tblMFT_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to tblMFT (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
