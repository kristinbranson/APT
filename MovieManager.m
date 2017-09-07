function varargout = MovieManager(varargin)
% Movie table GUI

% Last Modified by GUIDE v2.5 24-Aug-2017 11:16:51

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

mmc = varargin{1};
handles.mmController = mmc;
handles.output = hObject;
hObject.Visible = 'off';
hObject.DeleteFcn = @(s,e)delete(mmc);
guidata(hObject,handles);

function varargout = MovieManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

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

% function lclDeleteFig(src,evt)
% handles = guidata(src);
% listenObjs = handles.listener;
% for i=1:numel(listenObjs)
%   o = listenObjs{i};
%   if isvalid(o)
%     delete(o);
%   end
% end

% function menu_help_Callback(hObject, eventdata, handles)
% str = {...
%   'This window shows all movies currently in the project.'; 
%   'To change movies, double-click a table row or use the ''Switch to Movie'' button.'};
% msgbox(str,'Help');
