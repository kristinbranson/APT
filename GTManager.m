function varargout = GTManager(varargin)
% Movie table GUI

% Last Modified by GUIDE v2.5 04-Sep-2017 17:00:20

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

% GTManager<->Labeler messaging:
% 2. GTManager sets current movie/frame/target in Labeler based on 
%   treeTable interaction
% 3a. Labeler prop listeners completely refresh GTTable content (gtSuggMFTable)
% 3b. Labeler prop listeners incrementally update GTTable content (gtSuggMFTableLbled)
% 3c. Labeler prop listeners update GTTable selection, expand/collapse
%   (frame, target etc)

lObj = varargin{1};
handles.labeler = lObj;
handles.output = hObject;
hObject.Visible = 'off';

if isa(handles.tblGT,'matlab.ui.control.Table')
  tblOrig = handles.tblGT;
  tblOrig.Visible = 'off';
  handles.tblGTOrig = tblOrig;
else
  assert(false);
end

cbkNavDataRow = @(iData)cbkTreeTableDataRowNaved(hObject,iData);
% The Data table of this NTT has fields mov/frm/iTgt/hasLbl/GTerr. mov is
% an index into lObj.movieFilesAllGT.
ntt = NavigationTreeTable(tblOrig.Parent,[],cbkNavDataRow);
handles.navTreeTbl = ntt;

handles.listener = cell(0,1);
% Following listeners for table maintenance
handles.listener{end+1,1} = addlistener(lObj,...
  'gtIsGTMode','PostSet',@(s,e)cbkGTisGTModeChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'gtSuggMFTable','PostSet',@(s,e)cbkGTTableChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'gtSuggMFTableLbled','PostSet',@(s,e)cbkGTSuggMFTableLbledChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'movieRemoved',@(s,e)cbkMovieRemoved(hObject,s,e));
% Following listeners for table row selection
handles.listener{end+1,1} = addlistener(lObj,...
  'newMovie',@(s,e)cbkCurrMovFrmTgtChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'currFrame','PostSet',@(s,e)cbkCurrMovFrmTgtChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'currTarget','PostSet',@(s,e)cbkCurrMovFrmTgtChanged(hObject,s,e));

handles.figure1.DeleteFcn = @lclDeleteFig;
guidata(hObject,handles);
centerfig(handles.figure1,lObj.gdata.figure);

function varargout = GTManager_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

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

function cbkGTisGTModeChanged(hObject,src,evt)
% none atm

function cbkGTTableChanged(hObject,src,evt)
handles = guidata(hObject);
lObj = handles.labeler;
ntt = handles.navTreeTbl;
tbl = lObj.gtSuggMFTable;
ntt.setData(tbl);

function cbkGTSuggMFTableLbledChanged(hObject,src,evt)
handles = guidata(hObject);
lObj = handles.labeler;
ntt = handles.navTreeTbl;
tf = lObj.gtSuggMFTableLbled;
ntt.updateDataColumn('hasLbl',num2cell(tf));

function cbkMovieRemoved(hObject,src,evt)
iMovOrig2New = evt.iMovOrig2New;
handles = guidata(hObject);
ntt = handles.navTreeTbl;
tblData = ntt.treeTblData;
tblData = MFTable.remapIntegerKey(tblData,'mov',iMovOrig2New);
ntt.setData(tblData);

function cbkCurrMovFrmTgtChanged(hObject,src,evt)
handles = guidata(hObject);
lObj = handles.labeler;
if ~lObj.gtIsGTMode
  return;
end
mov = lObj.currMovie;
frm = lObj.currFrame;
iTgt = lObj.currTarget;
mftRow = table(mov,frm,iTgt);
ntt = handles.navTreeTbl;
[tf,iData] = ismember(mftRow,ntt.treeTblDat(:,MFTable.FLDSID));
if tf
  ntt.setSelectedDataRow(iData);
end  

function cbkTreeTableDataRowNaved(hObject,iData)
handles = guidata(hObject);
lObj = handles.labeler;
if ~lObj.gtIsGTMode
  warningNoTrace('GTManager:nav',...
    'Nagivation via GT Manager is disabled. Labeler is not in GT mode.');
  return;
end
ntt = handles.navTreeTbl;
mftRow = ntt.treeTblData(iData,:);
if 
XXX STOPPED HERE
end


function imovs = cbkGetSelectedMovies(hMMobj)
% Get current selection in Table
handles = guidata(hMMobj);
imovs = handles.navTreeTbl.getSelectedMovies();

function cbkSelectMovie(hMMobj,imov)
% Set current Labeler movie to movname
handles = guidata(hMMobj);
lObj = handles.labeler;
assert(isscalar(imov));
lObj.movieSet(imov);

function pbComputeGT_Callback(hObject, eventdata, handles)
tbl = handles.navTreeTbl;
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
tbl = handles.navTreeTbl;
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



function pbSuggestGTFrames_Callback(hObject, eventdata, handles)
function pbAddMovie_Callback(hObject, eventdata, handles)
function pbRmMovie_Callback(hObject, eventdata, handles)
