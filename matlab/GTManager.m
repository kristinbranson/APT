function varargout = GTManager(varargin)
% Movie table GUI

% Last Modified by GUIDE v2.5 08-May-2020 20:24:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
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

set(hObject,'MenuBar','None');
handles.menu_get_gt_frames.Label = 'GT Frames';

handles.menu_gtframes_suggest = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata)GTManager('menu_gtframes_suggest_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Auto-generate GT Frames',...
  'Tag','menu_gtframes_suggest',...
  'Checked','off',...
  'Visible','on');
handles.menu_gtframes_setlabeled = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata)GTManager('menu_gtframes_setlabeled_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set GT Frames to Current GT Labels',...
  'Tag','menu_gtframes_setlabeled',...
  'Checked','off',...
  'Visible','on');
handles.menu_gtframes_load = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata)GTManager('menu_gtframes_load_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Load GT Frames from file',...
  'Tag','menu_gtframes_load',...
  'Checked','off',...
  'Visible','on');

handles.menu_gtframes_suggest = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata)GTManager('menu_gtframes_summary_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','GT Frames Summary',...
  'Tag','menu_gtframes_summary',...
  'Separator','on',...
  'Visible','on');
%moveMenuItemAfter(handles.menu_file_export_labels_table,...
%  handles.menu_file_export_labels_trks);


lObj = varargin{1};
if isdeployed() || ~lObj.isgui,
  % AL 20171215. Compiled APTCluster on 15b, GTManager throws "Java tables
  % require Java Swing" from
  % cbkGTSuggUpdated->NavTreeTable/setData->treeTable.
  % 
  % This is foolishness, just avoid it for now. Should probably make a 
  % stripped-down track-only APTCluster that skips all the UI stuff anyway.
  handles.output = hObject;
  guidata(hObject,handles);
  return;
end

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

h = findall(handles.figure1,'-property','Units');
set(h,'Units','normalized');

% enabled when gt table set
h = findall(handles.figure1,'style','pushbutton');
set(h,'Enable','off');

cbkNavDataRow = @(iData)cbkTreeTableDataRowNaved(hObject,iData);
% The Data table of this NTT has fields mov/frm/iTgt/hasLbl/GTerr. mov is
% an index into lObj.movieFilesAllGT.
ntt = NavigationTreeTable(tblOrig.Parent,[],cbkNavDataRow);
handles.navTreeTbl = ntt;
handles.navTreeTblMovIdxs = nan(0,1);

handles.listener = cell(0,1);
% Following listeners for table maintenance
% handles.listener{end+1,1} = addlistener(lObj,...
%   'gtIsGTModeChanged',@(s,e)cbkGTisGTModeChanged(hObject,s,e));
% handles.listener{end+1,1} = addlistener(lObj,...
%   'gtSuggUpdated',@(s,e)cbkGTSuggUpdated(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'gtSuggMFTableLbledUpdated',@(s,e)cbkGTSuggMFTableLbledUpdated(hObject,s,e));
% handles.listener{end+1,1} = addlistener(lObj,...
%   'gtResUpdated',@(s,e)cbkGTResUpdated(hObject,s,e));
% Following listeners for table row selection
handles.listener{end+1,1} = addlistener(lObj,...
  'newMovie',@(s,e)cbkCurrMovFrmTgtChanged(hObject,s,e));
% handles.listener{end+1,1} = addlistener(lObj,...
%   'currFrame','PostSet',@(s,e)cbkCurrMovFrmTgtChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'didSetCurrTarget',@(s,e)(cbkCurrMovFrmTgtChanged(hObject,s,e)));

handles.figure1.DeleteFcn = @lclDeleteFig;
guidata(hObject,handles);
centerfig(handles.figure1,lObj.gdata.mainFigure_);

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
if isfield(handles,'navTreeTbl') && ~isempty(handles.navTreeTbl)
  delete(handles.navTreeTbl);
end

function cbkGTisGTModeChanged(hObject,src,evt)
% none atm

function cbkGTSuggUpdated(hObject,src,evt)
handles = guidata(hObject);
if isfield(handles, 'labeler') && isscalar(handles.labeler) && ishghandle(handles.labeler.controller_.mainFigure_)
  % All is well
else
  % Sometimes cbkGTSuggUpdated() gets called very early on, before handles.labeler is set
  return
end
lObj = handles.labeler;
% if lObj.isinit
%   return;
% end
ntt = handles.navTreeTbl;
tbl = lObj.gtSuggMFTable;
err = hlpGetGTErr(tbl,lObj);
hasLbl = lObj.gtSuggMFTableLbled;
tbl = [tbl table(hasLbl,err)];

% replace .mov with strings
tblMovIdxs = tbl.mov;
[iMovAbs,gt] = tblMovIdxs.get;
assert(all(gt));
iMovAbsUn = unique(iMovAbs);
iMovAbsUnCnt = arrayfun(@(x)nnz(x==iMovAbs),iMovAbsUn);
iMovAbsMax = max(iMovAbsUn);
iMovAbs2Cnt = zeros(iMovAbsMax,1); % iMovAbs2Cnt(iMovAbs) gives number of gt lbled frames for that mov
iMovAbs2Cnt(iMovAbsUn) = iMovAbsUnCnt;
iMovAbsCnt = iMovAbs2Cnt(iMovAbs);
% movstrs = lObj.getMovieFilesAllFullMovIdx(tblMovIdxs);
% movstrs = movstrs(:,1);
% movstrs = cellfun(@FSPath.twoLevelFilename,movstrs,'uni',0);
numDigits = floor(log10(lObj.nmoviesGT)+1);
fmt = sprintf('%%0%dd -- %%d frames',numDigits);
%movstrs = strcat(arrayfun(@(x)sprintf(fmt,x),iMovAbs,'uni',0),movstrs);
movstrs = arrayfun(@(x,y)sprintf(fmt,x,y),iMovAbs,iMovAbsCnt,'uni',0);
tbl.mov = movstrs;

COLS = [MFTable.FLDSID {'hasLbl' 'err'}];
PRETTYCOLS = {'Movie' 'Frame' 'Target' 'Has Labels' 'Error'};
if lObj.maIsMA
  % Drop target for multi-animal MK 20250523
  COLS = COLS([1 2 4 5]);
  PRETTYCOLS = PRETTYCOLS([1 2 4 5]);
end
tbl = tbl(:,COLS);
ntt.setData(tbl,'prettyHdrs',PRETTYCOLS);
handles.navTreeTblMovIdxs = tblMovIdxs;
guidata(hObject,handles);

pbEnable = onIff(~isempty(tbl));
h = findall(handles.figure1,'style','pushbutton');
set(h,'Enable',pbEnable);

function cbkGTSuggMFTableLbledUpdated(hObject,src,evt)
handles = guidata(hObject);
lObj = handles.labeler;
ntt = handles.navTreeTbl;
tf = lObj.gtSuggMFTableLbled;
ntt.updateDataColumn('hasLbl',num2cell(tf));

function cbkGTResUpdated(hObject,src,evt)
handles = guidata(hObject);
if isfield(handles, 'labeler') && isscalar(handles.labeler) && ishghandle(handles.labeler)
  % All is well
else
  % Sometimes cbkGTResUpdated() gets called very early on, before handles.labeler is set
  return
end
lObj = handles.labeler;
tblSugg = lObj.gtSuggMFTable;
ntt = handles.navTreeTbl;
err = hlpGetGTErr(tblSugg,lObj);
ntt.updateDataColumn('err',num2cell(err));

function err = hlpGetGTErr(tblSugg,lObj)
% Get computed GT results/err for given suggestion table
n = height(tblSugg);
err = nan(n,1);
tblRes = lObj.gtTblRes;
if ~isempty(tblRes)
  [tf,loc] = tblismember(tblSugg,tblRes,MFTable.FLDSID);
  err(tf) = tblRes.meanL2err(loc(tf));
end

function cbkCurrMovFrmTgtChanged(hObject,src,evt)
handles = guidata(hObject);
lObj = handles.labeler;
if lObj.isinit || ~lObj.hasMovie || ~lObj.gtIsGTMode
  return;
end
mIdx = lObj.currMovIdx;
frm = lObj.currFrame;
iTgt = lObj.currTarget;
ntt = handles.navTreeTbl;
nttData = ntt.treeTblData;
nData = ntt.nData;
nttMovIdxs = handles.navTreeTblMovIdxs;
assert(nData==numel(nttMovIdxs));
if nData>0
  if lObj.maIsMA
    tf = mIdx==nttMovIdxs & frm==nttData.frm;
  else
    tf = mIdx==nttMovIdxs & frm==nttData.frm & iTgt==nttData.iTgt;
  end
  iData = find(tf);
  if ~isempty(iData)
    assert(isscalar(iData));
    ntt.setSelectedDataRow(iData);
  end
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
mftRow.mov = handles.navTreeTblMovIdxs(iData);
lclNavToMFT(lObj,mftRow);

function lclNavToMFT(lObj,mftRow)
iMov = mftRow.mov.get();
if iMov~=lObj.currMovie
  lObj.movieSetGUI(iMov);
end
if isfield(mftRow,'iTgt')
  itgt = mftRow.iTgt;
else
  itgt = NaN;
end
lObj.setFrameAndTargetGUI(mftRow.frm,itgt);

% function imovs = cbkGetSelectedMovies(hMMobj)
% % Get current selection in Table
% handles = guidata(hMMobj);
% imovs = handles.navTreeTbl.getSelectedMovies();

function menu_gtframes_suggest_Callback(hObject, eventdata, handles)
LabelerGT.generateSuggestionsUI(handles.labeler);
function menu_gtframes_setlabeled_Callback(hObject, eventdata, handles)
LabelerGT.setSuggestionsToLabeledUI(handles.labeler);
function menu_gtframes_load_Callback(hObject, eventdata, handles)
LabelerGT.loadSuggestionsUI(handles.labeler);
function menu_gtframes_summary_Callback(hObject, eventdata, handles)
lObj = handles.labeler;
disp(lObj.gtLabeledFrameSummary);

function pbNextUnlabeled_Callback(hObject, eventdata, handles)
lObj = handles.labeler;
ntt = handles.navTreeTbl;
% if ntt.isEmpty()
%   msgbox('Groundtruthing frames have not been set. See the "Get GT Frames" menu.',...
%     'No GT frames set');
%   return;
% end

iDataSel = ntt.getSelectedDataRow();
if isempty(iDataSel)
  iDataSel = 0;
else
  iDataSel = iDataSel(end); 
end

% Assumed invariant: rows of ntt.treeTblData, lObj.gtSuggMFTable,
% lObj.gtSuggMFTableLbled all correspond

nttData = ntt.treeTblData;
tfUnlbled = ~lObj.gtSuggMFTableLbled;
nGT = height(nttData);
szassert(tfUnlbled,[nGT 1]);
rowsLook = iDataSel+1:nGT; % could be empty
iRow = find(tfUnlbled(rowsLook),1); % offset not applied yet
if isempty(iRow)
  msgbox('No more unlabeled frames.');
else
  iRow = iRow+iDataSel;
  mftRow = lObj.gtSuggMFTable(iRow,:);
  lclNavToMFT(lObj,mftRow)
end

function pbGoSelected_Callback(hObject, eventdata, handles)
% Switch to selected row (mov/frm/tgt)

ntt = handles.navTreeTbl;
% if ntt.isEmpty()
%   msgbox('Groundtruthing frames have not been set. See the "Get GT Frames" menu.',...
%     'No GT frames set');
%   return
% end

iData = ntt.getSelectedDataRow();
if isempty(iData)
  msgbox('Please select a row in the table.','No row selected');
  return;
end
if numel(iData)>1
  warningNoTrace('Multiple rows selected. Using first selected row.');
  iData = iData(1);
end
mftRow = ntt.treeTblData(iData,:);
mftRow.mov = handles.navTreeTblMovIdxs(iData);
lObj = handles.labeler;
lclNavToMFT(lObj,mftRow);

function pbComputeGT_Callback(hObject, eventdata, handles)
% ntt = handles.navTreeTbl;
% if ntt.isEmpty()
%   msgbox('Groundtruthing frames have not been set. See the "Get GT Frames" menu.',...
%     'No GT frames set');
%   return;
% end

lObj = handles.labeler;
lObj.gtComputeGTPerformance();
