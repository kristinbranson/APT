function hFig = GTManager(varargin)

if nargin == 1 && isa(varargin{1},'Labeler'),
  lObj = varargin{1};
else
  feval(varargin{:});
  return;
end

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

assert(isa(lObj,'Labeler'));
%obj.hFig = MovieManager(obj);
handles = struct;
handles.labeler = lObj;
      
hFig = uifigure('Units','pixels','Position',[951,1400,733,733],...
  'Name','Groundtruth Navigator','Visible','off');
mainController = lObj.controller_ ;
if ~isempty(mainController)
  mainController.addSatellite(hFig) ;
end

handles.figure1 = hFig;

handles.gl = uigridlayout(hFig,[4,1],'RowHeight',{'1x','1x',40},'tag','gl');

handles.tblGTMovie = uitable(handles.gl,...
  'ColumnName',{'','Movie','N to Label','N Labeled'},...
  'RowName',{},...
  'ColumnWidth',{35,'1x',100,100},...
  'tag','tblGTMovie',...
  'SelectionType','row','Multiselect','off',...
  'CellSelectionCallback',@(src,evt) cellSelectionCallbacktblGTMovie(hFig,src,evt)); % todo

columnnames = getTblFrameColumnNames(handles);
handles.tblFrame = uitable(handles.gl,...
  'ColumnName',columnnames,'tag','tblFrame',...
  'SelectionType','row','Multiselect','off',...
  'DoubleClickedFcn',@(src,evt) doubleClickFcnCallbacktblFrame(hFig,src,evt),...
  'ColumnSortable',true(1,numel(columnnames))); 
%   'CellSelectionCallback',@(src,evt) cellSelectionCallbacktblFrame(hFig,src,evt),... % todo

handles.gl_buttons = uigridlayout(handles.gl,[1,4],'Padding',[0,0,0,0],'tag','gl_buttons');
handles.pbNextUnlabeled = uibutton(handles.gl_buttons,'Text','Next Unlabeled','tag','pbNextUnlabeled',...
  'ButtonPushedFcn',@(src,evt) pbNextUnlabeled_Callback(hFig,src,evt),'Enable','off');

handles.pbGoSelected = uibutton(handles.gl_buttons,'Text','Go to Selected','tag','pbGoSelected',...
  'ButtonPushedFcn',@(src,evt) pbGoSelected_Callback(hFig,src,evt),'Enable','off');
handles.pbComputeGT = uibutton(handles.gl_buttons,'Text','Compute Accuracy','tag','pbComputeGT',...
  'ButtonPushedFcn',@(src,evt) pbComputeGT_Callback(hFig,src,evt),'Enable','off');
handles.pbUpdate = uibutton(handles.gl_buttons,'Text','Update','tag','pbUpdate',...
  'ButtonPushedFcn',@(src,evt) pbUpdate_Callback(hFig,src,evt),'Enable','on');
handles.pbs = [handles.pbNextUnlabeled,handles.pbGoSelected,handles.pbComputeGT,handles.pbUpdate];
set(hFig,'MenuBar','None');

handles.menu_get_gt_frames = uimenu('Tag','menu_get_gt_frames','Text','To-Label List','Parent',hFig);
handles.menu_gtframes_suggest = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata) menu_gtframes_suggest_Callback(hObject,eventdata,guidata(hObject)),...
  'Label','Randomly select to-label list...',...
  'Tag','menu_gtframes_suggest',...
  'Checked','off',...
  'Visible','on');
handles.menu_gtframes_setlabeled = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata) menu_gtframes_setlabeled_Callback(hObject,eventdata,guidata(hObject)),...
  'Label','Set to-label list to current groundtruth labels',...
  'Tag','menu_gtframes_setlabeled',...
  'Checked','off',...
  'Visible','on');
handles.menu_gtframes_load = uimenu('Parent',handles.menu_get_gt_frames,...
  'Callback',@(hObject,eventdata) menu_gtframes_load_Callback(hObject,eventdata,guidata(hObject)),...
  'Label','Load to-label list from file...',...
  'Tag','menu_gtframes_load',...
  'Checked','off',...
  'Visible','on');

% cbkNavDataRow = @(iData)cbkTreeTableDataRowNaved(hObject,iData);
% % The Data table of this NTT has fields mov/frm/iTgt/hasLbl/GTerr. mov is
% % an index into lObj.movieFilesAllGT.
% ntt = NavigationTreeTable(tblOrig.Parent,[],cbkNavDataRow);
% handles.navTreeTbl = ntt;
% handles.navTreeTblMovIdxs = nan(0,1);

handles.listener = cell(0,1);
% Following listeners for table maintenance
% handles.listener{end+1,1} = addlistener(lObj,...
%   'gtIsGTModeChanged',@(s,e)cbkGTisGTModeChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'gtSuggUpdated',@(s,e)cbkGTSuggUpdated(hFig,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'gtSuggMFTableLbledUpdated',@(s,e)cbkGTSuggUpdated(hFig,s,e));
% handles.listener{end+1,1} = addlistener(lObj,...
%   'gtResUpdated',@(s,e)cbkGTResUpdated(hObject,s,e));
% Following listeners for table row selection
handles.listener{end+1,1} = addlistener(lObj,...
  'newMovie',@(s,e)cbkCurrMovFrmTgtChanged(hFig,s,e));
% handles.listener{end+1,1} = addlistener(lObj,...
%   'currFrame','PostSet',@(s,e)cbkCurrMovFrmTgtChanged(hObject,s,e));
handles.listener{end+1,1} = addlistener(lObj,...
  'didSetCurrTarget',@(s,e)(cbkCurrMovFrmTgtChanged(hFig,s,e)));
handles.listener{end+1,1} = addlistener(lObj,...
  'gtResUpdated',@(s,e) (cbkGTResUpdated(hFig,s,e)));
handles.listener{end+1,1} = addlistener(lObj, ...
  'updateStuffInHlpSetCurrPrevFrame', @(s,e) cbkCurrMovFrmTgtChanged(hFig,s,e));



handles.figure1.DeleteFcn = @lclDeleteFig;
handles = updateAll(handles);
guidata(hFig,handles);
set(hFig,'Visible','on');

centerfig(handles.figure1,lObj.gdata.mainFigure_);

function columnnames = getTblFrameColumnNames(handles)
if handles.labeler.hasTrx,
  columnnames = {'Frame','Target','Labeled','Error'};
else
  columnnames = {'Frame','Has Labels','Error'};
end

function lclDeleteFig(src,evt)
handles = guidata(src);
if ~isempty(handles) && isfield(handles,'listener'),
  listenObjs = handles.listener;
  for i=1:numel(listenObjs)
    o = listenObjs{i};
    if isvalid(o)
      delete(o);
    end
  end
end

function setTableSelection(uitbl,row)

if isequal(uitbl.Selection,row),
  return;
end
uitbl.Selection = row;
if ~isempty(row),
  scroll(uitbl,"row",row);
end


function updateTblFrame(handles,fn)
rows = handles.tblGTMovie.Selection;
columnnames = getTblFrameColumnNames(handles);
handles.tblFrame.ColumnName = columnnames;
columnnames = getTblFrameColumnNames(handles);
handles.tblFrame.ColumnSortable = true(1,numel(columnnames));

doupdate = nargin > 1;

if isempty(rows)
  handles.tblFrame.Data = cell(0,numel(columnnames));
else
  row = rows(1);
  imov = handles.iMovUn(row);
  idx = handles.tbl.mov == imov;
  if doupdate,
    data = handles.tblFrame.Data;
  else
    data = cell(nnz(idx),numel(columnnames));
  end
  if ~doupdate || strcmpi(fn,'frm'),
    data(:,1) = num2cell(handles.tbl.frm(idx));
  end
  if handles.labeler.hasTrx && (~doupdate || strcmpi(fn,'iTgt')),
    data(:,2) = num2cell(handles.tbl.iTgt(idx));
  end
  if ~doupdate || strcmp(fn,'hasLbl'),
    col = double(handles.labeler.hasTrx) + 2;
    data(:,col) = num2cell(handles.hasLbl(idx));
  end
  if ~doupdate || strcmp(fn,'err'),
    col = double(handles.labeler.hasTrx) + 3;    
    if any(~isnan(handles.err)),
      data(:,col) = num2cell(handles.err(idx));
    else
      data(:,col) = cell(size(data,1),1);
    end
  end
  handles.tblFrame.Data = data;
  if isempty(handles.tblFrame.Selection) && ~isempty(data),
    setTableSelection(handles.tblFrame,1);
  end
end

function handles = updateAll(handles)

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
tbl_sugg = lObj.gtSuggMFTable;
tbl_label = lObj.labelGetMFTableLabeled('useTrain',0,'mftonly',true);
handles.tbl = unique([tbl_sugg;tbl_label],'rows');

handles.err = hlpGetGTErr(handles.tbl,lObj);
handles.hasLbl = lObj.getIsLabeledGT(handles.tbl);
iMovUnAbs = (1:lObj.nmoviesGT)';
handles.iMovUn = MovieIndex(iMovUnAbs,true);

% replace .mov with strings
iMov = handles.tbl.mov;
iMovUnLabeledCnt = arrayfun(@(x)nnz(x==iMov&handles.hasLbl),handles.iMovUn);
iMovUnToLabelCnt = arrayfun(@(x)nnz(x==iMov&~handles.hasLbl),handles.iMovUn);
movStrsUn = lObj.getMovieFilesAllFullMovIdx(handles.iMovUn);
% movstrs = lObj.getMovieFilesAllFullMovIdx(tblMovIdxs);
% movstrs = movstrs(:,1);
% movstrs = cellfun(@FSPath.twoLevelFilename,movstrs,'uni',0);

movTableData = [num2cell(iMovUnAbs(:)),movStrsUn(:,1),num2cell(iMovUnToLabelCnt(:)),num2cell(iMovUnLabeledCnt)];

handles.tblGTMovie.Data = movTableData;

updateTblFrame(handles);

pbEnable = onIff(~isempty(handles.tbl));
set(handles.pbs,'Enable',pbEnable);

function cbkGTisGTModeChanged(hObject,src,evt)
% fprintf('\nGTManager:cbkGTisGTModeChanged\n');
% stack = dbstack;
% for i = 1:length(stack)
%     fprintf('%s (line %d)\n', stack(i).name, stack(i).line);
% end
% if nargin >= 3,
%   disp(evt);
% end
% none atm

function cbkGTSuggUpdated(hObject,src,evt)

% fprintf('\nGTManager:cbkGTSuggUpdated\n');
% stack = dbstack;
% for i = 1:length(stack)
%     fprintf('%s (line %d)\n', stack(i).name, stack(i).line);
% end
% if nargin >= 3,
%   disp(evt);
% end
% t0 = tic;

handles = guidata(hObject);
handles = updateAll(handles);
guidata(hObject,handles);
% disp(toc(t0));

function cellSelectionCallbacktblGTMovie(hFig,src,evt)
handles = guidata(hFig);
updateTblFrame(handles);

function cbkGTSuggMFTableLbledUpdated(hObject,src,evt) % todo, remove this hook, put update button
% fprintf('\nGTManager:cbkGTSuggMFTableLbledUpdated\n');
% stack = dbstack;
% for i = 1:length(stack)
%     fprintf('%s (line %d)\n', stack(i).name, stack(i).line);
% end
% if nargin >= 3,
%   disp(evt);
% end
% t0 = tic;

handles = guidata(hObject);
lObj = handles.labeler;
newHasLbl = lObj.gtSuggMFTableLbled;
oldHasLbl = handles.hasLbl;
if isequaln(oldHasLbl,newHasLbl),
  return;
end
handles = updateAll(handles);
guidata(hObject,handles);

% disp(toc(t0));

function cbkGTResUpdated(hObject,src,evt)
% fprintf('\nGTManager:cbkGTResUpdated\n');
% stack = dbstack;
% for i = 1:length(stack)
%     fprintf('%s (line %d)\n', stack(i).name, stack(i).line);
% end
% if nargin >= 3,
%   disp(evt);
% end
% t0 = tic;

handles = guidata(hObject);
if isfield(handles, 'labeler') && isscalar(handles.labeler) && isa(handles.labeler,'Labeler'),
  % All is well
else
  % Sometimes cbkGTResUpdated() gets called very early on, before handles.labeler is set
  return
end
handles = updateAll(handles);
guidata(hObject,handles);
% disp(toc(t0));

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
% fprintf('\nGTManager:cbkCurrMovFrmTgtChanged\n');
% stack = dbstack;
% for i = 1:length(stack)
%     fprintf('%s (line %d)\n', stack(i).name, stack(i).line);
% end
% if nargin >= 3,
%   disp(evt);
% end
% t0 = tic;

handles = guidata(hObject);
lObj = handles.labeler;
if lObj.isinit || ~lObj.hasMovie || ~lObj.gtIsGTMode
  return;
end
mIdx = lObj.currMovIdx;
frm = lObj.currFrame;
iTgt = lObj.currTarget;
newrow = find(handles.iMovUn==mIdx);
oldrow = handles.tblGTMovie.Selection;
if ~isequal(newrow,oldrow),
  setTableSelection(handles.tblGTMovie,newrow);
  updateTblFrame(handles);
end

data = handles.tblFrame.Data;
if lObj.hasTrx,
  newrow = find(cell2mat(data(:,1))==frm & cell2mat(data(:,2))==iTgt);
else
  newrow = find(cell2mat(data(:,1))==frm);
end
setTableSelection(handles.tblFrame,newrow);
% disp(toc(t0));

function doubleClickFcnCallbacktblFrame(hObject,src,evt)
handles = guidata(hObject);
lObj = handles.labeler;
if ~lObj.gtIsGTMode
  warningNoTrace('GTManager:nav',...
    'Nagivation via GT Manager is disabled. Labeler is not in GT mode.');
  return;
end

[mov,ft] = getMFT(handles);
if numel(mov) ~= 1 || size(ft,1) ~= 1,
  return;
end
lclNavToMFT(lObj,mov(1),ft(1,:));

function [movrow,ftrow] = getSelection(handles)
if isempty(handles.tblFrame.Selection),
  ftrow = [];
else
  ftrow = unique(handles.tblFrame.Selection(:,1));
end
if isempty(handles.tblGTMovie.Selection),
  movrow = [];
else
  movrow = unique(handles.tblGTMovie.Selection(:,1));
end


function [mov,ft] = getMFT(handles,movrow,ftrow)
lObj = handles.labeler;
if nargin < 2,
  [movrow,ftrow] = getSelection(handles);
end

if isempty(handles.tblFrame.Data),

end
ft = double(cell2mat(handles.tblFrame.Data(ftrow,1)));
if lObj.hasTrx,
  ft = [ft,double(cell2mat(handles.tblFrame.Data(ftrow,2)))];
end
mov = handles.iMovUn(movrow);

function lclNavToMFT(lObj,mov,ft)
iMov = mov.get();
if iMov~=lObj.currMovie
  lObj.movieSetGUI(iMov);
end
if numel(ft) > 1,
  itgt = ft(2);
  lObj.setFrameAndTargetGUI(ft(1),itgt);
else
  lObj.setFrameGUI(ft(1));

  itgt = nan;
end

% function imovs = cbkGetSelectedMovies(hMMobj)
% % Get current selection in Table
% handles = guidata(hMMobj);
% imovs = handles.navTreeTbl.getSelectedMovies();

function menu_gtframes_suggest_Callback(hObject, eventdata, handles)
LabelerGT.generateSuggestionsUI(handles.labeler);
handles = updateAll(handles);
guidata(handles.figure1,handles);

function menu_gtframes_setlabeled_Callback(hObject, eventdata, handles)
LabelerGT.setSuggestionsToLabeledUI(handles.labeler);
handles = updateAll(handles);
guidata(handles.figure1,handles);

function menu_gtframes_load_Callback(hObject, eventdata, handles)
LabelerGT.loadSuggestionsUI(handles.labeler);
handles = updateAll(handles);
guidata(handles.figure1,handles);

function pbNextUnlabeled_Callback(hObject,src,evt)
% todo, use table sorting order
% todo sorting order seems to change when we call this
handles = guidata(hObject);
lObj = handles.labeler;
if ~any(handles.hasLbl),
  msgbox('No more unlabeled frames.','','modal');
end

[movrow0,ftrow] = getSelection(handles);
movrow = movrow0;
if isempty(movrow),
  movrow = 1;
  ftrow = 0;
end
if isempty(ftrow),
  ftrow = 0;
end
for movrow = movrow:numel(handles.iMovUn),
  mov = handles.iMovUn(movrow);
  idx = handles.tbl.mov == mov;
  tfUnlbled = ~handles.hasLbl(idx);
  iRow = find(tfUnlbled(ftrow+1:end),1);
  if ~isempty(iRow),
    iRow = iRow + ftrow;
    break;
  end
  ftrow = 0;
end
if isempty(iRow)
  msgbox('No more unlabeled frames.');
else
  if ~isequal(movrow,movrow0),
    setTableSelection(handles.tblGTMovie,movrow);
    handles = updateAll(handles);
  end
  [mov,ft] = getMFT(handles,movrow,iRow);
  setTableSelection(handles.tblFrame,iRow);
  lclNavToMFT(lObj,mov,ft);
end

guidata(hObject,handles);

function pbGoSelected_Callback(hObject, src, evt)
% Switch to selected row (mov/frm/tgt)
handles = guidata(hObject);
lObj = handles.labeler;
if ~lObj.gtIsGTMode
  warningNoTrace('GTManager:nav',...
    'Nagivation via GT Manager is disabled. Labeler is not in GT mode.');
  return;
end

[mov,ft] = getMFT(handles);
if isempty(mov)
  msgbox('Please select a row in each table.','No movie selected');
  return;
end
if isempty(ft),
  msgbox('Please select a row in each table.','No frame/target selected');
  return;
end
lclNavToMFT(lObj,mov(1),ft(1,:));

function pbComputeGT_Callback(hObject, src, evt)
handles = guidata(hObject);
lObj = handles.labeler;
mainController = lObj.controller_ ;
if isempty(mainController)
  whichlabels = 'all' ;
else
  response = mainController.askAboutUnrequestedGTLabelsIfNeeded_() ;
  if strcmp(response, 'cancel') 
    return
  end
  whichlabels = response ;
end
lObj.gtComputeGTPerformance('whichlabels',whichlabels);
handles = updateAll(handles);
guidata(hObject,handles);

function pbUpdate_Callback(hFig,src,evt)
handles = guidata(hFig);
handles = updateAll(handles);
guidata(hFig,handles);