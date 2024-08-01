function varargout = CPRVizTrackDiagsGUI(varargin)
% CPR diagnostics visualization

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CPRVizTrackDiagsGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @CPRVizTrackDiagsGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
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

% CPRVizTrackDiagsGUI(labelerObj)
function CPRVizTrackDiagsGUI_OpeningFcn(hObject, eventdata, handles, varargin)

lObj = varargin{1};
if isempty(lObj.tracker)
  error('CPRVizTrackDiagsGUI:track','No tracker found');
end
if ~strcmp(lObj.trackerAlgo,'cpr')
  error('Supported only for CPR tracking.');
end
if ~lObj.tracker.hasTrained
  error('CPRVizTrackDiagsGUI:train','Tracker has not been trained.');
end

vizObj = CPRVizTrackDiags(lObj,hObject);
listeners = cell(0,1);
listeners{end+1,1} = addlistener(lObj,'currFrame','PostSet',@(s,e)cbkCurrFrameChanged(s,e,vizObj));
listeners{end+1,1} = addlistener(lObj,'didSetCurrTarget',@(s,e)(cbkCurrTargetChanged(s,e,vizObj)));
listeners{end+1,1} = addlistener(lObj,'newProject',@(s,e)cbkNewProj(s,e,vizObj));
listeners{end+1,1} = addlistener(vizObj,'iRep','PostSet',@cbkRepChanged);
listeners{end+1,1} = addlistener(handles.sldReplicate,'ContinuousValueChange',@(s,e)sldReplicate_Callback(s,e,struct('vizObj',vizObj)));
listeners{end+1,1} = addlistener(handles.sldMajorIter,'ContinuousValueChange',@(s,e)sldMajorIter_Callback(s,e,struct('vizObj',vizObj)));
listeners{end+1,1} = addlistener(handles.sldMinorIter,'ContinuousValueChange',@(s,e)sldMinorIter_Callback(s,e,struct('vizObj',vizObj)));
listeners{end+1,1} = addlistener(vizObj,'t','PostSet',@cbkMajorIterChanged);
listeners{end+1,1} = addlistener(vizObj,'u','PostSet',@cbkMinorIterChanged);
listeners{end+1,1} = addlistener(vizObj,'iFernHilite','PostSet',@cbkIFernHiliteChanged);
handles.listeners = listeners;
handles.vizObj = vizObj;

tblpos = handles.tblFeatures.Position;
tblposunits = handles.tblFeatures.Units;
jt = uiextras.jTable.Table(...
  'parent',handles.tblFeatures.Parent,...
  'Units',tblposunits,...
  'SelectionMode','discontiguous',...
  'Editable','off');
jt.Position = tblpos;
jt.MouseClickedCallback = @(src,evt)cbkTblFeaturesClicked(src,evt);
delete(handles.tblFeatures);
handles.tblFeatures = jt;

handles.txReplicate.TooltipString = handles.etReplicate.TooltipString;
handles.sldReplicate.TooltipString = handles.etReplicate.TooltipString;
handles.txMajorIter.TooltipString = handles.etMajorIter.TooltipString;
handles.sldMajorIter.TooltipString = handles.etMajorIter.TooltipString;
handles.txMinorIter.TooltipString = handles.etMinorIter.TooltipString;
handles.sldMinorIter.TooltipString = handles.etMinorIter.TooltipString;

handles.output = hObject;

guidata(hObject, handles);

hSld = handles.sldReplicate;
hSld.Value = 1;
hSld.Min = 1;
hSld.Max = vizObj.nRep;
hSld.SliderStep = [1/hSld.Max 1/hSld.Max*5];
hSld = handles.sldMajorIter;
hSld.Value = 1;
hSld.Min = 1;
hSld.Max = vizObj.tMax;
hSld.SliderStep = [1/hSld.Max 1/hSld.Max*5];
hSld = handles.sldMinorIter;
hSld.Value = 1;
hSld.Min = 1;
hSld.Max = vizObj.uMax;
hSld.SliderStep = [1/hSld.Max 1/hSld.Max*5];

handles.cbShowViz.Value = 1;
%handles.tblFeatures.ColumnWidth = repmat({50},1,20);

vizObj.init();
vizObj.fireSetObs();

handles.txNumFerns.String = num2str(vizObj.M);
handles.txNumUsed.String = num2str(vizObj.metaNUse);
handles.txFtrType.String = vizObj.rcObj.prmFtr.type;
handles.txMetaType.String = vizObj.rcObj.prmFtr.metatype;
handles.txNumFeatures.String = num2str(vizObj.rcObj.prmFtr.F);

guidata(hObject, handles);

% UIWAIT makes CPRVizTrackDiagsGUI wait for user response (see UIRESUME)
% uiwait(handles.figCPRVizTrackDiagsGUI);

function varargout = CPRVizTrackDiagsGUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

function sldMajorIter_Callback(hObject, eventdata, handles)
v = hObject.Value;
handles.vizObj.t = round(v);
function etMajorIter_Callback(hObject, eventdata, handles)
val = round(str2double(hObject.String));
if ~isnan(val)
  handles.vizObj.t = val;
end
function cbkMajorIterChanged(src,evt)
vizObj = evt.AffectedObject;
if ~vizObj.isinit
  val = vizObj.t;
  vizObj.gdata.etMajorIter.String = num2str(val);
  vizObj.gdata.sldMajorIter.Value = val;
  updateLandmarkDist(vizObj);
  updateVizFeaturesAndTable(vizObj);
end

function sldMinorIter_Callback(hObject, eventdata, handles)
v = hObject.Value;
handles.vizObj.u = round(v);
function etMinorIter_Callback(hObject, eventdata, handles)
val = round(str2double(hObject.String));
if ~isnan(val)
  handles.vizObj.u = val;
end
function cbkMinorIterChanged(src,evt)
vizObj = evt.AffectedObject;
if ~vizObj.isinit
  val = vizObj.u;
  vizObj.gdata.etMinorIter.String = num2str(val);
  vizObj.gdata.sldMinorIter.Value = val;
  updateVizFeaturesAndTable(vizObj);
end

function etReplicate_Callback(hObject, eventdata, handles)
val = round(str2double(hObject.String));
if ~isnan(val)
  handles.vizObj.iRep = val;
end
function sldReplicate_Callback(hObject, eventdata, handles)
v = hObject.Value;
handles.vizObj.iRep = round(v);
function cbkRepChanged(src,evt)
vizObj = evt.AffectedObject;
if ~vizObj.isinit
  rep = vizObj.iRep;
  vizObj.gdata.etReplicate.String = num2str(rep);
  vizObj.gdata.sldReplicate.Value = rep;
  updateVizFeaturesAndTable(vizObj);
end

function cbkCurrFrameChanged(~,~,vizObj)
updateVizFeaturesAndTable(vizObj);

function cbkCurrTargetChanged(~,~,vizObj)
updateVizFeaturesAndTable(vizObj);

function updateLandmarkDist(vizObj)
ipts = vizObj.getLandmarksUsed();
ax = vizObj.gdata.axLandmarkDist;
histogram(ax,ipts,0.5:1:vizObj.lObj.nLabelPoints+0.5);
xlabel(ax,'landmark','fontweight','bold');
grid(ax,'on');
ax.XTick = 1:vizObj.lObj.nLabelPoints;
tstr = sprintf('Landmarks used, majorIter %d (N=%d).',vizObj.t,numel(ipts));
title(ax,tstr,'fontweight','bold','interpreter','none');

function updateVizFeaturesAndTable(vizObj)
% all tblFeatures info is driven by (implied) 1. prms.Ftr and (direct) 
% 2. rcObj.ftrSpecs, rcObj.ftrsUse

[fUse,xsUse,xsLbl] = vizObj.vizUpdate();
[M,nUse] = size(fUse);
szassert(xsUse,[M nUse]);
xsUse1 = xsUse{1};
ncol = size(xsUse1,2);
cellfun(@(x)assert(size(x,2)==ncol),xsUse);
assert(numel(xsLbl)==ncol);

tbldat = cell(M,nUse*(ncol+1));
for iFern=1:M
  row = cell(nUse,ncol+1);
  for iUse=1:nUse
    row{iUse,1} = fUse(iFern,iUse);
    if istable(xsUse{iFern,iUse})
      row(iUse,2:end) = table2cell(xsUse{iFern,iUse});
    else
      row(iUse,2:end) = num2cell(xsUse{iFern,iUse});
    end
  end
  tbldat(iFern,:) = row(:)'; % when iUse=2, interleave iUse1 and iUse2
end

rowlbl = repmat([{'iF'} xsLbl],nUse,1);
if nUse==2
  for iUse=1:nUse
    for j=1:size(rowlbl,2)
      str = rowlbl{iUse,j};
      if str(end)=='1' || str(end)=='2'
        str = [str '_']; %#ok<AGROW>
      end
      rowlbl{iUse,j} = [str num2str(iUse)];
    end
  end
end
tblcollbls = rowlbl(:)';

vizObj.gdata.tblFeatures.Data = tbldat;
vizObj.gdata.tblFeatures.ColumnName = tblcollbls;

function cbkTblFeaturesClicked(src,evt)
rows = src.SelectedRows;
if isempty(rows) 
  rows = 0;  
end
handles = guidata(src.Parent);
handles.vizObj.vizHiliteFernSet(rows(1));

function cbShowViz_Callback(hObject, eventdata, handles)
val = hObject.Value;
if val
  handles.vizObj.vizDetailShow();
else
  handles.vizObj.vizDetailHide();
end

function pbClearFeatureSelection_Callback(hObject, eventdata, handles)
handles.vizObj.vizHiliteFernSet(0);

function cbkIFernHiliteChanged(src,evt)
vizObj = evt.AffectedObject;
if ~vizObj.isinit
  iFHset = vizObj.iFernHilite;
  if iFHset==0
    iFHset = [];
  end
  vizObj.gdata.tblFeatures.SelectedRows = iFHset;
end

function menu_help_Callback(hObject, eventdata, handles)
HELPSTR = { ...
  'The landmark histogram includes all landmarks used in features selected during the current major iteration (including all minor iterations).'
  ''
  'The table details features selected for the current (major iteration, minor iteration) pair. Click a row of the table to highlight a single feature.'
  ''
  'Features are visualized in the main APT window. Solid triangles represent the estimated pose/shape given the current (replicate,major iteration) pair. White squares indicate computed, shape-indexed feature locations.'};
helpdlg(HELPSTR,'CPR Feature Visualization');

function cbkNewProj(src,evt,vizObj)
close(vizObj.gdata.figCPRVizTrackDiagsGUI);

function figCPRVizTrackDiagsGUI_CloseRequestFcn(hObject, eventdata, handles)
hList = handles.listeners;
for i=1:numel(hList)
  delete(hList{i});
end
%delete(handles.tblFeatures); % parented to panel, should be auto-deleted
delete(handles.vizObj);
delete(hObject);
