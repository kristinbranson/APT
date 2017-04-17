function varargout = CPRVizTrackDiagsGUI(varargin)

% Last Modified by GUIDE v2.5 17-Apr-2017 11:49:55

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
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

vizObj = CPRVizTrackDiags(lObj,hObject);
listeners = cell(0,1);
listeners{end+1,1} = addlistener(lObj,'currFrame','PostSet',@(s,e)cbkCurrFrameChanged(s,e,vizObj));
listeners{end+1,1} = addlistener(vizObj,'iRep','PostSet',@cbkRepChanged);
listeners{end+1,1} = addlistener(vizObj,'t','PostSet',@cbkMajorIterChanged);
listeners{end+1,1} = addlistener(vizObj,'u','PostSet',@cbkMinorIterChanged);
handles.listeners = listeners;
handles.vizObj = vizObj;

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
handles.tblFeatures.ColumnWidth = repmat({50},1,20);

vizObj.init();
vizObj.fireSetObs();

handles.txNumFerns.String = num2str(vizObj.M);
handles.txNumUsed.String = num2str(vizObj.metaNUse);
handles.txFtrType = vizObj.rcObj.prmFtr.type;
handles.txMetaType = vizObj.rcObj.prmFtr.metatype;
handles.txNumFeatures.String = num2str(vizObj.rcObj.prmFtr.F);

% UIWAIT makes CPRVizTrackDiagsGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

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

function etReplicateSpinner_Callback(hObject, eventdata, handles)
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
  vizObj.gdata.etReplicateSpinner.String = num2str(rep);
  vizObj.gdata.sldReplicate.Value = rep;
  updateVizFeaturesAndTable(vizObj);
end

function cbkCurrFrameChanged(src,evt,vizObj) %#ok<*INUSD>
updateVizFeaturesAndTable(vizObj);

function updateLandmarkDist(vizObj)
[ipts,ftrType] = vizObj.getLandmarksUsed();
ax = vizObj.gdata.axLandmarkDist;
histogram(ax,ipts,0.5:1:vizObj.lObj.nLabelPoints+0.5);
xlabel('landmark','fontweight','bold');
grid(ax,'on');
tstr = sprintf('N=%d. %d minorIters, %d ferns, ftrType ''%s''',numel(ipts),...
  vizObj.uMax,vizObj.rcObj.M,ftrType);
title(tstr,'fontweight','bold','interpreter','none');

function updateVizFeaturesAndTable(vizObj)
% all tblFeatures info is driven by (implied) 1. prms.Ftr and (direct) 
% 2. rcObj.ftrSpecs, rcObj.ftrsUse

[fUse,xsUse,xsLbl] = vizObj.vizUpdate();
[M,nUse] = size(fUse);
szassert(xsUse,[M nUse]);
ncol = numel(xsLbl);
tbldat = cell(M,nUse*(ncol+1));
for iFern=1:M
  row = cell(nUse,ncol+1);
  for iUse=1:nUse
    row{iUse,1} = fUse(iFern,iUse);
    row(iUse,2:end) = num2cell(xsUse{iFern,iUse});
  end
  tbldat(iFern,:) = row(:)'; % when iUse=2, interleave iUse1 and iUse2
end
rowlbl = repmat([{'iF'} xsLbl],2,1);
if nUse==2
  for iUse=1:nUse
    rowlbl(iUse,:) = cellfun(@(x)[x num2str(iUse)],rowlbl(iUse,:),'uni',0);
  end
end
tblcollbls = rowlbl(:)';

vizObj.gdata.tblFeatures.Data = tbldat;
vizObj.gdata.tblFeatures.ColumnName = tblcollbls;

function tblFeatures_CellSelectionCallback(hObject, eventdata, handles)
if isempty(eventdata.Indices) 
  handles.vizObj.vizHiliteFernClear();
else
  row = eventdata.Indices(1);
  handles.vizObj.vizHiliteFernSet(row);
end

function cbShowViz_Callback(hObject, eventdata, handles)
val = hObject.Value;
if val
  handles.vizObj.vizShow();
else
  handles.vizObj.vizHide();
end

function figure1_CloseRequestFcn(hObject, eventdata, handles)
hList = handles.listeners;
for i=1:numel(hList)
  delete(hList{i});
end
delete(handles.vizObj);
delete(hObject);

