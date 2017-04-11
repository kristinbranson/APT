function varargout = CPRVizTrackDiagsGUI(varargin)

% Last Modified by GUIDE v2.5 11-Apr-2017 12:51:17

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

h = handles.etReplicateSpinner;
hpos = h.Position;
jModel = javax.swing.SpinnerNumberModel(1,1,vizObj.tMax,1);
jSpinner = javax.swing.JSpinner(jModel);
jhSpinner = javacomponent(jSpinner,hpos,h.Parent);
set(jhSpinner,'StateChangedCallback',@(src,evt)cbkRepSpinnerCallback(src,evt,vizObj));
delete(h);
handles.spinReplicate = jhSpinner;

listeners = cell(0,1);
listeners{end+1,1} = addlistener(vizObj,'iRep','PostSet',@cbkRepChanged);
listeners{end+1,1} = addlistener(vizObj,'t','PostSet',@cbkMajorIterChanged);
listeners{end+1,1} = addlistener(vizObj,'u','PostSet',@cbkMinorIterChanged);
handles.listeners = listeners;
handles.vizObj = vizObj;

handles.output = hObject;

guidata(hObject, handles);

vizObj.init();

% UIWAIT makes CPRVizTrackDiagsGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function varargout = CPRVizTrackDiagsGUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

function sldMajorIter_Callback(hObject, eventdata, handles)

function etMajorIter_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if ~isnan(val)
  handles.vizObj.t = val;
end
function cbkMajorIterChanged(src,evt)
vizObj = evt.AffectedObject;
val = vizObj.t;
vizObj.gdata.etMajorIter.String = num2str(val);
updateLandmarkDist(vizObj);

function sldMinorIter_Callback(hObject, eventdata, handles)

function etMinorIter_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if ~isnan(val)
  handles.vizObj.u = val;
end
function cbkMinorIterChanged(src,evt)
vizObj = evt.AffectedObject;
val = vizObj.u;
vizObj.gdata.etMinorIter.String = num2str(val);

%function etReplicateSpinner_Callback(hObject, eventdata, handles)
function cbkRepSpinnerCallback(src,evt,vizObj)
rep = src.getValue;
vizObj.iRep = rep;

function cbkRepChanged(src,evt)
vizObj = evt.AffectedObject;
rep = vizObj.iRep;
vizObj.gdata.spinReplicate.setValue(rep);

function updateLandmarkDist(vizObj)
[ipts,ftrType] = vizObj.getLandmarksUsed();
ax = vizObj.gdata.axLandmarkDist;
histogram(ax,ipts,0.5:1:vizObj.lObj.nLabelPoints);
xlabel('landmark','fontweight','bold');
grid(ax,'on');
tstr = sprintf('%d minorIters, %d ferns, ftrType ''%s''',vizObj.uMax,...
  vizObj.rcObj.M,ftrType);
title(tstr,'fontweight','bold','interpreter','none');