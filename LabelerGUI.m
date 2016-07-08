function varargout = LabelerGUI(varargin)
% LARVALABELER MATLAB code for LarvaLabeler.fig
%      LARVALABELER, by itself, creates a new LARVALABELER or raises the existing
%      singleton*.
%
%      H = LARVALABELER returns the handle to a new LARVALABELER or the handle to
%      the existing singleton*.
%
%      LARVALABELER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LARVALABELER.M with the given input arguments.
%
%      LARVALABELER('Property','Value',...) creates a new LARVALABELER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LabelerGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LabelerGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LarvaLabeler

% Last Modified by GUIDE v2.5 05-Jul-2016 12:22:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
% AL20151104: 'dpi-aware' MATLAB graphics introduced in R2015b have trouble
% with .figs created in previous versions. Did significant testing across
% MATLAB versions and platforms and behavior appears at least mildly 
% wonky-- couldn't figure out a clean solution. For now use two .figs
if ispc && ~verLessThan('matlab','8.6') % 8.6==R2015b
  gui_Name = 'LabelerGUI_PC_15b';
elseif isunix
  gui_Name = 'LabelerGUI_lnx';
else
  gui_Name = 'LabelerGUI_PC_14b';
end
gui_State = struct('gui_Name',       gui_Name, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LabelerGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @LabelerGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1}) && exist(varargin{1}),
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function LabelerGUI_OpeningFcn(hObject,eventdata,handles,varargin) %#ok<INUSL>

if verLessThan('matlab','8.4')
  error('LabelerGUI:ver','LabelerGUI requires MATLAB version R2014b or later.');
end

% reinit uicontrol strings etc from GUIDE for cosmetic purposes
set(handles.txPrevIm,'String','');
set(handles.edit_frame,'String','');
set(handles.txStatus,'String','');
set(handles.txUnsavedChanges,'Visible','off');
set(handles.txLblCoreAux,'Visible','off');

handles.output = hObject;

handles.labelerObj = varargin{1};
varargin = varargin(2:end); %#ok<NASGU>
 
% multiview
nview = handles.labelerObj.nview;
figs = gobjects(1,nview);
ax = gobjects(1,nview);
figs(1) = handles.figure;
ax(1) = handles.axes_curr;
for i=2:nview
  figs(i) = figure('CloseRequestFcn',...
    @(s,e)cbkAuxFigCloseReq(s,e,handles.labelerObj));
  ax(i) = axes;
  handles.labelerObj.addDepHandle(figs(i));
end
handles.figs_all = figs;
handles.axes_all = ax;

arrayfun(@(x)colormap(x,gray),figs);
if nview>1
  viewNames = handles.labelerObj.viewNames;
  for i=1:nview
    figs(i).Name = sprintf('View: %s',viewNames{i});
  end
end

ims = gobjects(1,nview);
for iView=1:nview
  ims(iView) = imagesc(0,'Parent',ax(iView));
  set(ims(iView),'hittest','off');
  axisoff(ax(iView));
  hold(ax(iView),'on');
  set(ax(iView),'Color',[0 0 0]);
end
handles.images_all = ims;
handles.image_curr = ims(1);

% AL: important to get clickable points. Somehow this jiggers plot
% lims/scaling/coords so that points are more clickable; otherwise
% lblCore points in aux axes are impossible to click (eg without zooming
% way in or other contortions)
for i=2:numel(figs)
  figs(i).ResizeFcn = @cbkAuxAxResize;
end
%arrayfun(@(x)axis(x,'auto'),ax);



hold(handles.axes_occ,'on');
axis(handles.axes_occ,'ij');
axis(handles.axes_occ,[0 handles.labelerObj.nLabelPoints+1 0 2]);

handles.image_prev = imagesc(0,'Parent',handles.axes_prev);
set(handles.image_prev,'hittest','off');
axisoff(handles.axes_prev);
hold(handles.axes_prev,'on');
set(handles.axes_prev,'Color',[0 0 0]);

linkaxes([handles.axes_prev,handles.axes_curr]);

lObj = handles.labelerObj;

handles.labelTLManual = LabelTimeline(lObj,handles.axes_timeline_manual,true);
handles.figure.WindowButtonMotionFcn = @(src,evt)cbkWBMF(src,evt,lObj);
handles.figure.WindowButtonUpFcn = @(src,evt)cbkWBUF(src,evt,lObj);

listeners = cell(0,1);
listeners{end+1,1} = addlistener(handles.slider_frame,'ContinuousValueChange',@slider_frame_Callback);
listeners{end+1,1} = addlistener(handles.sldZoom,'ContinuousValueChange',@sldZoom_Callback);
listeners{end+1,1} = addlistener(lObj,'projname','PostSet',@cbkProjNameChanged);
listeners{end+1,1} = addlistener(lObj,'currFrame','PostSet',@cbkCurrFrameChanged);
listeners{end+1,1} = addlistener(lObj,'currTarget','PostSet',@cbkCurrTargetChanged);
listeners{end+1,1} = addlistener(lObj,'prevFrame','PostSet',@cbkPrevFrameChanged);
listeners{end+1,1} = addlistener(lObj,'labeledposNeedsSave','PostSet',@cbkLabeledPosNeedsSaveChanged);
listeners{end+1,1} = addlistener(lObj,'labelMode','PostSet',@cbkLabelModeChanged);
listeners{end+1,1} = addlistener(lObj,'targetZoomFac','PostSet',@cbkTargetZoomFacChanged);
listeners{end+1,1} = addlistener(lObj,'projFSInfo','PostSet',@cbkProjFSInfoChanged);
listeners{end+1,1} = addlistener(lObj,'moviename','PostSet',@cbkMovienameChanged);
listeners{end+1,1} = addlistener(lObj,'suspScore','PostSet',@cbkSuspScoreChanged);
listeners{end+1,1} = addlistener(lObj,'showTrxMode','PostSet',@cbkShowTrxModeChanged);
listeners{end+1,1} = addlistener(lObj,'tracker','PostSet',@cbkTrackerChanged);
listeners{end+1,1} = addlistener(lObj,'trackNFramesSmall','PostSet',@cbkTrackerNFramesChanged);
listeners{end+1,1} = addlistener(lObj,'trackNFramesLarge','PostSet',@cbkTrackerNFramesChanged);    
listeners{end+1,1} = addlistener(lObj,'trackNFramesNear','PostSet',@cbkTrackerNFramesChanged);
listeners{end+1,1} = addlistener(lObj,'movieCenterOnTarget','PostSet',@cbkMovieCenterOnTargetChanged);
listeners{end+1,1} = addlistener(lObj,'movieForceGrayscale','PostSet',@cbkMovieForceGrayscaleChanged);
listeners{end+1,1} = addlistener(lObj,'lblCore','PostSet',@cbkLblCoreChanged);
listeners{end+1,1} = addlistener(lObj,'newMovie',@cbkNewMovie);
listeners{end+1,1} = addlistener(handles.labelTLManual,'selectModeOn','PostSet',@cbkLabelTLManualSelectModeOn);
handles.listeners = listeners;

% These Labeler properties need their callbacks fired to properly init UI.
% Labeler will read .propsNeedInit from the GUIData to comply.
handles.propsNeedInit = {
  'labelMode' 
  'suspScore' 
  'showTrxMode' 
  'tracker' 
  'trackNFramesSmall' % trackNFramesLarge, trackNframesNear currently share same callback
  'movieCenterOnTarget'
  'movieForceGrayscale'};

set(handles.output,'Toolbar','figure');

colnames = handles.labelerObj.TBLTRX_STATIC_COLSTBL;
set(handles.tblTrx,'ColumnName',colnames,'Data',cell(0,numel(colnames)));
colnames = handles.labelerObj.TBLFRAMES_COLS; % AL: dumb b/c table update code uses hardcoded cols 
set(handles.tblFrames,'ColumnName',colnames,'Data',cell(0,numel(colnames)));

guidata(hObject, handles);

% UIWAIT makes LabelerGUI wait for user response (see UIRESUME)
% uiwait(handles.figure);

function varargout = LabelerGUI_OutputFcn(hObject, eventdata, handles) %#ok<*INUSL>
varargout{1} = handles.output;

function cbkAuxAxResize(src,data)
% AL 20160628: voodoo that may help make points more clickable. Sometimes
% pt clickability in MultiViewCalibrated mode is unstable (eg to anchor
% points etc)
ax = findall(src,'type','axes');
axis(ax,'image')
axis(ax,'auto');

function cbkAuxFigCloseReq(src,data,lObj)

if ~any(src==lObj.depHandles)
  delete(gcf);
  return;  
end

CLOSESTR = 'Close anyway';
DONTCLOSESTR = 'Cancel, don''t close';
sel = questdlg('This figure is required for your current multiview project.',...
  'Close Request Function',...
  DONTCLOSESTR,CLOSESTR,DONTCLOSESTR);
if isempty(sel)
  sel = DONTCLOSESTR;
end
switch sel
  case DONTCLOSESTR
    % none
  case CLOSESTR
    delete(gcf)
end

function cbkLblCoreChanged(src,evt)
lObj = evt.AffectedObject;
lblCore = lObj.lblCore;
if ~isempty(lblCore)
  lblCore.addlistener('hideLabels','PostSet',@cbkLblCoreHideLabelsChanged);
end

function cbkLblCoreHideLabelsChanged(src,evt)
lblCore = evt.AffectedObject;
gdata = lblCore.labeler.gdata;
gdata.menu_view_hide_labels.Checked = onIff(lblCore.hideLabels);

function cbkWBMF(src,evt,lObj)
lcore = lObj.lblCore;
if ~isempty(lcore)
  lcore.wbmf(src,evt);
end
lObj.gdata.labelTLManual.cbkWBMF(src,evt);

function cbkWBUF(src,evt,lObj)
if ~isempty(lObj.lblCore)
  lObj.lblCore.wbuf(src,evt);
end
lObj.gdata.labelTLManual.cbkWBUF(src,evt);

function cbkNewMovie(src,evt)
lObj = src;
gdata = lObj.gdata;
movRdrs = lObj.movieReader;
nframes = movRdrs(1).nframes;
ims = arrayfun(@(x)x.readframe(1),movRdrs,'uni',0);

% weirdo stuff
if isfield(movRdrs(1).info,'bitdepth')
  lObj.maxv = min(lObj.maxv,2^movRdrs(1).info.bitdepth-1);
elseif isa(ims{1},'uint16')
  lObj.maxv = min(2^16 - 1,lObj.maxv);
elseif isa(ims{1},'uint8')
  lObj.maxv = min(lObj.maxv,2^8 - 1);
else
  lObj.maxv = min(lObj.maxv,2^(ceil(log2(max(ims{1}(:)))/8)*8));
end
minvmaxv = [lObj.minv lObj.maxv];

hAxs = gdata.axes_all;
hIms = gdata.images_all;
assert(isequal(numel(ims),numel(hAxs),numel(hIms)));
for i=1:numel(ims)
  set(hIms(i),'CData',ims{i});
  set(hAxs(i),'CLim',minvmaxv,...
    'XLim',[.5,size(ims{i},2)+.5],...
    'YLim',[.5,size(ims{i},1)+.5]);
  axis(hAxs(i),'image');
  zoom(hAxs(i),'reset');
end

axprev = gdata.axes_prev;
set(axprev,'CLim',minvmaxv,...
  'XLim',[.5,size(ims{1},2)+.5],...
  'YLim',[.5,size(ims{1},1)+.5]);
zoom(axprev,'reset');

gdata.labelTLManual.initNewMovie();
gdata.labelTLManual.setLabelsFull();

sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
set(gdata.slider_frame,'Value',0,'SliderStep',sliderstep);
      
function cbkCurrFrameChanged(src,evt) %#ok<*INUSD>
lObj = evt.AffectedObject;
frm = lObj.currFrame;
nfrm = lObj.nframes;
gdata = lObj.gdata;
set(gdata.edit_frame,'String',num2str(frm));
sldval = (frm-1)/(nfrm-1);
if isnan(sldval)
  sldval = 0;
end
set(gdata.slider_frame,'Value',sldval);
gdata.labelTLManual.setCurrFrame(frm);

function cbkPrevFrameChanged(src,evt) %#ok<*INUSD>
lObj = evt.AffectedObject;
frm = lObj.prevFrame;
set(lObj.gdata.txPrevIm,'String',num2str(frm));

function cbkCurrTargetChanged(src,evt) %#ok<*INUSD>
lObj = evt.AffectedObject;
if lObj.hasTrx
  id = lObj.currTrxID;
  lObj.currImHud.updateTarget(id);
end

function cbkLabeledPosNeedsSaveChanged(src,evt)
lObj = evt.AffectedObject;
hTx = lObj.gdata.txUnsavedChanges;
val = lObj.labeledposNeedsSave;
if isscalar(val) && val
  set(hTx,'Visible','on');
else
  set(hTx,'Visible','off');
end

function cbkLabelModeChanged(src,evt)
lObj = evt.AffectedObject;
gd = lObj.gdata;
switch lObj.labelMode
  case LabelMode.SEQUENTIAL
    gd.menu_setup_sequential_mode.Checked = 'on';
    gd.menu_setup_template_mode.Checked = 'off';
    gd.menu_setup_highthroughput_mode.Checked = 'off';
    gd.menu_setup_tracking_correction_mode.Checked = 'off';
    gd.menu_setup_multiview_calibrated_mode.Checked = 'off';
    
    gd.menu_setup_createtemplate.Visible = 'off';
    gd.menu_setup_set_labeling_point.Visible = 'off';
    gd.menu_setup_unlock_all_frames.Visible = 'off';
    gd.menu_setup_lock_all_frames.Visible = 'off';
    gd.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.TEMPLATE
    gd.menu_setup_sequential_mode.Checked = 'off';
    gd.menu_setup_template_mode.Checked = 'on';
    gd.menu_setup_highthroughput_mode.Checked = 'off';
    gd.menu_setup_tracking_correction_mode.Checked = 'off';
    gd.menu_setup_multiview_calibrated_mode.Checked = 'off';

    gd.menu_setup_createtemplate.Visible = 'on';
    gd.menu_setup_set_labeling_point.Visible = 'off';
    gd.menu_setup_unlock_all_frames.Visible = 'off';
    gd.menu_setup_lock_all_frames.Visible = 'off';
    gd.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.HIGHTHROUGHPUT
    gd.menu_setup_sequential_mode.Checked = 'off';
    gd.menu_setup_template_mode.Checked = 'off';
    gd.menu_setup_highthroughput_mode.Checked = 'on';
    gd.menu_setup_tracking_correction_mode.Checked = 'off';
    gd.menu_setup_multiview_calibrated_mode.Checked = 'off';
    
    gd.menu_setup_createtemplate.Visible = 'off';
    gd.menu_setup_set_labeling_point.Visible = 'on';
    gd.menu_setup_unlock_all_frames.Visible = 'off';
    gd.menu_setup_lock_all_frames.Visible = 'off';
    gd.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.ERRORCORRECT
    gd.menu_setup_sequential_mode.Checked = 'off';
    gd.menu_setup_template_mode.Checked = 'off';
    gd.menu_setup_highthroughput_mode.Checked = 'off';
    gd.menu_setup_tracking_correction_mode.Checked = 'on';
    gd.menu_setup_multiview_calibrated_mode.Checked = 'off';
    
    gd.menu_setup_createtemplate.Visible = 'off';
    gd.menu_setup_set_labeling_point.Visible = 'off';
    gd.menu_setup_unlock_all_frames.Visible = 'on';
    gd.menu_setup_lock_all_frames.Visible = 'on';
    gd.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.MULTIVIEWCALIBRATED
    gd.menu_setup_sequential_mode.Checked = 'off';
    gd.menu_setup_template_mode.Checked = 'off';
    gd.menu_setup_highthroughput_mode.Checked = 'off';
    gd.menu_setup_tracking_correction_mode.Checked = 'off';
    gd.menu_setup_multiview_calibrated_mode.Checked = 'on';
    
    gd.menu_setup_createtemplate.Visible = 'off';
    gd.menu_setup_set_labeling_point.Visible = 'off';
    gd.menu_setup_unlock_all_frames.Visible = 'off';
    gd.menu_setup_lock_all_frames.Visible = 'off';
    gd.menu_setup_load_calibration_file.Visible = 'on';
end

function cbkTargetZoomFacChanged(src,evt)
lObj = evt.AffectedObject;
zf = lObj.targetZoomFac;
set(lObj.gdata.sldZoom,'Value',zf);

function cbkProjNameChanged(src,evt)
lObj = evt.AffectedObject;
pname = lObj.projname;
str = sprintf('Project %s created (unsaved) at %s',pname,datestr(now,16));
set(lObj.gdata.txStatus,'String',str);
set(lObj.gdata.txProjectName,'String',pname);

function cbkProjFSInfoChanged(src,evt)
lObj = evt.AffectedObject;
info = lObj.projFSInfo;
if ~isempty(info)  
  str = sprintf('Project %s %s at %s',info.filename,info.action,datestr(info.timestamp,16));
  set(lObj.gdata.txStatus,'String',str);
end

function cbkMovienameChanged(src,evt)
lObj = evt.AffectedObject;
mname = lObj.moviename;
set(lObj.gdata.txMoviename,'String',mname);
if ~isempty(mname)
  str = sprintf('new movie %s at %s',mname,datestr(now,16));
  set(lObj.gdata.txStatus,'String',str);
  
  % Fragile behavior when loading projects; want project status update to
  % persist and not movie status update. This depends on detailed ordering in 
  % Labeler.projLoad
end

function cbkMovieForceGrayscaleChanged(src,evt)
lObj = evt.AffectedObject;
tf = lObj.movieForceGrayscale;
lObj.gdata.menu_view_converttograyscale.Checked = onIff(tf);

function cbkSuspScoreChanged(src,evt)
lObj = evt.AffectedObject;
ss = lObj.suspScore;
lObj.currImHud.updateReadoutFields('hasSusp',~isempty(ss));

pnlSusp = lObj.gdata.pnlSusp;
tblSusp = lObj.gdata.tblSusp;
tfDoSusp = ~isempty(ss) && lObj.hasMovie && ~lObj.isinit;
if tfDoSusp 
  nfrms = lObj.nframes;
  ntgts = lObj.nTargets;
  [tgt,frm] = meshgrid(1:ntgts,1:nfrms);
  ss = ss{lObj.currMovie};
  
  frm = frm(:);
  tgt = tgt(:);
  ss = ss(:);
  tfnan = isnan(ss);
  frm = frm(~tfnan);
  tgt = tgt(~tfnan);
  ss = ss(~tfnan);
  
  [ss,idx] = sort(ss,1,'descend');
  frm = frm(idx);
  tgt = tgt(idx);
  
  mat = [frm tgt ss];
  tblSusp.Data = mat;
  pnlSusp.Visible = 'on';
  
  if verLessThan('matlab','R2015b') % findjobj doesn't work for >=2015b
    
    % make tblSusp column-sortable. 
    % AL 201510: Tried putting this in opening_fcn but
    % got weird behavior (findjobj couldn't find jsp)
    jscrollpane = findjobj(tblSusp);
    jtable = jscrollpane.getViewport.getView;
    jtable.setSortable(true);		% or: set(jtable,'Sortable','on');
    jtable.setAutoResort(true);
    jtable.setMultiColumnSortable(true);
    jtable.setPreserveSelectionsAfterSorting(true);
    % reset ColumnWidth, jtable messes it up
    cwidth = tblSusp.ColumnWidth;
    cwidth{end} = cwidth{end}-1;
    tblSusp.ColumnWidth = cwidth;
    cwidth{end} = cwidth{end}+1;
    tblSusp.ColumnWidth = cwidth;
  
    tblSusp.UserData = struct('jtable',jtable);   
  else
    % none
  end
  lObj.updateCurrSusp();
else
  tblSusp.Data = cell(0,3);
  pnlSusp.Visible = 'off';
end

% function cbkCurrSuspChanged(src,evt)
% lObj = evt.AffectedObject;
% ss = lObj.currSusp;
% if ~isequal(ss,[])
%   lObj.currImHud.updateSusp(ss);
% end

function cbkShowTrxModeChanged(src,evt)
lObj = evt.AffectedObject;
gd = lObj.gdata;
gd.menu_view_trajectories_showall.Checked = 'off';
gd.menu_view_trajectories_showcurrent.Checked = 'off';
gd.menu_view_trajectories_dontshow.Checked = 'off';
switch lObj.showTrxMode
  case ShowTrxMode.NONE
    gd.menu_view_trajectories_dontshow.Checked = 'on';
  case ShowTrxMode.CURRENT
    gd.menu_view_trajectories_showcurrent.Checked = 'on';
  case ShowTrxMode.ALL
    gd.menu_view_trajectories_showall.Checked = 'on';
end

function cbkTrackerChanged(src,evt)
lObj = evt.AffectedObject;
tf = ~isempty(lObj.tracker);
onOff = onIff(tf);
lObj.gdata.menu_track.Enable = onOff;
lObj.gdata.pbTrain.Enable = onOff;
lObj.gdata.pbTrack.Enable = onOff;

function cbkTrackerNFramesChanged(src,evt)
lObj = evt.AffectedObject;
initPUMTrack(lObj);

function initPUMTrack(lObj)
tms = enumeration('TrackMode');
menustrs = arrayfun(@(x)x.menuStr(lObj),tms,'uni',0);
hPUM = lObj.gdata.pumTrack;
hPUM.String = menustrs;
hPUM.UserData = tms;

function tm = getTrackMode(handles)
hPUM = handles.pumTrack;
tms = hPUM.UserData;
val = hPUM.Value;
tm = tms(val);

function cbkMovieCenterOnTargetChanged(src,evt)
lObj = evt.AffectedObject;
tf = lObj.movieCenterOnTarget;
lObj.gdata.menu_view_trajectories_centervideoontarget.Checked = onIff(tf);

function slider_frame_Callback(hObject,~)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles = guidata(hObject);
lObj = handles.labelerObj;
v = get(hObject,'Value');
f = round(1 + v * (lObj.nframes - 1));
cmod = handles.figure.CurrentModifier;
if ~isempty(cmod) && any(strcmp(cmod{1},{'control' 'shift'}))
  if f>lObj.currFrame
    lObj.frameUp(true);
  else
    lObj.frameDown(true);
  end  
else
  lObj.setFrame(f);
end

function slider_frame_CreateFcn(hObject,~,~)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edit_frame_Callback(hObject,~,handles)
lObj = handles.labelerObj;
f = str2double(get(hObject,'String'));
if isnan(f)
  set(hObject,'String',num2str(lObj.currFrame));
  return;
end
f = min(max(1,round(f)),lObj.nframes);
set(hObject,'String',num2str(f));
if f ~= lObj.currFrame
  lObj.setFrame(f)
end 
  
function edit_frame_CreateFcn(hObject,~,~)
if ispc && isequal(get(hObject,'BackgroundColor'), ...
                   get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end

function pbTrain_Callback(hObject, eventdata, handles)
handles.labelerObj.trackTrain();
function pbTrack_Callback(hObject, eventdata, handles)
tm = getTrackMode(handles);
handles.labelerObj.track(tm);

function pbClear_Callback(hObject, eventdata, handles)
handles.labelerObj.lblCore.clearLabels();

function tbAccept_Callback(hObject, eventdata, handles)
lc = handles.labelerObj.lblCore;
switch lc.state
  case LabelState.ADJUST
    lc.acceptLabels();
  case LabelState.ACCEPTED
    lc.unAcceptLabels();
  otherwise
    assert(false);
end

function tblTrx_CellSelectionCallback(hObject, eventdata, handles) %#ok<*DEFNU>
% hObject    handle to tblTrx (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
lObj = handles.labelerObj;
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  dat = get(hObject,'Data');
  id = dat{row,1};
  lObj.setTargetID(id);
end

hlpRemoveFocus(hObject,handles);

function hlpRemoveFocus(h,handles)
% Hack to manage focus. As usual the uitable is causing problems. The
% tables used for Target/Frame nav cause problems with focus/Keypresses as
% follows:
% 1. A row is selected in the target table, selecting that target.
% 2. If nothing else is done, the table has focus and traps arrow
% keypresses to navigate the table, instead of doing LabelCore stuff
% (moving selected points, changing frames, etc).
% 3. The following lines of code force the focus off the uitable.
%
% Other possible solutions: 
% - Figure out how to disable arrow-key nav in uitables. Looks like need to
% drop into Java and not super simple.
% - Don't use uitables, or use them in a separate figure window.
uicontrol(handles.txStatus);

function tblFrames_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to tblFrames (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
lObj = handles.labelerObj;
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  dat = get(hObject,'Data');
  lObj.setFrame(dat{row,1});
end

hlpRemoveFocus(hObject,handles);

function sldZoom_Callback(hObject, eventdata, ~)
handles = guidata(hObject);
lObj = handles.labelerObj;
zoomFac = get(hObject,'Value');
lObj.videoSetTargetZoomFac(zoomFac);
hlpRemoveFocus(hObject,handles);

function pbResetZoom_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.videoResetView();

function tblSusp_CellSelectionCallback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if verLessThan('matlab','R2015b')
  jt = lObj.gdata.tblSusp.UserData.jtable;
  row = jt.getSelectedRow; % 0 based
  frm = jt.getValueAt(row,0);
  iTgt = jt.getValueAt(row,1);
  if ~isempty(frm)
    frm = frm.longValueReal;
    iTgt = iTgt.longValueReal;
    lObj.setFrameAndTarget(frm,iTgt);
    hlpRemoveFocus(hObject,handles);
  end
else
  row = eventdata.Indices(1);
  dat = hObject.Data;
  frm = dat(row,1);
  iTgt = dat(row,2);
  lObj.setFrameAndTarget(frm,iTgt);
  hlpRemoveFocus(hObject,handles);
end

function tbTLSelectMode_Callback(hObject, eventdata, handles)
handles.labelTLManual.selectModeOn = hObject.Value;

function cbkLabelTLManualSelectModeOn(src,evt)
lblTLObj = evt.AffectedObject;
lblTLObj.lObj.gdata.tbTLSelectMode.Value = lblTLObj.selectModeOn;

%% menu
function menu_file_quick_open_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(false);
  if ~tfsucc
    return;
  end
  movfile = movfile{1};
  trxfile = trxfile{1};
  handles.labelerObj.projQuickOpen(movfile,trxfile);
end
function menu_file_new_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  lObj.projNew();
end
function menu_file_save_Callback(hObject, eventdata, handles)
handles.labelerObj.projSaveSmart();
function menu_file_saveas_Callback(hObject, eventdata, handles)
handles.labelerObj.projSaveAs();
function menu_file_load_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  lObj.projLoad();
end

function tfcontinue = hlpSave(labelerObj)
tfcontinue = true;
OPTION_SAVE = 'Save labels first';
OPTION_PROC = 'Proceed without saving';
OPTION_CANC = 'Cancel';
if labelerObj.labeledposNeedsSave
  res = questdlg('You have unsaved changes to your labels. If you proceed without saving, your changes will be lost.',...
    'Unsaved changes',OPTION_SAVE,OPTION_PROC,OPTION_CANC,OPTION_SAVE);
  switch res
    case OPTION_SAVE
      labelerObj.projSaveSmart();
    case OPTION_CANC
      tfcontinue = false;
    case OPTION_PROC
      % none
  end
end

function menu_file_managemovies_Callback(hObject,~,handles)
h = MovieManager(handles.labelerObj);
handles.labelerObj.addDepHandle(h);

function menu_file_import_labels_trk_curr_mov_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if ~lObj.hasMovie
  error('LabelerGUI:noMovie','No movie is loaded.');
end
iMov = lObj.currMovie;
if lObj.labelposMovieHasLabels(iMov)
  resp = questdlg('Current movie has labels that will be overwritten. OK?',...
    'Import Labels','OK, Proceed','Cancel','Cancel');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'OK, Proceed'
      % none
    case 'Cancel'
      return;
    otherwise
      assert(false); 
  end
end
handles.labelerObj.labelImportTrkCurrMov();

function menu_file_export_labels_trks_Callback(hObject, eventdata, handles)
handles.labelerObj.labelExportTrk();

% function menu_file_openmovietrx_Callback(hObject, eventdata, handles)
% lObj = handles.labelerObj;
% if hlpSave(lObj)
%   lObj.loadMovie([],[]);
%   if lObj.hasMovie
%     lObj.labelingInit();
%   end
% end

function menu_help_Callback(hObject, eventdata, handles)
lblCore = handles.labelerObj.lblCore;
if isempty(lblCore)
  h = 'Please open a movie first.';
else
  h = lblCore.getLabelingHelp();
end
msgbox(h,'Labeling Actions','help','modal');

function menu_setup_sequential_mode_Callback(hObject, eventdata, handles)
handles.labelerObj.labelingInit('labelMode',LabelMode.SEQUENTIAL);
function menu_setup_template_mode_Callback(hObject, eventdata, handles)
handles.labelerObj.labelingInit('labelMode',LabelMode.TEMPLATE);
function menu_setup_highthroughput_mode_Callback(hObject, eventdata, handles)
handles.labelerObj.labelingInit('labelMode',LabelMode.HIGHTHROUGHPUT);
function menu_setup_tracking_correction_mode_Callback(hObject, eventdata, handles)
handles.labelerObj.labelingInit('labelMode',LabelMode.ERRORCORRECT);
function menu_setup_multiview_calibrated_mode_Callback(hObject, eventdata, handles)
handles.labelerObj.labelingInit('labelMode',LabelMode.MULTIVIEWCALIBRATED);
function menu_setup_set_labeling_point_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
%npts = lObj.nLabelPoints;
ipt = lObj.lblCore.iPoint;
ret = inputdlg('Select labeling point','Point number',1,{num2str(ipt)});
if isempty(ret)
  return;
end
ret = str2double(ret{1});
lObj.lblCore.setIPoint(ret);
function menu_setup_load_calibration_file_Callback(hObject, eventdata, handles)
lastCalFile = RC.getprop('lastCalibrationFile');
if isempty(lastCalFile)
  lastCalFile = pwd;
end
[fname,pth] = uigetfile('*.mat','Load Calibration File',lastCalFile);
if isequal(fname,0)
  return;
end
fname = fullfile(pth,fname);
handles.labelerObj.labelLoadCalibrationFileRaw(fname);
RC.saveprop('lastCalibrationFile',fname);

function menu_setup_unlock_all_frames_Callback(hObject, eventdata, handles)
handles.labelerObj.labelPosSetAllMarked(false);
function menu_setup_lock_all_frames_Callback(hObject, eventdata, handles)
handles.labelerObj.labelPosSetAllMarked(true);

function CloseImContrast(labelerObj)
labelerObj.videoSetContrastFromAxesCurr();

function menu_view_adjustbrightness_Callback(hObject, eventdata, handles)
hConstrast = imcontrast_kb(handles.axes_curr);
addlistener(hConstrast,'ObjectBeingDestroyed',@(s,e) CloseImContrast(handles.labelerObj));
function menu_view_converttograyscale_Callback(hObject, eventdata, handles)
tf = ~strcmp(hObject.Checked,'on');
lObj = handles.labelerObj;
lObj.movieForceGrayscale = tf;
if lObj.hasMovie
  % Pure convenience: update image for user rather than wait for next 
  % frame-switch. Could also put this in Labeler.set.movieForceGrayscale.
  lObj.setFrame(lObj.currFrame,'tfforcereadmovie',true);
end
function menu_view_gammacorrect_Callback(hObject, eventdata, handles)
val = inputdlg('Gamma value:','Gamma correction');
if isempty(val)
  return;
end
val = str2double(val{1});
handles.labelerObj.videoApplyGammaGrayscale(val);

function menu_file_quit_Callback(hObject, eventdata, handles)
CloseGUI(handles);

function menu_view_trajectories_showall_Callback(hObject, eventdata, handles)
handles.labelerObj.setShowTrxMode(ShowTrxMode.ALL);
function menu_view_trajectories_showcurrent_Callback(hObject, eventdata, handles)
handles.labelerObj.setShowTrxMode(ShowTrxMode.CURRENT);
function menu_view_trajectories_dontshow_Callback(hObject, eventdata, handles)
handles.labelerObj.setShowTrxMode(ShowTrxMode.NONE);
function menu_view_trajectories_centervideoontarget_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.movieCenterOnTarget = ~lObj.movieCenterOnTarget;
function menu_view_flip_flipud_movie_only_Callback(hObject, eventdata, handles)
handles.labelerObj.videoFlipUDVidOnly();
function menu_view_flip_flipud_Callback(hObject, eventdata, handles)
handles.labelerObj.videoFlipUD();
function menu_view_flip_fliplr_Callback(hObject, eventdata, handles)
handles.labelerObj.videoFlipLR();
function menu_view_hide_labels_Callback(hObject, eventdata, handles)
lblCore = handles.labelerObj.lblCore;
if ~isempty(lblCore)
  lblCore.labelsHideToggle();
end

function menu_track_setparametersfile_Callback(hObject, eventdata, handles)
prmFile = RC.getprop('lastCPRParamFile');
if isempty(prmFile)
  prmFile = pwd;
end
[f,p] = uigetfile('*.yaml','Select CPR tracking parameters file',prmFile);
if isequal(f,0)
  return;
end
prmFile = fullfile(p,f);
RC.saveprop('lastCPRParamFile',prmFile);
handles.labelerObj.setTrackParamFile(prmFile);
function menu_track_select_training_data_Callback(hObject, eventdata, handles)
handles.labelerObj.tracker.trnDataSelect();

function menu_track_savetrackingresults_Callback(hObject, eventdata, handles)
handles.labelerObj.trackSaveResultsAs();
function menu_track_loadtrackingresults_Callback(hObject, eventdata, handles)
handles.labelerObj.trackLoadResultsAs();

function menu_track_retrain_Callback(hObject, eventdata, handles)
handles.labelerObj.trackRetrain();

function figure_CloseRequestFcn(hObject, eventdata, handles)
CloseGUI(handles);

function CloseGUI(handles)
if hlpSave(handles.labelerObj)
  delete(handles.figure);
  delete(handles.labelerObj);
end
