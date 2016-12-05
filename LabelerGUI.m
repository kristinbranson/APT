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

% Last Modified by GUIDE v2.5 28-Sep-2016 10:14:28

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

function LabelerGUI_OpeningFcn(hObject,eventdata,handles,varargin) 

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

handles.menu_setup_set_nframe_skip = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_set_nframe_skip_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set Frame Increment',...
  'Tag','menu_setup_set_nframe_skip',...
  'Checked','off',...
  'Visible','off');
moveMenuItemAfter(handles.menu_setup_set_nframe_skip,...
  handles.menu_setup_set_labeling_point);

% add menu item for hiding prediction markers
handles.menu_view_hide_predictions = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_hide_predictions_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Hide predictions',...
  'Tag','menu_view_hide_predictions',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_hide_predictions,handles.menu_view_hide_labels);

handles.menu_view_show_tick_labels = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_tick_labels_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show tick labels',...
  'Tag','menu_view_show_tick_labels',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_tick_labels,handles.menu_view_hide_predictions);
handles.menu_view_show_grid = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_grid_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show grid',...
  'Tag','menu_view_show_grid',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_grid,handles.menu_view_show_tick_labels);
handles.menu_view_show_3D_axes = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_3D_axes_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show/Refresh 3D world axes',...
  'Tag','menu_view_show_3D_axes',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_3D_axes,handles.menu_view_show_grid);

handles.menu_track_export_base = uimenu('Parent',handles.menu_track,...
  'Label','Export current tracking results',...
  'Tag','menu_track_export_base');  
handles.menu_track_export_current_movie = uimenu('Parent',handles.menu_track_export_base,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_export_current_movie_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Current movie only',...
  'Tag','menu_track_export_current_movie');  
handles.menu_track_export_all_movies = uimenu('Parent',handles.menu_track_export_base,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_export_all_movies_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','All movies',...
  'Tag','menu_track_export_all_movies');  

% add menu item for setting current frames labels to tracked positions
handles.menu_track_set_labels = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_set_labels_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set manual labels to predicted pose',...
  'Tag','menu_track_set_labels');  

% MultiViewCalibrated2 labelmode
handles.menu_setup_multiview_calibrated_mode_2 = uimenu(...
  'Parent',handles.menu_labeling_setup,...
  'Label','Multiview Calibrated mode 2',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_multiview_calibrated_mode_2_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_setup_multiview_calibrated_mode_2');  
moveMenuItemAfter(handles.menu_setup_multiview_calibrated_mode_2,...
  handles.menu_setup_multiview_calibrated_mode);

% misc labelmode/Setup menu
LABELMODE_SETUPMENU_MAP = ...
  {LabelMode.NONE '';
   LabelMode.SEQUENTIAL 'menu_setup_sequential_mode';
   LabelMode.TEMPLATE 'menu_setup_template_mode';
   LabelMode.HIGHTHROUGHPUT 'menu_setup_highthroughput_mode';
   LabelMode.ERRORCORRECT 'menu_setup_tracking_correction_mode';
   LabelMode.MULTIVIEWCALIBRATED 'menu_setup_multiview_calibrated_mode';
   LabelMode.MULTIVIEWCALIBRATED2 'menu_setup_multiview_calibrated_mode_2'};
tmp = LABELMODE_SETUPMENU_MAP;
tmp(:,1) = cellfun(@char,tmp(:,1),'uni',0);
tmp(2:end,2) = cellfun(@(x)handles.(x),tmp(2:end,2),'uni',0);
tmp = tmp';
handles.labelMode2SetupMenu = struct(tmp{:});
tmp = LABELMODE_SETUPMENU_MAP(2:end,[2 1]);
tmp = tmp';
handles.setupMenu2LabelMode = struct(tmp{:});

hold(handles.axes_occ,'on');
axis(handles.axes_occ,'ij');

handles.image_curr = imagesc(0,'Parent',handles.axes_curr);
set(handles.image_curr,'hittest','off');
%axisoff(handles.axes_curr);
hold(handles.axes_curr,'on');
set(handles.axes_curr,'Color',[0 0 0]);
handles.image_prev = imagesc(0,'Parent',handles.axes_prev);
set(handles.image_prev,'hittest','off');
%axisoff(handles.axes_prev);
hold(handles.axes_prev,'on');
set(handles.axes_prev,'Color',[0 0 0]);

handles.figs_all = handles.figure;
handles.axes_all = handles.axes_curr;
handles.images_all = handles.image_curr;

linkaxes([handles.axes_prev,handles.axes_curr]);

lObj = handles.labelerObj;

handles.labelTLInfo = InfoTimeline(lObj,handles.axes_timeline_manual);
set(handles.pumInfo,'String',handles.labelTLInfo.getProps());

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
listeners{end+1,1} = addlistener(lObj,'movieInvert','PostSet',@cbkMovieInvertChanged);
listeners{end+1,1} = addlistener(lObj,'lblCore','PostSet',@cbkLblCoreChanged);
listeners{end+1,1} = addlistener(lObj,'newProject',@cbkNewProject);
listeners{end+1,1} = addlistener(lObj,'newMovie',@cbkNewMovie);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'selectModeOn','PostSet',@cbklabelTLInfoSelectModeOn);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props','PostSet',@cbklabelTLInfoPropsUpdated);
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
  'movieForceGrayscale' 
  'movieInvert'};

set(handles.output,'Toolbar','figure');

colnames = handles.labelerObj.TBLTRX_STATIC_COLSTBL;
set(handles.tblTrx,'ColumnName',colnames,'Data',cell(0,numel(colnames)));
colnames = handles.labelerObj.TBLFRAMES_COLS; % AL: dumb b/c table update code uses hardcoded cols 
set(handles.tblFrames,'ColumnName',colnames,'Data',cell(0,numel(colnames)));

% Set the size of gui slightly smaller than screen size.
scsz = get(groot,'Screensize');
set(hObject,'Units','Pixels');
fsz = get(hObject,'Position');
fsz(1) = max(25,fsz(1));
fsz(2) = max(25,fsz(2));
fsz(3) = min(fsz(3),round( (scsz(3)-fsz(1))*0.9));
fsz(4) = min(fsz(4),round( (scsz(4)-fsz(2))*0.9));
set(hObject,'Position',fsz);
set(hObject,'Units','normalized');

handles.depHandles = gobjects(0,1);

guidata(hObject, handles);

% UIWAIT makes LabelerGUI wait for user response (see UIRESUME)
% uiwait(handles.figure);

function varargout = LabelerGUI_OutputFcn(hObject, eventdata, handles) %#ok<*INUSL>
varargout{1} = handles.output;

function handles = addDepHandle(hFig,h)

handles = guidata(hFig);
assert(handles.figure==hFig);

% GC dead handles
tfValid = arrayfun(@isvalid,handles.depHandles);
handles.depHandles = handles.depHandles(tfValid,:);
    
tfSame = arrayfun(@(x)x==h,handles.depHandles);
if ~any(tfSame)
  handles.depHandles(end+1,1) = h;
end

guidata(hFig,handles);

function handles = setShortcuts(handles)

prefs = handles.labelerObj.projPrefs;
if ~isfield(prefs,'Shortcuts')
  return;
end
prefs = prefs.Shortcuts;
fns = fieldnames(prefs);
ismenu = false(1,numel(fns));
for i = 1:numel(fns)
  h = findobj(handles.figure,'Tag',fns{i},'-property','Accelerator');
  if isempty(h) || ~ishandle(h)
    continue;
  end
  ismenu(i) = true;
  set(h,'Accelerator',prefs.(fns{i}));
end

handles.shortcutkeys = cell(1,nnz(~ismenu));
handles.shortcutfns = cell(1,nnz(~ismenu));
idxnotmenu = find(~ismenu);

for ii = 1:numel(idxnotmenu)
  i = idxnotmenu(ii);
  handles.shortcutkeys{ii} = prefs.(fns{i});
  handles.shortcutfns{ii} = fns{i};
end

function cbkAuxAxResize(src,data)
% AL 20160628: voodoo that may help make points more clickable. Sometimes
% pt clickability in MultiViewCalibrated mode is unstable (eg to anchor
% points etc)
ax = findall(src,'type','axes');
axis(ax,'image')
axis(ax,'auto');

function cbkAuxFigCloseReq(src,data,lObj)

handles = lObj.gdata;
if ~any(src==handles.depHandles)
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
handles = lblCore.labeler.gdata;
handles.menu_view_hide_labels.Checked = onIff(lblCore.hideLabels);

function cbkTrackerHideVizChanged(src,evt,hmenu_view_hide_predictions)
tracker = evt.AffectedObject;
hmenu_view_hide_predictions.Checked = onIff(tracker.hideViz);

function cbkKPF(src,evt,lObj)

tfKPused = false;

handles = guidata(src);
% KB20160724: shortcuts from preferences
if all(isfield(handles,{'shortcutkeys','shortcutfns'}))
  % control key pressed?
  if ismember('control',evt.Modifier) && numel(evt.Modifier) == 1 && any(strcmpi(evt.Key,handles.shortcutkeys))
    i = find(strcmpi(evt.Key,handles.shortcutkeys),1);
    h = findobj(handles.figure,'Tag',handles.shortcutfns{i},'-property','Callback');
    if isempty(h)
      fprintf('Unknown shortcut handle %s\n',handles.shortcutfns{i});
    else
      cb = get(h,'Callback');
      if isa(cb,'function_handle')
        cb(h,[]);
        tfKPused = true;
      elseif iscell(cb)
        cb{1}(cb{2:end});
        tfKPused = true;
      elseif ischar(cb)
        evalin('base',[cb,';']);
        tfKPused = true;
      end
    end
  end  
end

if tfKPused
  return;
end

lcore = lObj.lblCore;
if ~isempty(lcore)
  tfKPused = lcore.kpf(src,evt);
  if tfKPused
    return;
  end
end

% TODO timeline use me
      
function cbkWBMF(src,evt,lObj)
lcore = lObj.lblCore;
if ~isempty(lcore)
  lcore.wbmf(src,evt);
end
lObj.gdata.labelTLInfo.cbkWBMF(src,evt);

function cbkWBUF(src,evt,lObj)
if ~isempty(lObj.lblCore)
  lObj.lblCore.wbuf(src,evt);
end
lObj.gdata.labelTLInfo.cbkWBUF(src,evt);

function cbkNewProject(src,evt)

lObj = src;
handles = lObj.gdata;

% figs, axes, images
deleteValidHandles(handles.figs_all(2:end));
handles.figs_all = handles.figs_all(1);
handles.axes_all = handles.axes_all(1);
handles.images_all = handles.images_all(1);

nview = lObj.nview;
figs = gobjects(1,nview);
axs = gobjects(1,nview);
ims = gobjects(1,nview);
figs(1) = handles.figs_all;
axs(1) = handles.axes_all;
ims(1) = handles.images_all;
set(ims(1),'CData',0); % reset image
for iView=2:nview
  figs(iView) = figure(...
    'CloseRequestFcn',@(s,e)cbkAuxFigCloseReq(s,e,lObj),...
    'Color',figs(1).Color...
    );
  axs(iView) = axes;
  handles = addDepHandle(handles.figure,figs(iView));
  
  ims(iView) = imagesc(0,'Parent',axs(iView));
  set(ims(iView),'hittest','off');
  %axisoff(axs(iView));
  hold(axs(iView),'on');
  set(axs(iView),'Color',[0 0 0]);
end
handles.figs_all = figs;
handles.axes_all = axs;
handles.images_all = ims;

ViewConfig.setCfgOnViews(lObj.projPrefs.View,figs,axs,ims,handles.axes_prev);

arrayfun(@(x)colormap(x,gray),figs);
viewNames = lObj.viewNames;
for i=1:nview
  vname = viewNames{i};
  if isempty(vname)
    figs(i).Name = ''; 
  else
    figs(i).Name = sprintf('View: %s',vname);    
  end
end

% % AL: important to get clickable points. Somehow this jiggers plot
% % lims/scaling/coords so that points are more clickable; otherwise
% % lblCore points in aux axes are impossible to click (eg without zooming
% % way in or other contortions)
% for i=2:numel(figs)
%   figs(i).ResizeFcn = @cbkAuxAxResize;
% end
% %arrayfun(@(x)axis(x,'auto'),ax);

hTmp = findall(handles.figs_all,'-property','KeyPressFcn','-not','Tag','edit_frame');
set(hTmp,'KeyPressFcn',@(src,evt)cbkKPF(src,evt,lObj));
set(handles.figs_all,'WindowButtonMotionFcn',@(src,evt)cbkWBMF(src,evt,lObj));
set(handles.figs_all,'WindowButtonUpFcn',@(src,evt)cbkWBUF(src,evt,lObj));

axis(handles.axes_occ,[0 lObj.nLabelPoints+1 0 2]);

handles = setShortcuts(handles);

% AL: Some init hell, initNewMovie() actually inits mostly project-level stuff 
handles.labelTLInfo.initNewMovie();

if isfield(handles,'movieMgr') && isvalid(handles.movieMgr)
  delete(handles.movieMgr);
end
handles.movieMgr = MovieManager(handles.labelerObj);
handles.movieMgr.Visible = 'off';

guidata(handles.figure,handles);
  
function cbkNewMovie(src,evt)
lObj = src;
handles = lObj.gdata;
movRdrs = lObj.movieReader;
ims = arrayfun(@(x)x.readframe(1),movRdrs,'uni',0);
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(isequal(lObj.nview,numel(ims),numel(hAxs),numel(hIms)));

for iView = 1:lObj.nview	
	set(hIms(iView),'CData',ims{iView});
  
  % Right now we leave axis lims as-is. The large majority of the time,
  % the new movie will have the same size as the old.
end

handles.labelTLInfo.initNewMovie();
handles.labelTLInfo.setLabelsFull();

nframes = lObj.nframes;
sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
set(handles.slider_frame,'Value',0,'SliderStep',sliderstep);

function zoomOutFullView(hAx,hIm)
set(hAx,...
  'XLim',[.5,size(hIm.CData,2)+.5],...
  'YLim',[.5,size(hIm.CData,1)+.5]);
axis(hAx,'image');
zoom(hAx,'reset');

function cbkCurrFrameChanged(src,evt) %#ok<*INUSD>
lObj = evt.AffectedObject;
frm = lObj.currFrame;
nfrm = lObj.nframes;
handles = lObj.gdata;
set(handles.edit_frame,'String',num2str(frm));
sldval = (frm-1)/(nfrm-1);
if isnan(sldval)
  sldval = 0;
end
set(handles.slider_frame,'Value',sldval);
if ~lObj.isinit
  handles.labelTLInfo.setCurrFrame(frm);
end

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

function menuSetupLabelModeHelp(handles,labelMode)
% Set .Checked for menu_setup_<variousLabelModes> based on labelMode
menus = fieldnames(handles.setupMenu2LabelMode);
for m = menus(:)',m=m{1}; %#ok<FXSET>
  handles.(m).Checked = 'off';
end
hMenu = handles.labelMode2SetupMenu.(char(labelMode));
hMenu.Checked = 'on';

function cbkLabelModeChanged(src,evt)
lObj = evt.AffectedObject;
handles = lObj.gdata;
lblMode = lObj.labelMode;
menuSetupLabelModeHelp(handles,lblMode);
switch lblMode
  case LabelMode.SEQUENTIAL
    handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.TEMPLATE
    handles.menu_setup_createtemplate.Visible = 'on';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.HIGHTHROUGHPUT
    handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'on';
    handles.menu_setup_set_nframe_skip.Visible = 'on';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.ERRORCORRECT
    handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'on';
    handles.menu_setup_lock_all_frames.Visible = 'on';
    handles.menu_setup_load_calibration_file.Visible = 'off';
  case {LabelMode.MULTIVIEWCALIBRATED LabelMode.MULTIVIEWCALIBRATED2}
    handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'on';
end

lc = lObj.lblCore;
tfShow3DAxes = ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration;
handles.menu_view_show_3D_axes.Enable = onIff(tfShow3DAxes);

function cbkTargetZoomFacChanged(src,evt)
lObj = evt.AffectedObject;
zf = lObj.targetZoomFac;
set(lObj.gdata.sldZoom,'Value',zf);

function cbkProjNameChanged(src,evt)
lObj = evt.AffectedObject;
handles = lObj.gdata;
pname = lObj.projname;
str = sprintf('Project %s created (unsaved) at %s',pname,datestr(now,16));
set(handles.txStatus,'String',str);
set(handles.txProjectName,'String',pname);

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
handles = lObj.gdata;
set(handles.txMoviename,'String',mname);
if ~isempty(mname)
  str = sprintf('new movie %s at %s',mname,datestr(now,16));
  set(handles.txStatus,'String',str);
  
  % Fragile behavior when loading projects; want project status update to
  % persist and not movie status update. This depends on detailed ordering in 
  % Labeler.projLoad
end

function cbkMovieForceGrayscaleChanged(src,evt)
lObj = evt.AffectedObject;
tf = lObj.movieForceGrayscale;
mnu = lObj.gdata.menu_view_converttograyscale;
mnu.Checked = onIff(tf);

function cbkMovieInvertChanged(src,evt)
lObj = evt.AffectedObject;
figs = lObj.gdata.figs_all;
movInvert = lObj.movieInvert;
viewNames = lObj.viewNames;
for i=1:lObj.nview
  name = viewNames{i};
  if isempty(name)
    name = ''; 
  else
    name = sprintf('View: %s',name);
  end
  if movInvert(i)
    name = [name ' (movie inverted)']; %#ok<AGROW>
  end
  figs(i).Name = name;
end

function cbkSuspScoreChanged(src,evt)
lObj = evt.AffectedObject;
ss = lObj.suspScore;
lObj.currImHud.updateReadoutFields('hasSusp',~isempty(ss));

handles = lObj.gdata;
pnlSusp = handles.pnlSusp;
tblSusp = handles.tblSusp;
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
handles = lObj.gdata;
handles.menu_view_trajectories_showall.Checked = 'off';
handles.menu_view_trajectories_showcurrent.Checked = 'off';
handles.menu_view_trajectories_dontshow.Checked = 'off';
switch lObj.showTrxMode
  case ShowTrxMode.NONE
    handles.menu_view_trajectories_dontshow.Checked = 'on';
  case ShowTrxMode.CURRENT
    handles.menu_view_trajectories_showcurrent.Checked = 'on';
  case ShowTrxMode.ALL
    handles.menu_view_trajectories_showall.Checked = 'on';
end

function cbkTrackerChanged(src,evt)
lObj = evt.AffectedObject;
tf = ~isempty(lObj.tracker);
onOff = onIff(tf);
handles = lObj.gdata;
handles.menu_track.Enable = onOff;
handles.pbTrain.Enable = onOff;
handles.pbTrack.Enable = onOff;
if tf
  lObj.tracker.addlistener('hideViz','PostSet',@(src1,evt1) cbkTrackerHideVizChanged(src1,evt1,handles.menu_view_hide_predictions));
end

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
mnu = lObj.gdata.menu_view_trajectories_centervideoontarget;
mnu.Checked = onIff(tf);

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
handles.labelTLInfo.selectModeOn = hObject.Value;

function cbklabelTLInfoSelectModeOn(src,evt)
lblTLObj = evt.AffectedObject;
tb = lblTLObj.lObj.gdata.tbTLSelectMode;
tb.Value = lblTLObj.selectModeOn;

function cbklabelTLInfoPropsUpdated(src,evt)
% Update the props dropdown menu and timeline.
handles = guidata(src);
props = handles.labelTLInfo.getProps();
set(handles.pumInfo,'String',props);


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
  
  cfg = Labeler.cfgGetLastProjectConfigNoView;
  if cfg.NumViews>1
    warndlg('Your last project had multiple views. Opening movie with single view.');
    cfg.NumViews = 1;
    cfg.ViewNames = cfg.ViewNames(1);
    cfg.View = cfg.View(1);
  end
  lm = LabelMode.(cfg.LabelMode);
  if lm.multiviewOnly
    cfg.LabelMode = char(LabelMode.TEMPLATE);
  end
  
  lObj.initFromConfig(cfg);
    
  [~,projName,~] = fileparts(movfile);
  lObj.projNew(projName);
  lObj.movieAdd(movfile,trxfile);
  lObj.movieSet(1);      
end
function menu_file_new_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  cfg = ProjectSetup(handles.figure);
  if ~isempty(cfg)    
    lObj.initFromConfig(cfg);
    lObj.projNew(cfg.ProjectName);
    handles = lObj.gdata; % initFromConfig, projNew have updated handles
    menu_file_managemovies_Callback([],[],handles);
  end  
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

function menu_file_managemovies_Callback(~,~,handles)
if isfield(handles,'movieMgr')
  handles.movieMgr.Visible = 'on';
  figure(handles.movieMgr);
else
  error('LabelerGUI:movieMgr','Please create or load a project.');
end

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
handles.labelerObj.labelImportTrkPrompt(iMov);

function menu_file_export_labels_trks_Callback(hObject, eventdata, handles)
handles.labelerObj.labelExportTrk();

function menu_help_Callback(hObject, eventdata, handles)

function menu_help_labeling_actions_Callback(hObject, eventdata, handles)

lblCore = handles.labelerObj.lblCore;
if isempty(lblCore)
  h = 'Please open a movie first.';
else
  h = lblCore.getLabelingHelp();
end
msgbox(h,'Labeling Actions','help','modal');


function menu_setup_sequential_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_template_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_highthroughput_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_tracking_correction_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_multiview_calibrated_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_multiview_calibrated_mode_2_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menuSetupLabelModeCbkGeneric(hObject,handles)
lblMode = handles.setupMenu2LabelMode.(hObject.Tag);
handles.labelerObj.labelingInit('labelMode',lblMode);

function menu_setup_set_nframe_skip_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lc = lObj.lblCore;
assert(isa(lc,'LabelCoreHT'));
nfs = lc.nFrameSkip;
ret = inputdlg('Select labeling frame increment','Set increment',1,{num2str(nfs)});
if isempty(ret)
  return;
end
val = str2double(ret{1});
lc.nFrameSkip = val;
lObj.labelPointsPlotInfo.HighThroughputMode.NFrameSkip = val;
% This state is duped between labelCore and lppi b/c the lifetimes are
% different. LabelCore exists only between movies etc, and is initted from
% lppi. Hmm

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

function CloseImContrast(lObj,iAxRead,iAxApply)
% ReadClim from axRead and apply to axApply

axAll = lObj.gdata.axes_all;
axRead = axAll(iAxRead);
axApply = axAll(iAxApply);
tfApplyAxPrev = any(iAxApply==1); % axes_prev mirrors axes_curr

clim = get(axRead,'CLim');
if isempty(clim)
	% none; can occur when Labeler is closed
else
	warnst = warning('off','MATLAB:graphicsversion:GraphicsVersionRemoval');
	set(axApply,'CLim',clim);
	warning(warnst);
	if tfApplyAxPrev
		set(lObj.gdata.axes_prev,'CLim',clim);
	end
% 	lObj.minv(iAxApply) = clim(1);
% 	lObj.maxv(iAxApply) = clim(2);
end		

function [tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(handles)
lObj = handles.labelerObj;
if ~lObj.isMultiView
	tfproceed = 1;
	iAxRead = 1;
	iAxApply = 1;
else
  fignames = {handles.figs_all.Name}';
  for iFig = 1:numel(fignames)
    if isempty(fignames{iFig})
      fignames{iFig} = sprintf('<unnamed view %d>',iFig);
    end
  end
  opts = [{'All views together'}; fignames];
  [sel,tfproceed] = listdlg(...
    'PromptString','Select view(s) to adjust',...
    'ListString',opts,...
    'SelectionMode','single');
  if tfproceed
    switch sel
      case 1
        iAxRead = 1;
        iAxApply = 1:numel(handles.axes_all);
      otherwise
        iAxRead = sel-1;
        iAxApply = sel-1;
    end
  else
    iAxRead = nan;
    iAxApply = nan;
  end
end

function menu_view_adjustbrightness_Callback(hObject, eventdata, handles)
[tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  try
  	hConstrast = imcontrast_kb(handles.axes_all(iAxRead));
  catch ME
    switch ME.identifier
      case 'images:imcontrast:unsupportedImageType'
        error(ME.identifier,'%s %s',ME.message,'Try View > Convert to grayscale.');
      otherwise
        ME.rethrow();
    end
  end
	addlistener(hConstrast,'ObjectBeingDestroyed',...
		@(s,e) CloseImContrast(handles.labelerObj,iAxRead,iAxApply));
end
  
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
[tfok,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if ~tfok
	return;
end
val = inputdlg('Gamma value:','Gamma correction');
if isempty(val)
  return;
end
gamma = str2double(val{1});
ViewConfig.applyGammaCorrection(handles.images_all,handles.axes_all,...
  handles.axes_prev,iAxApply,gamma);
		
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
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  lObj = handles.labelerObj;
  lObj.movieInvert(iAxApply) = ~lObj.movieInvert(iAxApply);
  if lObj.hasMovie
    lObj.setFrame(lObj.currFrame,'tfforcereadmovie',true);
  end
end
function menu_view_flip_flipud_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  for iAx = iAxApply(:)'
    ax = handles.axes_all(iAx);
    ax.YDir = toggleAxisDir(ax.YDir);
    if ax==handles.axes_curr
      ax2 = handles.axes_prev;
      ax2.YDir = toggleAxisDir(ax2.YDir);
    end
  end
end
function menu_view_flip_fliplr_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  for iAx = iAxApply(:)'
    ax = handles.axes_all(iAx);
    ax.XDir = toggleAxisDir(ax.XDir);
    if ax==handles.axes_curr
      ax2 = handles.axes_prev;
      ax2.XDir = toggleAxisDir(ax2.XDir);
    end
  end
end
function menu_view_fit_entire_image_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms);

function menu_view_reset_views_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
viewCfg = lObj.projPrefs.View;
ViewConfig.setCfgOnViews(viewCfg,handles.figs_all,handles.axes_all,...
  handles.images_all,handles.axes_prev);
movInvert = ViewConfig.getMovieInvert(viewCfg);
lObj.movieInvert = movInvert;
function menu_view_hide_labels_Callback(hObject, eventdata, handles)
lblCore = handles.labelerObj.lblCore;
if ~isempty(lblCore)
  lblCore.labelsHideToggle();
end

function menu_view_hide_predictions_Callback(hObject, eventdata, handles)
tracker = handles.labelerObj.tracker;
if ~isempty(tracker)
  tracker.hideVizToggle();
end

function menu_view_show_tick_labels_Callback(hObject, eventdata, handles)
% just use checked state of menu for now, no other state
toggleOnOff(hObject,'Checked');
hlpTickGrid(handles);
function menu_view_show_grid_Callback(hObject, eventdata, handles)
% just use checked state of menu for now, no other state
toggleOnOff(hObject,'Checked');
hlpTickGrid(handles);
function hlpTickGrid(handles)
tfTickOn = strcmp(handles.menu_view_show_tick_labels.Checked,'on');
tfGridOn = strcmp(handles.menu_view_show_grid.Checked,'on');

if tfTickOn || tfGridOn
  set(handles.axes_all,'XTickMode','auto','YTickMode','auto');
else
  set(handles.axes_all,'XTick',[],'YTick',[]);
end
if tfTickOn
  set(handles.axes_all,'XTickLabelMode','auto','YTickLabelMode','auto');
else
  set(handles.axes_all,'XTickLabel',[],'YTickLabel',[]);
end
if tfGridOn
  arrayfun(@(x)grid(x,'on'),handles.axes_all);
else
  arrayfun(@(x)grid(x,'off'),handles.axes_all);
end

function menu_view_show_3D_axes_Callback(hObject,eventdata,handles)
if isfield(handles,'hShow3D')
  deleteValidHandles(handles.hShow3D);
end
handles.hShow3D = gobjects(0,1);

tfHide = strcmp(hObject.Checked,'on');

if tfHide
  hObject.Checked = 'off';
else
  lObj = handles.labelerObj;
  lc = lObj.lblCore;
  if ~( ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration )
    error('LabelerGUI:multiView',...
      'Labeling mode must support multiple, calibrated views.');
  end
  vcd = lObj.viewCalibrationData;
  if isempty(vcd)
    error('LabelerGUI:vcd','No view calibration data set.');
  end
  % Hmm, is this weird, getting the vcd off Labeler not LabelCore. They
  % should match however
  assert(isa(vcd,'CalRig'));
  crig = vcd;

  nview = lObj.nview;
  for iview=1:nview
    ax = handles.axes_all(iview);

    VIEWDISTFRAC = 5;

    % Start from where we want the 3D axes to be located in the view
    xl = ax.XLim;
    yl = ax.YLim;
    x0 = diff(xl)/VIEWDISTFRAC+xl(1);
    y0 = diff(yl)/VIEWDISTFRAC+yl(1);

    % Project out into 3D; pick a pt
    [u_p,v_p,w_p] = crig.reconstruct2d(x0,y0,iview);
    RECON_T = 5; % don't know units here
    u0 = u_p(1)+RECON_T*u_p(2);
    v0 = v_p(1)+RECON_T*v_p(2);
    w0 = w_p(1)+RECON_T*w_p(2);

    % Loop and find the scale where the the maximum projected length is ~
    % 1/8th the current view
    SCALEMIN = 0;
    SCALEMAX = 20;
    SCALEN = 300;
    avViewSz = (diff(xl)+diff(yl))/2;
    tgtDX = avViewSz/VIEWDISTFRAC*.8;  
    scales = linspace(SCALEMIN,SCALEMAX,SCALEN);
    for iScale = 1:SCALEN
      % origin is (u0,v0,w0) in 3D; (x0,y0) in 2D

      s = scales(iScale);    
      [x1,y1] = crig.project3d(u0+s,v0,w0,iview);
      [x2,y2] = crig.project3d(u0,v0+s,w0,iview);
      [x3,y3] = crig.project3d(u0,v0,w0+s,iview);
      d1 = sqrt( (x1-x0).^2 + (y1-y0).^2 );
      d2 = sqrt( (x2-x0).^2 + (y2-y0).^2 );
      d3 = sqrt( (x3-x0).^2 + (y3-y0).^2 );
      if d1>tgtDX || d2>tgtDX || d3>tgtDX
        fprintf(1,'Found scale for t=%.2f: %.2f\n',RECON_T,s);
        break;
      end
    end

    LINEWIDTH = 2;
    FONTSIZE = 12;
    handles.hShow3D(end+1,1) = plot(ax,[x0 x1],[y0 y1],'r-','LineWidth',LINEWIDTH);
    handles.hShow3D(end+1,1) = text(x1,y1,'x','Color',[1 0 0],...
      'fontweight','bold','fontsize',FONTSIZE,'parent',ax);
    handles.hShow3D(end+1,1) = plot(ax,[x0 x2],[y0 y2],'g-','LineWidth',LINEWIDTH);
    handles.hShow3D(end+1,1) = text(x2,y2,'y','Color',[0 1 0],...
      'fontweight','bold','fontsize',FONTSIZE,'parent',ax);
    handles.hShow3D(end+1,1) = plot(ax,[x0 x3],[y0 y3],'y-','LineWidth',LINEWIDTH);
    handles.hShow3D(end+1,1) = text(x3,y3,'z','Color',[1 1 0],...
      'fontweight','bold','fontsize',FONTSIZE,'parent',ax);
  end
  hObject.Checked = 'on';
end
guidata(hObject,handles);

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

function menu_track_retrain_Callback(hObject, eventdata, handles)
handles.labelerObj.trackRetrain();

function menu_track_track_and_export_Callback(hObject, eventdata, handles)
tm = getTrackMode(handles);
handles.labelerObj.trackAndExport(tm);

function menu_track_export_current_movie_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
iMov = lObj.currMovie;
if iMov==0
  error('LabelerGUI:noMov','No movie currently set.');
end
lObj.trackExportResults(iMov);

function menu_track_export_all_movies_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
nMov = lObj.nmovies;
if nMov==0
  error('LabelerGUI:noMov','No movies in project.');
end
lObj.trackExportResults(1:nMov);

function menu_track_set_labels_Callback(hObject,eventdata,handles)
xy = handles.labelerObj.tracker.getCurrentPrediction();
if any(isnan(xy(:))),
  fprintf('No predictions for current frame, not labeling.\n');
  return;
end
disp(xy);
handles.labelerObj.lblCore.assignLabelCoords(xy);

function figure_CloseRequestFcn(hObject, eventdata, handles)
CloseGUI(handles);

function CloseGUI(handles)
if hlpSave(handles.labelerObj)
  tfValid = arrayfun(@isvalid,handles.depHandles);
  hValid = handles.depHandles(tfValid);
  arrayfun(@delete,hValid);
  handles.depHandles = gobjects(0,1);
  if isfield(handles,'movieMgr') && ~isempty(handles.movieMgr) ...
      && isvalid(handles.movieMgr)
    delete(handles.movieMgr);
  end
  
  delete(handles.figure);
  delete(handles.labelerObj);
end

function pumInfo_Callback(hObject, eventdata, handles)
contents = cellstr(get(hObject,'String'));
cprop = contents{get(hObject,'Value')};
handles.labelTLInfo.setCurProp(cprop);

function pumInfo_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
