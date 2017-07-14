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

% Last Modified by GUIDE v2.5 16-Jun-2017 15:12:09

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
if nargin && ischar(varargin{1}) && exist(varargin{1}), %#ok<EXIST>
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

hObject.Name = 'APT';

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

handles.menu_setup_streamlined = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_streamlined_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Streamlined',...
  'Tag','menu_setup_streamlined',...
  'Checked','off',...
  'Separator','on',...
  'Visible','off');
moveMenuItemAfter(handles.menu_setup_streamlined,...
  handles.menu_setup_set_labeling_point);

handles.menu_view_rotate_video_target_up = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_rotate_video_target_up_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Rotate video so target always points up',...
  'Tag','menu_view_rotate_video_target_up',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_rotate_video_target_up,...
  handles.menu_view_trajectories_centervideoontarget);

handles.menu_view_hide_predictions = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_hide_predictions_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Hide predictions',...
  'Tag','menu_view_hide_predictions',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_hide_predictions,handles.menu_view_hide_labels);
handles.menu_view_hide_imported_predictions = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_hide_imported_predictions_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Hide imported predictions',...
  'Tag','menu_view_hide_imported_predictions',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_hide_imported_predictions,handles.menu_view_hide_predictions);

handles.menu_view_show_replicates = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_replicates_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show predicted replicates',...
  'Tag','menu_view_show_replicates',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_replicates,handles.menu_view_hide_imported_predictions);
handles.menu_view_hide_trajectories = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_hide_trajectories_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Hide trajectories',...
  'Tag','menu_view_hide_trajectories',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_hide_trajectories,handles.menu_view_show_replicates);
handles.menu_view_plot_trajectories_current_target_only = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_plot_trajectories_current_target_only_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Plot trajectories only for current target',...
  'Tag','menu_view_plot_trajectories_current_target_only',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_plot_trajectories_current_target_only,...
  handles.menu_view_hide_trajectories);

delete(handles.menu_view_trajectories);

handles.menu_view_show_tick_labels = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_tick_labels_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show tick labels',...
  'Tag','menu_view_show_tick_labels',...
  'Separator','on',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_tick_labels,...
  handles.menu_view_plot_trajectories_current_target_only);
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

handles.menu_track_crossvalidate = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_crossvalidate_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Perform cross validation',...
  'Tag','menu_track_crossvalidate',...
  'Separator','off',...
  'Checked','off');
moveMenuItemAfter(handles.menu_track_crossvalidate,handles.menu_track_retrain);

handles.menu_track_store_full_tracking = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_store_full_tracking_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Store tracking replicates/iterations',...
  'Tag','menu_track_store_full_tracking',...
  'Separator','on',...
  'Checked','off');
moveMenuItemBefore(handles.menu_track_store_full_tracking,handles.menu_track_track_and_export);

handles.menu_track_view_tracking_diagnostics = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_view_tracking_diagnostics_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','View tracking diagnostics',...
  'Tag','menu_track_view_tracking_diagnostics',...
  'Separator','off',...
  'Checked','off');
moveMenuItemAfter(handles.menu_track_view_tracking_diagnostics,handles.menu_track_store_full_tracking);

handles.menu_track_track_and_export.Separator = 'off';
handles.menu_track_setparametersfile.Label = 'Configure tracking parameters';
handles.menu_track_setparametersfile.Callback = @(hObject,eventdata)LabelerGUI('menu_track_setparametersfile_Callback',hObject,eventdata,guidata(hObject));
handles.menu_track_use_all_labels_to_train = uimenu(...
  'Parent',handles.menu_track,...
  'Label','Include all labels in training data',...
  'Tag','menu_track_use_all_labels_to_train',...
  'Separator','on',...
  'Callback',@(h,evtdata)LabelerGUI('menu_track_use_all_labels_to_train_Callback',h,evtdata,guidata(h)));
moveMenuItemAfter(handles.menu_track_use_all_labels_to_train,handles.menu_track_setparametersfile);
handles.menu_track_select_training_data.Label = 'Downsample training data';
moveMenuItemAfter(handles.menu_track_select_training_data,handles.menu_track_use_all_labels_to_train);
handles.menu_track_training_data_montage = uimenu(...
  'Parent',handles.menu_track,...
  'Label','Training Data Montage',...
  'Tag','menu_track_training_data_montage',...
  'Callback',@(h,evtdata)LabelerGUI('menu_track_training_data_montage_Callback',h,evtdata,guidata(h)));
moveMenuItemAfter(handles.menu_track_training_data_montage,handles.menu_track_select_training_data);

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

handles.menu_track_set_labels = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_set_labels_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set manual labels to predicted pose',...
  'Tag','menu_track_set_labels');  

tfBGok = ~isempty(ver('distcomp')) && ~verLessThan('distcomp','6.10');
onoff = onIff(tfBGok);
handles.menu_track_background_predict = uimenu('Parent',handles.menu_track,...
  'Label','Background prediction','Tag','menu_track_background_predict',...
  'Separator','on','Enable',onoff);
moveMenuItemAfter(handles.menu_track_background_predict,...
  handles.menu_track_set_labels);

handles.menu_track_background_predict_start = uimenu(...
  'Parent',handles.menu_track_background_predict,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_background_predict_start_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Start/enable background prediction',...
  'Tag','menu_track_background_predict_start');
handles.menu_track_background_predict_end = uimenu(...
  'Parent',handles.menu_track_background_predict,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_background_predict_end_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Stop background prediction',...
  'Tag','menu_track_background_predict_end');
handles.menu_track_background_predict_stats = uimenu(...
  'Parent',handles.menu_track_background_predict,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_background_predict_stats_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Background prediction stats',...
  'Tag','menu_track_background_predict_stats');

% MultiViewCalibrated2 labelmode
handles.menu_setup_multiview_calibrated_mode_2 = uimenu(...
  'Parent',handles.menu_labeling_setup,...
  'Label','Multiview Calibrated mode 2',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_multiview_calibrated_mode_2_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_setup_multiview_calibrated_mode_2');  
moveMenuItemAfter(handles.menu_setup_multiview_calibrated_mode_2,...
  handles.menu_setup_multiview_calibrated_mode);

delete(handles.menu_setup_multiview_calibrated_mode);
handles.menu_setup_multiview_calibrated_mode = [];
delete(handles.menu_setup_tracking_correction_mode);
handles.menu_setup_tracking_correction_mode = [];

handles.menu_help_about = uimenu(...
  'Parent',handles.menu_help,...
  'Label','About',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_help_about_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_help_about');  
moveMenuItemBefore(handles.menu_help_about,handles.menu_help_labeling_actions);

hCMenu = uicontextmenu('parent',handles.figure);
uimenu('Parent',hCMenu,'Label','Freeze to current main window',...
  'Callback',@(src,evt)cbkFreezePrevAxesToMainWindow(src,evt));
uimenu('Parent',hCMenu,'Label','Display last frame seen in main window',...
  'Callback',@(src,evt)cbkUnfreezePrevAxes(src,evt));
handles.axes_prev.UIContextMenu = hCMenu;

% misc labelmode/Setup menu
LABELMODE_SETUPMENU_MAP = ...
  {LabelMode.NONE '';
   LabelMode.SEQUENTIAL 'menu_setup_sequential_mode';
   LabelMode.TEMPLATE 'menu_setup_template_mode';
   LabelMode.HIGHTHROUGHPUT 'menu_setup_highthroughput_mode';
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
set(handles.axes_occ,'XTick',[],'YTick',[]);

handles.image_curr = imagesc(0,'Parent',handles.axes_curr);
set(handles.image_curr,'hittest','off');
hold(handles.axes_curr,'on');
set(handles.axes_curr,'Color',[0 0 0]);
handles.image_prev = imagesc(0,'Parent',handles.axes_prev);
set(handles.image_prev,'hittest','off');
hold(handles.axes_prev,'on');
set(handles.axes_prev,'Color',[0 0 0]);

handles.figs_all = handles.figure;
handles.axes_all = handles.axes_curr;
handles.images_all = handles.image_curr;

lObj = handles.labelerObj;

handles.labelTLInfo = InfoTimeline(lObj,handles.axes_timeline_manual);
set(handles.pumInfo,'String',handles.labelTLInfo.getPropsDisp());

listeners = cell(0,1);
listeners{end+1,1} = addlistener(handles.slider_frame,'ContinuousValueChange',@slider_frame_Callback);
listeners{end+1,1} = addlistener(handles.sldZoom,'ContinuousValueChange',@sldZoom_Callback);
listeners{end+1,1} = addlistener(handles.axes_curr,'XLim','PostSet',@(s,e)axescurrXLimChanged(s,e,handles));
listeners{end+1,1} = addlistener(handles.axes_curr,'XDir','PostSet',@(s,e)axescurrXDirChanged(s,e,handles));
listeners{end+1,1} = addlistener(handles.axes_curr,'YDir','PostSet',@(s,e)axescurrYDirChanged(s,e,handles));
listeners{end+1,1} = addlistener(lObj,'projname','PostSet',@cbkProjNameChanged);
listeners{end+1,1} = addlistener(lObj,'currFrame','PostSet',@cbkCurrFrameChanged);
listeners{end+1,1} = addlistener(lObj,'currTarget','PostSet',@cbkCurrTargetChanged);
listeners{end+1,1} = addlistener(lObj,'labeledposNeedsSave','PostSet',@cbkLabeledPosNeedsSaveChanged);
listeners{end+1,1} = addlistener(lObj,'labelMode','PostSet',@cbkLabelModeChanged);
listeners{end+1,1} = addlistener(lObj,'labels2Hide','PostSet',@cbkLabels2HideChanged);
%listeners{end+1,1} = addlistener(lObj,'targetZoomRadius','PostSet',@cbkTargetZoomFacChanged);
listeners{end+1,1} = addlistener(lObj,'projFSInfo','PostSet',@cbkProjFSInfoChanged);
listeners{end+1,1} = addlistener(lObj,'moviename','PostSet',@cbkMovienameChanged);
listeners{end+1,1} = addlistener(lObj,'suspScore','PostSet',@cbkSuspScoreChanged);
listeners{end+1,1} = addlistener(lObj,'showTrx','PostSet',@cbkShowTrxChanged);
listeners{end+1,1} = addlistener(lObj,'showTrxCurrTargetOnly','PostSet',@cbkShowTrxCurrTargetOnlyChanged);
listeners{end+1,1} = addlistener(lObj,'tracker','PostSet',@cbkTrackerChanged);
listeners{end+1,1} = addlistener(lObj,'trackNFramesSmall','PostSet',@cbkTrackerNFramesChanged);
listeners{end+1,1} = addlistener(lObj,'trackNFramesLarge','PostSet',@cbkTrackerNFramesChanged);    
listeners{end+1,1} = addlistener(lObj,'trackNFramesNear','PostSet',@cbkTrackerNFramesChanged);
listeners{end+1,1} = addlistener(lObj,'movieCenterOnTarget','PostSet',@cbkMovieCenterOnTargetChanged);
listeners{end+1,1} = addlistener(lObj,'movieRotateTargetUp','PostSet',@cbkMovieRotateTargetUpChanged);
listeners{end+1,1} = addlistener(lObj,'movieForceGrayscale','PostSet',@cbkMovieForceGrayscaleChanged);
listeners{end+1,1} = addlistener(lObj,'movieInvert','PostSet',@cbkMovieInvertChanged);
listeners{end+1,1} = addlistener(lObj,'lblCore','PostSet',@cbkLblCoreChanged);
listeners{end+1,1} = addlistener(lObj,'newProject',@cbkNewProject);
listeners{end+1,1} = addlistener(lObj,'newMovie',@cbkNewMovie);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'selectOn','PostSet',@cbklabelTLInfoSelectOn);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props','PostSet',@cbklabelTLInfoPropsUpdated);
handles.listeners = listeners;

hZ = zoom(hObject);
hZ.ActionPostCallback = @cbkPostZoom;

% These Labeler properties need their callbacks fired to properly init UI.
% Labeler will read .propsNeedInit from the GUIData to comply.
handles.propsNeedInit = {
  'labelMode' 
  'suspScore' 
  'showTrx' 
  'showTrxCurrTargetOnly'
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

handles.sldZoom.Min = 0;
handles.sldZoom.Max = 1;
handles.sldZoom.Value = 0;

handles.depHandles = gobjects(0,1);

handles.isPlaying = false;
handles.pbPlay.CData = Icons.ims.play;
handles.pbPlay.BackgroundColor = handles.edit_frame.BackgroundColor;
handles.pbPlaySeg.CData = Icons.ims.playsegment;
handles.pbPlaySeg.BackgroundColor = handles.edit_frame.BackgroundColor;

handles.pbPlaySeg.TooltipString = 'play nearby frames; labels not updated';

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
  cbkLblCoreHideLabelsChanged([],struct('AffectedObject',lblCore));
  if isprop(lblCore,'streamlined')
    lblCore.addlistener('streamlined','PostSet',@cbkLblCoreStreamlinedChanged);
    cbkLblCoreStreamlinedChanged([],struct('AffectedObject',lblCore));
  end
end

function cbkLblCoreHideLabelsChanged(src,evt)
lblCore = evt.AffectedObject;
handles = lblCore.labeler.gdata;
handles.menu_view_hide_labels.Checked = onIff(lblCore.hideLabels);

function cbkLblCoreStreamlinedChanged(src,evt)
lblCore = evt.AffectedObject;
handles = lblCore.labeler.gdata;
handles.menu_setup_streamlined.Checked = onIff(lblCore.streamlined);

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
%lObj.gdata.labelTLInfo.cbkWBMF(src,evt);

function cbkWBUF(src,evt,lObj)
if ~isempty(lObj.lblCore)
  lObj.lblCore.wbuf(src,evt);
end
%lObj.gdata.labelTLInfo.cbkWBUF(src,evt);

function cbkWSWF(src,evt,lObj)
scrollcnt = evt.VerticalScrollCount;
scrollamt = evt.VerticalScrollAmount;
fcurr = lObj.currFrame;
f = fcurr - round(scrollcnt*scrollamt); % scroll "up" => larger frame number
f = min(max(f,1),lObj.nframes);
cmod = lObj.gdata.figure.CurrentModifier;
tfMod = ~isempty(cmod) && any(strcmp(cmod{1},{'control' 'shift'}));
if tfMod
  if f>fcurr
    lObj.frameUp(true);
  else
    lObj.frameDown(true);
  end
else
  lObj.setFrameProtected(f);
end

function cbkNewProject(src,evt)

lObj = src;
handles = lObj.gdata;

% figs, axes, images
deleteValidHandles(handles.figs_all(2:end));
handles.figs_all = handles.figs_all(1);
handles.axes_all = handles.axes_all(1);
handles.images_all = handles.images_all(1);
handles.axes_occ = handles.axes_occ(1);

nview = lObj.nview;
figs = gobjects(1,nview);
axs = gobjects(1,nview);
ims = gobjects(1,nview);
axsOcc = gobjects(1,nview);
figs(1) = handles.figs_all;
axs(1) = handles.axes_all;
ims(1) = handles.images_all;
axsOcc(1) = handles.axes_occ;

% all occluded-axes will have ratios widthAxsOcc:widthAxs and 
% heightAxsOcc:heightAxs equal to that of axsOcc(1):axs(1)
axsOcc1Pos = axsOcc(1).Position;
ax1Pos = axs(1).Position;
axOccSzRatios = axsOcc1Pos(3:4)./ax1Pos(3:4);
axOcc1XColor = axsOcc(1).XColor;

set(ims(1),'CData',0); % reset image
for iView=2:nview
  figs(iView) = figure(...
    'CloseRequestFcn',@(s,e)cbkAuxFigCloseReq(s,e,lObj),...
    'Color',figs(1).Color,...
    'Menubar','none',...
    'Toolbar','figure',...
    'UserData',struct('view',iView)...
    );
  axs(iView) = axes;
  handles = addDepHandle(handles.figure,figs(iView));
  
  ims(iView) = imagesc(0,'Parent',axs(iView));
  set(ims(iView),'hittest','off');
  %axisoff(axs(iView));
  hold(axs(iView),'on');
  set(axs(iView),'Color',[0 0 0]);
  
  axparent = axs(iView).Parent;
  axpos = axs(iView).Position;
  axunits = axs(iView).Units;
  axpos(3:4) = axpos(3:4).*axOccSzRatios;
  axsOcc(iView) = axes('Parent',axparent,'Position',axpos,'Units',axunits,...
    'Color',[0 0 0],'Box','on','XTick',[],'YTick',[],'XColor',axOcc1XColor,...
    'YColor',axOcc1XColor);
  hold(axsOcc(iView),'on');
  axis(axsOcc(iView),'ij');
end
handles.figs_all = figs;
handles.axes_all = axs;
handles.images_all = ims;
handles.axes_occ = axsOcc;

axis(handles.axes_occ,[0 lObj.nLabelPoints+1 0 2]);

% The link destruction/recreation may not be necessary
if isfield(handles,'hLinkPrevCurr') && isvalid(handles.hLinkPrevCurr)
  delete(handles.hLinkPrevCurr);
end
viewCfg = lObj.projPrefs.View;
handles.newProjAxLimsSetInConfig = hlpSetConfigOnViews(viewCfg,handles,...
  viewCfg(1).CenterOnTarget); % lObj.CenterOnTarget is not set yet
AX_LINKPROPS = {'XLim' 'YLim' 'XDir' 'YDir'};
handles.hLinkPrevCurr = ...
  linkprop([handles.axes_curr,handles.axes_prev],AX_LINKPROPS);

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
% for i=1:numel(axs)
%   zoomOutFullView(axs(i),[],true);
% end

arrayfun(@(x)zoom(x,'off'),handles.figs_all); % Cannot set KPF if zoom or pan is on
arrayfun(@(x)pan(x,'off'),handles.figs_all);
hTmp = findall(handles.figs_all,'-property','KeyPressFcn','-not','Tag','edit_frame');
set(hTmp,'KeyPressFcn',@(src,evt)cbkKPF(src,evt,lObj));
set(handles.figs_all,'WindowButtonMotionFcn',@(src,evt)cbkWBMF(src,evt,lObj));
set(handles.figs_all,'WindowButtonUpFcn',@(src,evt)cbkWBUF(src,evt,lObj));
set(handles.figs_all,'WindowScrollWheelFcn',@(src,evt)cbkWSWF(src,evt,lObj));

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
%movRdrs = lObj.movieReader;
%ims = arrayfun(@(x)x.readframe(1),movRdrs,'uni',0);
hAxs = handles.axes_all;
hIms = handles.images_all; % Labeler has already loaded with first frame
assert(isequal(lObj.nview,numel(hAxs),numel(hIms)));

tfResetAxLims = evt.isFirstMovieOfProject || lObj.movieRotateTargetUp;
tfResetAxLims = repmat(tfResetAxLims,lObj.nview,1);
if isfield(handles,'newProjAxLimsSetInConfig')
  % AL20170520 Legacy projects did not save their axis lims in the .lbl
  % file. 
  tfResetAxLims = tfResetAxLims | ~handles.newProjAxLimsSetInConfig;
  handles = rmfield(handles,'newProjAxLimsSetInConfig');
end
tfResetCLims = evt.isFirstMovieOfProject;

% Deal with Axis and Color limits.
for iView = 1:lObj.nview	  
  % AL20170518. Different scenarios leads to different desired behavior
  % here:
  %
  % 1. User created a new project (without specifying axis lims in the cfg)
  % and added the first movie. Here, ViewConfig.setCfgOnViews should have
  % set .X/YLimMode to 'auto', so that the axis would  rescale for the 
  % first frame to be shown. However, given the practical vagaries of 
  % initialization, it is too fragile to rely on this. Instead, in the case 
  % of a first-movie-of-a-proj, we explicitly zoom the axes out to fit the 
  % image.
  %
  % 2. User changed movies in an existing project (no targets).
  % Here, the user has already set axis limits appropriately so we do not
  % want to touch the axis limits.
  %
  % 3. User changed movies in an existing project with targets and 
  % Center/Rotate Movie on Target is on. 
  % Here, the user will probably appreciate a wide/full view before zooming
  % back into a target.
  %
  % 4. User changed movies in an eixsting project, except the new movie has
  % a different size compared to the previous. CURRENTLY THIS IS
  % 'UNSUPPORTED' ie we don't attempt to make this behave nicely. The vast
  % majority of projects will have movies of a given/fixed size.
  
  if tfResetAxLims(iView)
    zoomOutFullView(hAxs(iView),hIms(iView),true);
  end
  if tfResetCLims
    hAxs(iView).CLimMode = 'auto';
  end
end

handles.labelTLInfo.initNewMovie();
handles.labelTLInfo.setLabelsFull();

nframes = lObj.nframes;
sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
set(handles.slider_frame,'Value',0,'SliderStep',sliderstep);

ifo = lObj.movieInfoAll{lObj.currMovie,1}.info;
minzoomrad = 10;
maxzoomrad = (ifo.nc+ifo.nr)/4;
handles.sldZoom.UserData = log([minzoomrad maxzoomrad]);

TRX_MENUS = {...
  'menu_view_trajectories_centervideoontarget'
  'menu_view_rotate_video_target_up'
  'menu_view_hide_trajectories'
  'menu_view_plot_trajectories_current_target_only'
  'tblTrx'};
onOff = onIff(lObj.hasTrx);
cellfun(@(x)set(handles.(x),'Enable',onOff),TRX_MENUS);

guidata(handles.figure,handles);

function zoomOutFullView(hAx,hIm,resetCamUpVec)
if isequal(hIm,[])
  axis(hAx,'auto');
else
  set(hAx,...
    'XLim',[.5,size(hIm.CData,2)+.5],...
    'YLim',[.5,size(hIm.CData,1)+.5]);
end
axis(hAx,'image');
zoom(hAx,'reset');
if resetCamUpVec
  hAx.CameraUpVectorMode = 'auto';
end
hAx.CameraViewAngleMode = 'auto';
hAx.CameraPositionMode = 'auto';
hAx.CameraTargetMode = 'auto';

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
  handles.labelTLInfo.newFrame(frm);
end

function cbkCurrTargetChanged(src,evt) %#ok<*INUSD>
lObj = evt.AffectedObject;
if lObj.hasTrx && ~lObj.isinit
  id = lObj.currTrxID;
  lObj.currImHud.updateTarget(id);
  lObj.gdata.labelTLInfo.newTarget();
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
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.TEMPLATE
    handles.menu_setup_createtemplate.Visible = 'on';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
  case LabelMode.HIGHTHROUGHPUT
    handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'on';
    handles.menu_setup_set_nframe_skip.Visible = 'on';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
%   case LabelMode.ERRORCORRECT
%     handles.menu_setup_createtemplate.Visible = 'off';
%     handles.menu_setup_set_labeling_point.Visible = 'off';
%     handles.menu_setup_set_nframe_skip.Visible = 'off';
%     handles.menu_setup_streamlined.Visible = 'off';
%     handles.menu_setup_unlock_all_frames.Visible = 'on';
%     handles.menu_setup_lock_all_frames.Visible = 'on';
%     handles.menu_setup_load_calibration_file.Visible = 'off';
  case {LabelMode.MULTIVIEWCALIBRATED2}
    handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'on';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'on';
end

lc = lObj.lblCore;
tfShow3DAxes = ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration;
handles.menu_view_show_3D_axes.Enable = onIff(tfShow3DAxes);

% function cbkTargetZoomFacChanged(src,evt)
% lObj = evt.AffectedObject;
% zf = lObj.targetZoomFac;
% set(lObj.gdata.sldZoom,'Value',zf);

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

function cbkTrackerChanged(src,evt)
lObj = evt.AffectedObject;
tf = ~isempty(lObj.tracker);
onOff = onIff(tf);
handles = lObj.gdata;
handles.menu_track.Enable = onOff;
handles.pbTrain.Enable = onOff;
handles.pbTrack.Enable = onOff;
handles.menu_view_hide_predictions.Enable = onOff;
if tf
  lObj.tracker.addlistener('hideViz','PostSet',@(src1,evt1) cbkTrackerHideVizChanged(src1,evt1,handles.menu_view_hide_predictions));
  lObj.tracker.addlistener('trnDataDownSamp','PostSet',@(src1,evt1) cbkTrackerTrnDataDownSampChanged(src1,evt1,handles));
  lObj.tracker.addlistener('showVizReplicates','PostSet',@(src1,evt1) cbkTrackerShowVizReplicatesChanged(src1,evt1,handles));
  lObj.tracker.addlistener('storeFullTracking','PostSet',@(src1,evt1) cbkTrackerStoreFullTrackingChanged(src1,evt1,handles));
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

function cbkMovieRotateTargetUpChanged(src,evt)
lObj = evt.AffectedObject;
tf = lObj.movieRotateTargetUp;
if tf
  ax = lObj.gdata.axes_curr;
  warnst = warning('off','LabelerGUI:axDir');
  for f={'XDir' 'YDir'},f=f{1}; %#ok<FXSET>
    if strcmp(ax.(f),'reverse')
      warningNoTrace('LabelerGUI:ax','Setting main axis .%s to ''normal''.',f);
      ax.(f) = 'normal';
    end
  end
  warning(warnst);
end
mnu = lObj.gdata.menu_view_rotate_video_target_up;
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
    tfSetOccurred = lObj.frameUp(true);
  else
    tfSetOccurred = lObj.frameDown(true);
  end
else
  tfSetOccurred = lObj.setFrameProtected(f);
end
  
if ~tfSetOccurred
  sldval = (lObj.currFrame-1)/(lObj.nframes-1);
  if isnan(sldval)
    sldval = 0;
  end
  set(hObject,'Value',sldval);
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
if ~lObj.trxCheckFramesLive(f)
  set(hObject,'String',num2str(lObj.currFrame));
  warnstr = sprintf('Frame %d is out-of-range for current target.',f);
  warndlg(warnstr,'Out of range');
  return;
end
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
wbObj = WaitBarWithCancel('Tracking');
oc = onCleanup(@()delete(wbObj));
handles.labelerObj.track(tm,'wbObj',wbObj);
if wbObj.isCancel
  msg = wbObj.cancelMessage('Tracking canceled');
  msgbox(msg,'Track');
end

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

% Current/last row selection is maintained in hObject.UserData

lObj = handles.labelerObj;
if ~lObj.hasTrx
  return;
end

rows = eventdata.Indices(:,1);
rowsprev = hObject.UserData;
hObject.UserData = rows;
dat = hObject.Data;

if isscalar(rows)
  id = dat{rows(1),1};
  lObj.setTargetID(id);
  lObj.labelsOtherTargetHideAll();
else
  % addon to existing selection
  rowsnew = setdiff(rows,rowsprev);  
  idsnew = cell2mat(dat(rowsnew,1));
  lObj.labelsOtherTargetShowIDs(idsnew);
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

% 20170428
% Notes -- Zooms Views Angles et al
% 
% Zoom.
% In APT we refer to the "zoom" as effective magnification determined by 
% the axis limits, ie how many pixels are shown along x and y. Currently
% the pixels and axis are always square.
% 
% The zoom level can be adjusted in a variety of ways: via the zoom slider,
% the Unzoom button, the manual zoom tools in the toolbar, or 
% View > Zoom out. 
%
% Camroll.
% When Trx are available, the movie can be rotated so that the Trx are
% always at a given orientation (currently, "up"). This is achieved by
% "camrolling" the axes, ie setting axes.CameraUpVector. Currently
% manually camrolling is not available.
%
% CamViewAngle.
% The CameraViewAngle is the AOV of the 'camera' viewing the axes. When
% "camroll" is off (either there are no Trx, or rotateSoTargetIsUp is
% off), axis.CameraViewAngleMode is set to 'auto' and MATLAB selects a
% CameraViewAngle so that the axis fills its outerposition. When camroll is
% on, MATLAB by default chooses a CameraViewAngle that is relatively wide, 
% so that the square axes is very visible as it rotates around. This is a
% bit distracting so currently we choose a smaller CamViewAngle (very 
% arbitrarily). There may be a better way to handle this.

function axescurrXLimChanged(hObject,eventdata,handles)
% log(zoomrad) = logzoomradmax + sldval*(logzoomradmin-logzoomradmax)
ax = eventdata.AffectedObject;
radius = diff(ax.XLim)/2;
hSld = handles.sldZoom;
if ~isempty(hSld.UserData) % empty during init
  userdata = hSld.UserData;
  logzoomradmin = userdata(1);
  logzoomradmax = userdata(2);
  sldval = (log(radius)-logzoomradmax)/(logzoomradmin-logzoomradmax);
  sldval = min(max(sldval,0),1);
  hSld.Value = sldval;
end
function axescurrXDirChanged(hObject,eventdata,handles)
videoRotateTargetUpAxisDirCheckWarn(handles);
function axescurrYDirChanged(hObject,eventdata,handles)
videoRotateTargetUpAxisDirCheckWarn(handles);
function videoRotateTargetUpAxisDirCheckWarn(handles)
ax = handles.axes_curr;
if (strcmp(ax.XDir,'reverse') || strcmp(ax.YDir,'reverse')) && ...
    handles.labelerObj.movieRotateTargetUp
  warningNoTrace('LabelerGUI:axDir',...
    'Main axis ''XDir'' or ''YDir'' is set to ''reverse'' and .movieRotateTargetUp is set. Graphics behavior may be unexpected; proceed at your own risk.');
end

function sldZoom_Callback(hObject, eventdata, ~)
% log(zoomrad) = logzoomradmax + sldval*(logzoomradmin-logzoomradmax)
handles = guidata(hObject);
lObj = handles.labelerObj;
v = hObject.Value;
userdata = hObject.UserData;
logzoomrad = userdata(2)+v*(userdata(1)-userdata(2));
zoomRad = exp(logzoomrad);
lObj.videoZoom(zoomRad);
hlpRemoveFocus(hObject,handles);

function cbkPostZoom(src,evt)
if verLessThan('matlab','R2016a')
  setappdata(src,'manualZoomOccured',true);
end

function pbResetZoom_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms,false(1,numel(hIms)));

function pbSetZoom_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.targetZoomRadiusDefault = diff(handles.axes_curr.XLim)/2;

function pbRecallZoom_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.videoZoom(lObj.targetZoomRadiusDefault);

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
tl = handles.labelTLInfo;
tl.selectOn = hObject.Value;

function pbClearSelection_Callback(hObject, eventdata, handles)
tl = handles.labelTLInfo;
tl.selectClearSelection();

function cbklabelTLInfoSelectOn(src,evt)
lblTLObj = evt.AffectedObject;
tb = lblTLObj.lObj.gdata.tbTLSelectMode;
tb.Value = lblTLObj.selectOn;

function cbklabelTLInfoPropsUpdated(src,evt)
% Update the props dropdown menu and timeline.
handles = guidata(src);
props = handles.labelTLInfo.getPropsDisp();
set(handles.pumInfo,'String',props);

function cbkFreezePrevAxesToMainWindow(src,evt)
handles = guidata(src);
handles.labelerObj.setPrevAxesMode(PrevAxesMode.FROZEN);
function cbkUnfreezePrevAxes(src,evt)
handles = guidata(src);
handles.labelerObj.setPrevAxesMode(PrevAxesMode.LASTSEEN);

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
  lObj.movieSet(1,'isFirstMovie',true);      
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

function menu_file_import_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if ~lObj.hasMovie
  error('LabelerGUI:noMovie','No movie is loaded.');
end
iMov = lObj.currMovie;
lObj.labels2ImportTrkPrompt(iMov);

function [tfok,rawtrkname] = hlpRawtrkname(lObj)
rawtrkname = inputdlg('Enter name/pattern for trkfile(s) to be exported. Available macros: $movdir, $movfile, $projdir, $projfile, $projname.',...
  'Export Trk File',1,{lObj.defaultTrkRawname()});
tfok = ~isempty(rawtrkname);
if tfok
  rawtrkname = rawtrkname{1};
end

function menu_file_export_labels_trks_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
[tfok,rawtrkname] = hlpRawtrkname(lObj);
if ~tfok
  return;
end
lObj.labelExportTrk(1:lObj.nmovies,'rawtrkname',rawtrkname);

function menu_help_Callback(hObject, eventdata, handles)

function menu_help_labeling_actions_Callback(hObject, eventdata, handles)
lblCore = handles.labelerObj.lblCore;
if isempty(lblCore)
  h = 'Please open a movie first.';
else
  h = lblCore.getLabelingHelp();
end
msgbox(h,'Labeling Actions','help');

function menu_help_about_Callback(hObject, eventdata, handles)
str = {'APT: Branson Lab Animal Part Tracker'};
msgbox(str,'About');

function menu_setup_sequential_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_template_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_highthroughput_mode_Callback(hObject,eventdata,handles)
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

function menu_setup_streamlined_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lc = lObj.lblCore;
assert(isa(lc,'LabelCoreMultiViewCalibrated2'));
lc.streamlined = ~lc.streamlined;

function menu_setup_set_labeling_point_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
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

[crObj,tfSetViewSizes] = CalRig.loadCreateCalRigObjFromFile(fname);

lObj = handles.labelerObj;
vcdPW = lObj.viewCalProjWide;
if isempty(vcdPW)
  resp = questdlg('Should calibration apply to i) all movies in project or ii) current movie only?',...
    'Calibration load',...
    'All movies in project',...
    'Current movie only',...
    'Cancel',...
    'All movies in project');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'All movies in project'
      tfProjWide = true;      
    case 'Current movie only'
      tfProjWide = false;      
    otherwise
      return;
  end
else
  tfProjWide = vcdPW;
end

% Currently there is no UI for altering lObj.viewCalProjWide once it is set

if tfProjWide
  lObj.viewCalSetProjWide(crObj,'tfSetViewSizes',tfSetViewSizes);
else
  lObj.viewCalSetCurrMovie(crObj,'tfSetViewSizes',tfSetViewSizes);
end

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

function cbkShowTrxChanged(src,evt)
lObj = evt.AffectedObject;
handles = lObj.gdata;
onOff = onIff(~lObj.showTrx);
handles.menu_view_hide_trajectories.Checked = onOff;
function cbkShowTrxCurrTargetOnlyChanged(src,evt)
lObj = evt.AffectedObject;
handles = lObj.gdata;
onOff = onIff(lObj.showTrxCurrTargetOnly);
handles.menu_view_plot_trajectories_current_target_only.Checked = onOff;
function menu_view_hide_trajectories_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.setShowTrx(~lObj.showTrx);
function menu_view_plot_trajectories_current_target_only_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.setShowTrxCurrTargetOnly(~lObj.showTrxCurrTargetOnly);

function menu_view_trajectories_centervideoontarget_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.movieCenterOnTarget = ~lObj.movieCenterOnTarget;
function menu_view_rotate_video_target_up_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.movieRotateTargetUp = ~lObj.movieRotateTargetUp;
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
  end
end
function menu_view_flip_fliplr_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  for iAx = iAxApply(:)'
    ax = handles.axes_all(iAx);
    ax.XDir = toggleAxisDir(ax.XDir);
%     if ax==handles.axes_curr
%       ax2 = handles.axes_prev;
%       ax2.XDir = toggleAxisDir(ax2.XDir);
%     end
  end
end
function menu_view_fit_entire_image_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms,true(1,numel(hAxs)));

function menu_view_reset_views_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
viewCfg = lObj.projPrefs.View;
hlpSetConfigOnViews(viewCfg,handles,lObj.movieCenterOnTarget);
movInvert = ViewConfig.getMovieInvert(viewCfg);
lObj.movieInvert = movInvert;
lObj.movieCenterOnTarget = viewCfg(1).CenterOnTarget;
lObj.movieRotateTargetUp = viewCfg(1).RotateTargetUp;

function tfAxLimsSpecifiedInCfg = hlpSetConfigOnViews(viewCfg,handles,centerOnTarget)
axs = handles.axes_all;
tfAxLimsSpecifiedInCfg = ViewConfig.setCfgOnViews(viewCfg,handles.figs_all,axs,...
  handles.images_all,handles.axes_prev);
if ~centerOnTarget
  [axs.CameraUpVectorMode] = deal('auto');
  [axs.CameraViewAngleMode] = deal('auto');
  [axs.CameraTargetMode] = deal('auto');
  [axs.CameraPositionMode] = deal('auto');
end
[axs.DataAspectRatio] = deal([1 1 1]);
handles.menu_view_show_tick_labels.Checked = onIff(~isempty(axs(1).XTickLabel));
handles.menu_view_show_grid.Checked = axs(1).XGrid;

function menu_view_hide_labels_Callback(hObject, eventdata, handles)
lblCore = handles.labelerObj.lblCore;
if ~isempty(lblCore)
  lblCore.labelsHideToggle();
end

function menu_view_hide_predictions_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
tracker = lObj.tracker;
if ~isempty(tracker)
  tracker.hideVizToggle();
end

function menu_view_hide_imported_predictions_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.labels2VizToggle();

function cbkTrackerShowVizReplicatesChanged(hObject, eventdata, handles)
handles.menu_view_show_replicates.Checked = ...
  onIff(handles.labelerObj.tracker.showVizReplicates);

function menu_view_show_replicates_Callback(hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
vsr = tObj.showVizReplicates;
vsrnew = ~vsr;
sft = tObj.storeFullTracking;
if vsrnew && ~sft
  qstr = 'Replicates will be stored with tracking results. This can significantly increase program memory usage.';
  resp = questdlg(qstr,'Warning: Memory Usage','OK, continue','Cancel','OK, continue');
  if isempty(resp)
    resp = 'Cancel';
  end
  if strcmp(resp,'Cancel')
    return;
  end
  tObj.storeFullTracking = true;
end
tObj.showVizReplicates = vsrnew;

function cbkLabels2HideChanged(src,evt)
lObj = evt.AffectedObject;
if isempty(lObj.tracker)
  handles = lObj.gdata;
  handles.menu_view_hide_predictions.Checked = onIff(lObj.labels2Hide);
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
  vcd = lObj.viewCalibrationDataCurrent;
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
% Really, "configure parameters"

lObj = handles.labelerObj;
tObj = lObj.tracker;
assert(~isempty(tObj));
assert(isa(tObj,'CPRLabelTracker'));

% Start with "new" parameter tree/specification which provides "new"
% parameters structure, types etc
prmBaseYaml = fullfile(APT.Root,'trackers','cpr','params_apt.yaml');
tPrm = parseConfigYaml(prmBaseYaml);

% Now overlay either the current parameters or some other starting pt
sPrmOld = tObj.getParams();
if isempty(sPrmOld) % eg new tracker
  sPrmNewOverlay = RC.getprop('lastCPRAPTParams');
  % sPrmNewOverlay could be [] if prop hasn't been set
else
  sPrmNewOverlay = CPRParam.old2new(sPrmOld,lObj);
end

% Set new-style params that map to Labeler props instead of old-style 
% params.
%
% Note. sPrmNewOverlay could be [] (see above). In this case, the following 
% lines will create the sPrmNewOverlay struct.
sPrmNewOverlay.ROOT.Track.NFramesSmall = lObj.trackNFramesSmall;
sPrmNewOverlay.ROOT.Track.NFramesLarge = lObj.trackNFramesLarge;
sPrmNewOverlay.ROOT.Track.NFramesNeighborhood = lObj.trackNFramesNear;

tPrm.structapply(sPrmNewOverlay);
sPrm = ParameterSetup(handles.figure,tPrm); % modal

if isempty(sPrm)
  % user canceled; none
else
  RC.saveprop('lastCPRAPTParams',sPrm);
  [sPrm,lObj.trackNFramesSmall,lObj.trackNFramesLarge,...
    lObj.trackNFramesNear] = CPRParam.new2old(sPrm,lObj.nPhysPoints,lObj.nview);
  tObj.setParams(sPrm);  
end

function cbkTrackerTrnDataDownSampChanged(src,evt,handles)
tracker = evt.AffectedObject;
if tracker.trnDataDownSamp
  handles.menu_track_use_all_labels_to_train.Checked = 'off';
  handles.menu_track_select_training_data.Checked = 'on';
else
  handles.menu_track_use_all_labels_to_train.Checked = 'on';
  handles.menu_track_select_training_data.Checked = 'off';
end

function menu_track_use_all_labels_to_train_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
tObj = lObj.tracker;
if isempty(tObj)
  error('LabelerGUI:tracker','No tracker for this project.');
end
if tObj.hasTrained && tObj.trnDataDownSamp
  resp = questdlg('A tracker has already been trained with downsampled training data. Proceeding will clear all previous trained/tracked results. OK?',...
    'Clear Existing Tracker','Yes, clear previous tracker','Cancel','Cancel');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'Yes, clear previous tracker'
      % none
    case 'Cancel'
      return;
  end
end
tObj.trnDataUseAll();

function menu_track_select_training_data_Callback(hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
if tObj.hasTrained
  resp = questdlg('A tracker has already been trained. Downsampling training data will clear all previous trained/tracked results. Proceed?',...
    'Clear Existing Tracker','Yes, clear previous tracker','Cancel','Cancel');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'Yes, clear previous tracker'
      % none
    case 'Cancel'
      return;
  end
end
tObj.trnDataSelect();

function menu_track_training_data_montage_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
lObj.tracker.trainingDataMontage();

function menu_track_retrain_Callback(hObject, eventdata, handles)
handles.labelerObj.trackRetrain();

function menu_track_crossvalidate_Callback(hObject, eventdata, handles)

lObj = handles.labelerObj;
if lObj.tracker.hasTrained
  resp = questdlg('Any existing trained tracker and tracking results will be cleared. Proceed?',...
    'Cross Validation',...
    'OK, Proceed','Cancel','Cancel');
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

wbObj = WaitBarWithCancel('Cross Validation');
oc = onCleanup(@()delete(wbObj));
[dGTTrkCell,pTrkCell] = lObj.trackCrossValidate('wbObj',wbObj);
if wbObj.isCancel
  msg = wbObj.cancelMessage('Cross validation canceled');
  msgbox(msg,'Cross Validation');
  return;
end

[nGT,nFold,muErr,muErrPt,tblErrMov] = ...
  Labeler.trackCrossValidateStats(dGTTrkCell,pTrkCell);
str = { ...
  sprintf('GT dataset: %d labeled frames across %d movies',nGT,height(tblErrMov));
  sprintf('Number of folds: %d',nFold);
  '';
  sprintf('Mean err, all points (px): %.2f',muErr)};
for ipt=1:numel(muErrPt)
  str{end+1,1} = sprintf('  ... point %d: %.2f',ipt,muErrPt(ipt)); %#ok<AGROW>
end
str{end+1,1} = '';
str{end+1,1} = sprintf('Mean err, all movies (px): %.2f',muErr);
for imov=1:height(tblErrMov)
  trow = tblErrMov(imov,:);
  [path,movS] = myfileparts(trow.mov{1});
  [~,path] = myfileparts(path);
  mov = fullfile(path,movS);
  str{end+1,1} = sprintf('  ... movie %s (%d rows): %.2f',mov,...
    trow.count,trow.err); %#ok<AGROW>
end 

hDlg = dialog('Name','Cross Validation','resize','on','WindowStyle','normal');
BORDER = 0.025;
hTxt = uicontrol('Parent',hDlg,'Style','edit',...
  'units','normalized','position',[BORDER BORDER 1-2*BORDER 1-2*BORDER],...
  'enable','on','Max',2,'horizontalalignment','left',...
  'String',str,'FontName','Courier New');

function cbkTrackerStoreFullTrackingChanged(hObject, eventdata, handles)
onoff = onIff(handles.labelerObj.tracker.storeFullTracking);
handles.menu_track_store_full_tracking.Checked = onoff;
handles.menu_track_view_tracking_diagnostics.Enable = onoff;

function menu_track_store_full_tracking_Callback(hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
svr = tObj.showVizReplicates;
sft = tObj.storeFullTracking;
sftnew = ~sft;
if ~sftnew && svr
  qstr = 'Replicates will no longer by shown. OK?';
  resp = questdlg(qstr,'Tracking Storage','OK, continue','No, cancel','OK, continue');
  if isempty(resp)
    resp = 'No, cancel';
  end
  if strcmp(resp,'No, cancel')
    return;
  end
  tObj.showVizReplicates = false;
end
tObj.storeFullTracking = sftnew;

function menu_track_view_tracking_diagnostics_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;

% Look for existing/open CPRVizTrackDiagsGUI
for i=1:numel(handles.depHandles)
  h = handles.depHandles(i);
  if isvalid(h) && strcmp(h.Tag,'figCPRVizTrackDiagsGUI')
    figure(h);
    return;
  end
end

lc = lObj.lblCore;
if ~isempty(lc) && ~lc.hideLabels
  warningNoTrace('LabelerGUI:hideLabels','Hiding labels.');
  lc.labelsHide();
end
hVizGUI = CPRVizTrackDiagsGUI(handles.labelerObj);
addDepHandle(handles.figure,hVizGUI);

function menu_track_track_and_export_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
tm = getTrackMode(handles);
[tfok,rawtrkname] = hlpRawtrkname(lObj);
if ~tfok
  return;
end
handles.labelerObj.trackAndExport(tm,'rawtrkname',rawtrkname);

function menu_track_export_current_movie_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
iMov = lObj.currMovie;
if iMov==0
  error('LabelerGUI:noMov','No movie currently set.');
end
[tfok,rawtrkname] = hlpRawtrkname(lObj);
if ~tfok
  return;
end
lObj.trackExportResults(iMov,'rawtrkname',rawtrkname);

function menu_track_export_all_movies_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
nMov = lObj.nmovies;
if nMov==0
  error('LabelerGUI:noMov','No movies in project.');
end
iMov=1:nMov;
[tfok,rawtrkname] = hlpRawtrkname(lObj);
if ~tfok
  return;
end
lObj.trackExportResults(iMov,'rawtrkname',rawtrkname);

function menu_track_set_labels_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
tObj = lObj.tracker;
if ~isempty(tObj)
  xy = tObj.getPredictionCurrentFrame();
  xy = xy(:,:,lObj.currTarget); % "targets" treatment differs from below
  if any(isnan(xy(:)))
    fprintf('No predictions for current frame, not labeling.\n');
    return;
  end
  disp(xy);
  
  % AL20161219: possibly dangerous, assignLabelCoords prob was intended
  % only as a util method for subclasses rather than public API. This may
  % not do the right thing for some concrete LabelCores.
  lObj.lblCore.assignLabelCoords(xy);
else
  if lObj.nTrx>1
    error('LabelerGUI:setLabels','Unsupported for multiple targets.');
  end  
  iMov = lObj.currMovie;
  frm = lObj.currFrame;
  if iMov==0
    error('LabelerGUI:setLabels','No movie open.');
  end
  lpos2 = lObj.labeledpos2{iMov};
  assert(size(lpos2,4)==1); % "targets" treatment differs from above
  lpos2xy = lpos2(:,:,frm);
  lObj.labelPosSet(lpos2xy);
  
  lObj.lblCore.newFrame(frm,frm,1);
end

function menu_track_background_predict_start_Callback(hObject,eventdata,handles)
tObj = handles.labelerObj.tracker;
if tObj.asyncIsPrepared
  tObj.asyncStartBGWorker();
else
  if ~tObj.hasTrained
    errordlg('A tracker has not been trained.','Background Tracking');
    return;
  end
  tObj.asyncPrepare();
  tObj.asyncStartBGWorker();
end
  
function menu_track_background_predict_end_Callback(hObject,eventdata,handles)
tObj = handles.labelerObj.tracker;
if tObj.asyncIsPrepared
  tObj.asyncStopBGWorker();
else
  warndlg('Background worker is not running.','Background tracking');
end

function menu_track_background_predict_stats_Callback(hObject,eventdata,handles)
tObj = handles.labelerObj.tracker;
if tObj.asyncIsPrepared
  tObj.asyncComputeStats();
else
  warningNoTrace('LabelerGUI:bgTrack',...
    'No background tracking information available.','Background tracking');
end

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

function play(hObject,handles,iconStrPlay,playMeth)
lObj = handles.labelerObj;
oc = onCleanup(@()playCleanup(hObject,handles,iconStrPlay));
if ~handles.isPlaying
  handles.isPlaying = true;
  guidata(hObject,handles);
  hObject.CData = Icons.ims.stop;
  lObj.(playMeth);
end
function playCleanup(hObject,handles,iconStrPlay)
hObject.CData = Icons.ims.(iconStrPlay);
handles.isPlaying = false;
guidata(hObject,handles);

function pbPlaySeg_Callback(hObject, eventdata, handles)
play(hObject,handles,'playsegment','videoPlaySegment');
function pbPlay_Callback(hObject, eventdata, handles)
play(hObject,handles,'play','videoPlay');
