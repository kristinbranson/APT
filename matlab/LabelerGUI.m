function varargout = LabelerGUI(varargin)
% Labeler GUI

% Last Modified by GUIDE v2.5 07-Nov-2018 14:52:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
% AL20151104: 'dpi-aware' MATLAB graphics introduced in R2015b have trouble
% with .figs created in previous versions. Did significant testing across
% MATLAB versions and platforms and behavior appears at least mildly 
% wonky-- couldn't figure out a clean solution. For now use two .figs
if ispc && ~verLessThan('matlab','8.6') % 8.6==R2015b
  gui_Name = 'LabelerGUI_PC_15b';
elseif isunix
  %gui_Name = 'LabelerGUI_lnx';
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

%MK 20190906 - A special function to function handle of the local functions.
% Typically this would have got handled by calling LabelerGUI(fn,...), but
% since our gui_Name doesn't match the name of this file this doesn't work
% anymore.
if nargin==2 && ischar(varargin{1}) && ischar(varargin{2}) && strcmp(varargin{1},'get_local_fn') 
    if exist(varargin{2}, 'file')
        fn = str2func(varargin{2});
    else
        fn = 0;
    end
    varargout = {fn};
    return
end

if nargin && ischar(varargin{1}) && exist(varargin{1}) %#ok<EXIST>
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function LabelerGUI_OpeningFcn(hObject,eventdata,handles,varargin) 

handles.labelerObj = varargin{1} ;
handles.controller = varargin{2} ;

if verLessThan('matlab','8.4')
  handles.labelerObj.lerror('LabelerGUI:ver','LabelerGUI requires MATLAB version R2014b or later.');
end

if handles.labelerObj.isgui,
  hfigsplash = splashScreen(handles);
end

% handles.SetStatusFun = @(~,s,varargin) fprintf([s,'...\n']);
% handles.ClearStatusFun = @(varargin) fprintf('Done.\n');
% handles.RefreshStatusFun = @(varargin) fprintf('\n');

hObject.Name = 'APT';
hObject.HandleVisibility = 'on';

% delete unused stuff from toolbar
h = findall(hObject,'type','uitoolbar');
KEEP = {'Exploration.Rotate' 'Exploration.Pan' 'Exploration.ZoomOut' ...
  'Exploration.ZoomIn'};
hh = findall(h,'-not','type','uitoolbar','-property','Tag');
for h=hh(:)'
  if ~ishandle(h),
    continue;
  end
  if ~any(strcmp(h.Tag,KEEP))
    delete(h);
  end
end

% reinit uicontrol strings etc from GUIDE for cosmetic purposes
set(handles.txPrevIm,'String','');
set(handles.edit_frame,'String','');
set(handles.popupmenu_prevmode,'Visible','off');
set(handles.pushbutton_freezetemplate,'Visible','off');
set(handles.txStatus,'String','Ready.');

% somehow this got corrupted in .figs
set(handles.tblFrames,'Data',[]);
% Correct this, outdated in .figs
set(handles.tblFrames,'ColumnName',{'Frame' 'Tgts' 'Pts'});

%syncStatusBarTextWhenClear(handles);
set(handles.txUnsavedChanges,'Visible','off');
set(handles.txLblCoreAux,'Visible','off');
%set(handles.pnlSusp,'Visible','off');

% color of status bar when GUI is busy vs idle
handles.idlestatuscolor = [0,1,0];
handles.busystatuscolor = [1,0,1];
% setappdata(handles.txStatus,'SetStatusFun',@SetStatus);
% setappdata(handles.txStatus,'ClearStatusFun',@ClearStatus);

% set location of background training status
pos1 = handles.txStatus.Position;
pos2 = handles.txBGTrain.Position;
r1 = pos1(1)+pos1(3);
r2 = pos2(1)+pos2(3);
pos2(1) = min(pos2(1),r1 + .01);
pos2(3) = r2-pos2(1);
handles.txBGTrain.Position = pos2;
handles.txBGTrain.FontWeight = 'normal';
handles.txBGTrain.FontSize = handles.txStatus.FontSize;

handles.labelerObj.setStatus('Initializing APT...');
% handles.SetStatusFun = @SetStatus;
% handles.ClearStatusFun = @ClearStatus;
% handles.SetStatusBarTextWhenClearFun = @setStatusBarTextWhenClear;
% handles.RefreshStatusFun = @refreshStatus;

%handles.pnlSusp.Visible = 'off';

PURP = [80 31 124]/256;
handles.tbTLSelectMode.BackgroundColor = PURP;

handles.output = hObject;

varargin = varargin(2:end); %#ok<NASGU>

set(handles.menu_file_quick_open,'Visible','off');

handles.menu_file_import_labels_table = uimenu('Parent',handles.menu_file_import_export_advanced,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_import_labels_table_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Import Labels from Table (All Movies)',...
  'Tag','menu_file_import_labels_table',...
  'Checked','off',...
  'Visible','on');
moveMenuItemAfter(handles.menu_file_import_labels_table,...
  handles.menu_file_import_labels_trk_curr_mov);

handles.menu_file_export_all_movies = uimenu('Parent',handles.menu_file_importexport,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_export_all_movies_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Export Predictions to Trk Files (All Movies)...',...
  'Tag','menu_file_export_all_movies'); 
moveMenuItemAfter(handles.menu_file_export_all_movies,handles.menu_file_export_labels2_trk_curr_mov);

handles.menu_file_clear_imported = uimenu('Parent',handles.menu_file_importexport,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_clear_imported_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Clear imported predictions (All Movies)...',...
  'Tag','menu_file_clear_imported',...
  'Separator','on' ...
  ); 
moveMenuItemAfter(handles.menu_file_clear_imported,handles.menu_file_export_all_movies);

handles.menu_file_export_labels_table = uimenu('Parent',handles.menu_file_import_export_advanced,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_export_labels_table_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Export Labels as Table',...
  'Tag','menu_file_export_labels_table',...
  'Checked','off',...
  'Visible','on');
moveMenuItemAfter(handles.menu_file_export_labels_table,...
  handles.menu_file_export_labels_trks);

handles.menu_file_export_stripped_lbl = uimenu('Parent',handles.menu_file_import_export_advanced,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_export_stripped_lbl_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Export Training Data...',...
  'Tag','menu_file_export_stripped_lbl',...
  'Checked','off',...
  'Visible','on');
moveMenuItemAfter(handles.menu_file_export_stripped_lbl,...
  handles.menu_file_export_labels_table);

handles.menu_file_crop_mode = uimenu('Parent',handles.menu_file,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_crop_mode_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Edit cropping',...
  'Tag','menu_file_crop_mode',...
  'Checked','off',...
  'Separator','on',...
  'Visible','on');
moveMenuItemAfter(handles.menu_file_crop_mode,...
  handles.menu_file_importexport);

handles.menu_file_clean_tempdir = uimenu('Parent',handles.menu_file,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_clean_tempdir_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Clean temporary directory',...
  'Tag','menu_file_clean_tempdir',...
  'Checked','off',...
  'Separator','on',...
  'Visible','on');
moveMenuItemAfter(handles.menu_file_clean_tempdir,...
  handles.menu_file_crop_mode);

handles.menu_file_bundle_tempdir = uimenu('Parent',handles.menu_file,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_file_bundle_tempdir_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Bundle working directory',...
  'Tag','menu_file_bundle_tempdir',...
  'Visible','on');
moveMenuItemAfter(handles.menu_file_bundle_tempdir,...
  handles.menu_file_clean_tempdir);

% Label/Setup menu
mnuLblSetup = handles.menu_labeling_setup;
mnuLblSetup.Position = 3;
if isprop(mnuLblSetup,'Text')
  mnuLblSetup.Text = 'Label';
else
  mnuLblSetup.Label = 'Label';
end

handles.menu_setup_multiview_calibrated_mode_2 = uimenu(...
  'Parent',handles.menu_labeling_setup,...
  'Label','Multiview',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_multiview_calibrated_mode_2_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_setup_multiview_calibrated_mode_2');
delete(handles.menu_setup_multiview_calibrated_mode);
handles.menu_setup_multiview_calibrated_mode = [];
delete(handles.menu_setup_tracking_correction_mode);
handles.menu_setup_tracking_correction_mode = [];
delete(handles.menu_setup_createtemplate);
handles.menu_setup_multianimal_mode = uimenu(...
  'Parent',handles.menu_labeling_setup,...
  'Label','Multianimal',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_multianimal_mode_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_setup_multianimal_mode');
moveMenuItemAfter(handles.menu_setup_multianimal_mode,...
  handles.menu_setup_multiview_calibrated_mode_2);

handles.menu_setup_use_calibration = uimenu(...
  'Parent',handles.menu_labeling_setup,...
  'Label','Use calibration',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_use_calibration_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_setup_use_calibration',...
  'Checked','off');  

handles.menu_setup_label_overlay_montage = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_label_overlay_montage_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Label Overlay Montage',...
  'Tag','menu_setup_label_overlay_montage',...
  'Visible','on');
% handles.menu_setup_label_overlay_montage_trx_centered = uimenu('Parent',handles.menu_labeling_setup,...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_label_overlay_montage_trx_centered_Callback',hObject,eventdata,guidata(hObject)),...
%   'Label','Label Overlay Montage (trx centered)',...
%   'Tag','menu_setup_label_overlay_montage_trx_centered',...
%   'Visible','on');
handles.menu_setup_set_nframe_skip = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_set_nframe_skip_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set Frame Increment',...
  'Tag','menu_setup_set_nframe_skip',...
  'Visible','on');
handles.menu_setup_streamlined = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_streamlined_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Streamlined',...
  'Tag','menu_setup_streamlined',...
  'Checked','off',...
  'Visible','on');

handles.menu_setup_ma_twoclick_align = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_ma_twoclick_align_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Two-click animal alignment',...
  'Tag','menu_setup_ma_twoclick_align',...
  'Checked','off',...
  'Visible','on');

handles.menu_setup_sequential_add_mode = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_sequential_add_mode_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Add landmarks mode',...
  'Tag','menu_setup_sequential_add_mode',...
  'Checked','off',...
  'Visible','off');

LABEL_MENU_ORDER = {
   'menu_setup_sequential_mode'
   'menu_setup_template_mode'
   'menu_setup_highthroughput_mode'
   'menu_setup_multiview_calibrated_mode_2'   
   'menu_setup_multianimal_mode'   
   'menu_setup_sequential_add_mode'
   'menu_setup_streamlined'
   'menu_setup_load_calibration_file'
   'menu_setup_use_calibration'
   'menu_setup_ma_twoclick_align'
   'menu_setup_label_overlay_montage' % 'menu_setup_label_overlay_montage_trx_centered'
   'menu_setup_set_labeling_point'
   'menu_setup_set_nframe_skip'
   'menu_setup_lock_all_frames'
   'menu_setup_unlock_all_frames'};
menuReorder(handles.menu_labeling_setup,LABEL_MENU_ORDER);
handles.menu_setup_label_overlay_montage.Separator = 'on';
handles.menu_setup_set_labeling_point.Separator = 'on';
handles.menu_setup_streamlined.Separator = 'on';
handles.menu_setup_load_calibration_file.Separator = 'off';
handles.menu_setup_load_calibration_file.Text = 'Select calibration file...';

handles.menu_view_show_bgsubbed_frames = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_bgsubbed_frames_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show background subtracted frames',...
  'Tag','menu_view_show_bgsubbed_frames',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_bgsubbed_frames,...
  handles.menu_view_gammacorrect);

handles.menu_view_rotate_video_target_up = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_rotate_video_target_up_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Rotate video so target always points up',...
  'Tag','menu_view_rotate_video_target_up',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_rotate_video_target_up,...
  handles.menu_view_trajectories_centervideoontarget);

handles.menu_view_show_axes_toolbar = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_axes_toolbar_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show axes toolbar',...
  'Tag','menu_view_show_axes_toolbar',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_axes_toolbar,...
  handles.menu_view_rotate_video_target_up);

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

handles.menu_view_show_preds_curr_target_only = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_preds_curr_target_only_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show predictions for current target only',...
  'Tag','menu_view_show_preds_curr_target_only',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_preds_curr_target_only,handles.menu_view_hide_imported_predictions);

handles.menu_view_show_imported_preds_curr_target_only = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_imported_preds_curr_target_only_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Show imported predictions for current target only',...
  'Tag','menu_view_show_imported_preds_curr_target_only',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_show_imported_preds_curr_target_only,handles.menu_view_show_preds_curr_target_only);

deleteValidHandles(handles.menu_view_landmark_colors.Children);
set(handles.menu_view_landmark_colors,'Callback',@menu_view_landmark_colors_Callback);

handles.menu_view_showhide_skeleton = uimenu('Parent',handles.menu_view,...
  'Label','Show skeleton',...
  'Tag','menu_view_showhide_skeleton',...
  'Checked','off',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_showhide_skeleton_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_view_showhide_skeleton,handles.menu_view_landmark_colors);

% handles.menu_view_showhide_advanced = uimenu('Parent',handles.menu_view,...
%   'Label','Advanced',...
%   'Tag','menu_view_showhide_advanced');
% moveMenuItemAfter(handles.menu_view_showhide_advanced,handles.menu_view_showhide_skeleton);
% handles.menu_view_showhide_advanced_hidepredtxtlbls = uimenu('Parent',handles.menu_view_showhide_advanced,...
%   'Label','Hide Prediction Text Labels',...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_view_showhide_advanced_hidepredtxtlbls_Callback',hObject,eventdata,guidata(hObject)),...
%   'Tag','menu_view_showhide_advanced_hidepredtxtlbls',...
%   'Checked','off');

handles.menu_view_hide_trajectories = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_hide_trajectories_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Hide trajectories',...
  'Tag','menu_view_hide_trajectories',...
  'Checked','off');
%moveMenuItemAfter(handles.menu_view_hide_trajectories,handles.menu_track_cpr_show_replicates);
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
moveMenuItemAfter(handles.menu_view_occluded_points_box,handles.menu_view_show_grid);

% AL20201008 Zoom/Pan hotkey toggle
% Semi-hack to enable hot-key toggling of zoom/pan for MA labeling. During
% MA labeling in large FOV vids, finding animals/targets is a major
% ergonomic consideration. We want to leverage the built-in MATLAB zoom and
% pan tools, but having to click them (whether in the "new" locations in
% the upper-right or original toolbar locs) is time-consuming.
%
% So we wanted to use hotkeys to toggle these modes which runs into the
% following difficulty. When zoom/pan mode are on, MATLAB 
% overwrites/replaces the KeypressFcn and WindowsKeyPressFcn for a figure. 
% So it is possible to use a KeyPressFcn-defined hotkey to *activate* 
% zoom/pan mode, but then unactivating cannot work the same way (without 
% some other hack for re-setting the KPF etc.)
%
% We add these menu options to enable the Accelerator-style hotkeys which
% are unaffected by zoom/pan state. We only show these menu options for MA
% mode and in any case it doesn't hurt to have them.
handles.menu_view_zoom_toggle = uimenu('Parent',handles.menu_view,...
  'Callback',@(h,e)zoom(hObject),...
  'Label','Toggle Zoom',...
  'Tag','menu_view_zoom_toggle',...
  'Separator','on');
handles.menu_view_pan_toggle = uimenu('Parent',handles.menu_view,...
  'Callback',@(h,e)pan(hObject),...
  'Label','Toggle Pan',...
  'Tag','menu_view_pan_toggle' ...
  );
handles.menu_view_showhide_maroi = uimenu('Parent',handles.menu_view,...
  'Label','Show target label ROIs',...
  'Tag','menu_view_showhide_maroi',...
  'Checked','off',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_showhide_maroi_Callback',hObject,eventdata,guidata(hObject)));
handles.menu_view_showhide_maroiaux = uimenu('Parent',handles.menu_view,...
  'Label','Show extra label ROIs',...
  'Tag','menu_view_showhide_maroiaux',...
  'Checked','off',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_showhide_maroiaux_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_view_zoom_toggle,handles.menu_view_occluded_points_box);
moveMenuItemAfter(handles.menu_view_pan_toggle,handles.menu_view_zoom_toggle);
moveMenuItemAfter(handles.menu_view_showhide_maroi,handles.menu_view_pan_toggle);
moveMenuItemAfter(handles.menu_view_showhide_maroiaux,handles.menu_view_showhide_maroi);

% handles.menu_view_show_3D_axes = uimenu('Parent',handles.menu_view,...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_view_show_3D_axes_Callback',hObject,eventdata,guidata(hObject)),...
%   'Label','Show/Refresh 3D world axes',...
%   'Tag','menu_view_show_3D_axes',...
%   'Checked','off');
% moveMenuItemAfter(handles.menu_view_show_3D_axes,handles.menu_view_show_grid);

set(handles.menu_track_setparametersfile,...
  'Label','Set training parameters...',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_setparametersfile_Callback',hObject,eventdata,guidata(hObject)),...
  'Separator','on'); % separator b/c trackers are listed above

handles.menu_track_edit_skeleton = uimenu('Parent',handles.menu_track,...
  'Label','Landmark parameters...',...
  'Tag','menu_track_edit_skeleton',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_edit_skeleton_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_track_edit_skeleton,handles.menu_track_setparametersfile);

handles.menu_track_settrackparams = uimenu('Parent',handles.menu_track,...
  'Label','Set tracking parameters...',...
  'Tag','menu_track_settrackparams',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_settrackparams_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_track_settrackparams,handles.menu_track_edit_skeleton);

handles.menu_track_viz_dataaug = uimenu('Parent',handles.menu_track,...
  'Label','Visualize sample training images...',...
  'Tag','menu_track_viz_dataaug',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_viz_dataaug_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_track_viz_dataaug,handles.menu_track_settrackparams);

% handles.menu_track_set_landmark_matches = uimenu(...
%   'Parent',handles.menu_track,...
%   'Label','Set landmark flip pairings...',...
%   'Tag','menu_track_set_landmark_matches',...
%   'Callback',@(h,evtdata)LabelerGUI('menu_track_set_landmark_matches_Callback',h,evtdata,guidata(h)));
% moveMenuItemAfter(handles.menu_track_set_landmark_matches,handles.menu_track_setparametersfile);

% handles.menu_track_use_all_labels_to_train = uimenu(...
%   'Parent',handles.menu_track,...
%   'Label','Include all labels in training data',...
%   'Tag','menu_track_use_all_labels_to_train',...
%   'Separator','on',...
%   'Callback',@(h,evtdata)LabelerGUI('menu_track_use_all_labels_to_train_Callback',h,evtdata,guidata(h)));
% moveMenuItemAfter(handles.menu_track_use_all_labels_to_train,handles.menu_track_setparametersfile);
% handles.menu_track_select_training_data.Label = 'Downsample training data';
% handles.menu_track_select_training_data.Visible = 'off';
% moveMenuItemAfter(handles.menu_track_select_training_data,handles.menu_track_use_all_labels_to_train);
handles.menu_track_training_data_montage = uimenu(...
  'Parent',handles.menu_track,...
  'Label','Training Data Montage',...
  'Tag','menu_track_training_data_montage',...
  'Callback',@(h,evtdata)LabelerGUI('menu_track_training_data_montage_Callback',h,evtdata,guidata(h)));
%moveMenuItemAfter(handles.menu_track_training_data_montage,handles.menu_track_select_training_data);
moveMenuItemAfter(handles.menu_track_training_data_montage,handles.menu_track_viz_dataaug);
delete(handles.menu_track_select_training_data);

handles.menu_track_batch_track = uimenu(...
  'Parent',handles.menu_track,...
  'Label','Track multiple videos...',...
  'Tag','menu_track_batch_track',...
  'Callback',@(h,evtdata)LabelerGUI('menu_track_batch_track_Callback',h,evtdata,guidata(h)),...
  'Separator','on');
moveMenuItemAfter(handles.menu_track_batch_track,handles.menu_track_training_data_montage);

handles.menu_track_current_movie = uimenu(...
  'Parent',handles.menu_track,...
  'Label','Track current movie...',...
  'Tag','menu_track_current_movie',...
  'Callback',@(h,evtdata)LabelerGUI('menu_track_current_movie_Callback',h,evtdata,guidata(h)));
moveMenuItemAfter(handles.menu_track_current_movie,handles.menu_track_batch_track);

handles.menu_track_all_movies = uimenu(...
  'Parent',handles.menu_track,...
  'Label','Track all movies in project...',...
  'Tag','menu_track_all_movies',...
  'Callback',@(h,evtdata)LabelerGUI('menu_track_all_movies_Callback',h,evtdata,guidata(h)));
moveMenuItemAfter(handles.menu_track_all_movies,handles.menu_track_current_movie);

% handles.menu_track_id = uimenu(...
%   'Parent',handles.menu_track,...
%   'Label','Link using Identity',...
%   'Tag','menu_track_id',...
%   'Callback',@(h,evtdata)LabelerGUI('menu_track_id_Callback',h,evtdata,guidata(h)));
% moveMenuItemAfter(handles.menu_track_id,handles.menu_track_all_movies);

moveMenuItemAfter(handles.menu_track_track_and_export,handles.menu_track_retrain);

handles.menu_track_trainincremental = handles.menu_track_retrain;
handles = rmfield(handles,'menu_track_retrain');
handles.menu_track_trainincremental.Callback = @(h,edata)LabelerGUI('menu_track_trainincremental_Callback',h,edata,guidata(h));
handles.menu_track_trainincremental.Label = 'Incremental Train';
handles.menu_track_trainincremental.Tag = 'menu_track_trainincremental';
handles.menu_track_trainincremental.Visible = 'off';
%handles.menu_track_track_and_export.Separator = 'off';

% handles.menu_track_export_base = uimenu('Parent',handles.menu_track,...
%   'Label','Export current tracking results',...
%   'Tag','menu_track_export_base',...
%   'Separator','on');  
% moveMenuItemAfter(handles.menu_track_export_base,handles.menu_track_track_and_export);
% handles.menu_track_export_current_movie = uimenu('Parent',handles.menu_track_export_base,...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_track_export_current_movie_Callback',hObject,eventdata,guidata(hObject)),...
%   'Label','Current movie only',...
%   'Tag','menu_track_export_current_movie');  

handles.menu_track_clear_tracking_results = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_clear_tracking_results_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Clear tracking results',...
  'Tag','menu_track_clear_tracking_results',...
  'Separator','on');  
moveMenuItemAfter(handles.menu_track_clear_tracking_results,handles.menu_track_all_movies);

handles.menu_track_clear_tracker = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_clear_tracker_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Clear trained tracker',...
  'Tag','menu_track_clear_tracker',...
  'Separator','off');  
moveMenuItemAfter(handles.menu_track_clear_tracker,handles.menu_track_clear_tracking_results);

handles.menu_track_set_labels = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_set_labels_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set manual labels to automatic prediction',...
  'Tag','menu_track_set_labels',...
  'Separator','on');  

% tfBGok = ~isempty(ver('distcomp')) && ~verLessThan('distcomp','6.10');
% onoff = onIff(tfBGok);
% handles.menu_track_background_predict = uimenu('Parent',handles.menu_track,...
%   'Label','Background prediction','Tag','menu_track_background_predict',...
%   'Separator','on','Enable',onoff);
% moveMenuItemAfter(handles.menu_track_background_predict,...
%   handles.menu_track_set_labels);

% handles.menu_track_background_predict_start = uimenu(...
%   'Parent',handles.menu_track_background_predict,...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_track_background_predict_start_Callback',hObject,eventdata,guidata(hObject)),...
%   'Label','Start/enable background prediction',...
%   'Tag','menu_track_background_predict_start');
% handles.menu_track_background_predict_end = uimenu(...
%   'Parent',handles.menu_track_background_predict,...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_track_background_predict_end_Callback',hObject,eventdata,guidata(hObject)),...
%   'Label','Stop background prediction',...
%   'Tag','menu_track_background_predict_end');
% handles.menu_track_background_predict_stats = uimenu(...
%   'Parent',handles.menu_track_background_predict,...
%   'Callback',@(hObject,eventdata)LabelerGUI('menu_track_background_predict_stats_Callback',hObject,eventdata,guidata(hObject)),...
%   'Label','Background prediction stats',...
%   'Tag','menu_track_background_predict_stats');
% 

handles.menu_track_cpr_storefull = uimenu('Parent',handles.menu_track,...
  'Label','(CPR) Store tracking replicates/iterations',...
  'Tag','menu_track_cpr_storefull',...
  'Separator','on');
% moveMenuItemAfter(handles.menu_track_cpr_storefull,...
%   handles.menu_track_cpr_show_replicates);
handles.menu_track_cpr_storefull_dont_store = uimenu(...
  'Parent',handles.menu_track_cpr_storefull,...
  'Label','Don''t store replicates',...
  'Tag','menu_track_cpr_storefull_dont_store',...
  'Checked','on',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_cpr_storefull_dont_store_Callback',hObject,eventdata,guidata(hObject)));
handles.menu_track_cpr_storefull_store_final_iteration = uimenu(...
  'Parent',handles.menu_track_cpr_storefull,...
  'Label','Store replicates, final iteration only',...
  'Tag','menu_track_cpr_storefull_store_final_iteration',...
  'Checked','off',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_cpr_storefull_store_final_iteration_Callback',hObject,eventdata,guidata(hObject)));
handles.menu_track_cpr_storefull_store_all_iterations = uimenu(...
  'Parent',handles.menu_track_cpr_storefull,...
  'Label','Store replicates, all iterations',...
  'Tag','menu_track_cpr_storefull_store_all_iterations',...
  'Checked','off',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_cpr_storefull_store_all_iterations_Callback',hObject,eventdata,guidata(hObject)));

handles.menu_track_cpr_show_replicates = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_cpr_show_replicates_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','(CPR) Show predicted replicates',...
  'Tag','menu_track_cpr_show_replicates',...
  'Checked','off');
moveMenuItemAfter(handles.menu_track_cpr_show_replicates,...
  handles.menu_track_cpr_storefull);

handles.menu_track_cpr_view_diagnostics = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_cpr_view_diagnostics_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','(CPR) View tracking diagnostics',...
  'Tag','menu_track_cpr_view_diagnostics',...
  'Separator','off',...
  'Checked','off');
moveMenuItemAfter(handles.menu_track_cpr_view_diagnostics,...
  handles.menu_track_cpr_show_replicates);

handles.menu_track_auto_params_update = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_auto_params_update_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Auto compute training parameters',...
  'Tag','menu_track_auto_params_update',...
  'Checked','on',...
  'Visible','on');
moveMenuItemAfter(handles.menu_track_auto_params_update,...
  handles.menu_track_setparametersfile);


handles.menu_help_about = uimenu(...
  'Parent',handles.menu_help,...
  'Label','About',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_help_about_Callback',hObject,eventdata,guidata(hObject)),...
  'Tag','menu_help_about');  
moveMenuItemAfter(handles.menu_help_about,handles.menu_help_labeling_actions);

handles.menu_help_doc = uimenu(...
  'Parent',handles.menu_help,...
  'Label','Documentation',...
  'Callback',@(hObject,eventdata)web('https://kristinbranson.github.io/APT'),...
  'Tag','menu_help_doc');  
moveMenuItemBefore(handles.menu_help_doc,handles.menu_help_labeling_actions);

% Go menu
handles.menu_go = uimenu('Parent',handles.figure,'Position',4,'Label','Go');
handles.menu_go_targets_summary = uimenu('Parent',handles.menu_go,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_go_targets_summary_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Switch targets...',...
  'Tag','menu_go_targets_summary',...
  'Separator','off',...
  'Checked','off');
handles.menu_go_movies_summary = uimenu('Parent',handles.menu_go,...
  'Callback',@(hObject,eventdata) menu_file_managemovies_Callback(hObject,eventdata,guidata(hObject)),...
  'Label','Switch movies...',...
  'Tag','menu_go_movies_summary',...
  'Separator','off',...
  'Checked','off');
handles.menu_go_nav_prefs = uimenu('Parent',handles.menu_go,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_go_nav_prefs_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Navigation preferences...',...
  'Tag','menu_go_nav_prefs',...
  'Separator','off',...
  'Checked','off');
handles.menu_go_gt_frames = uimenu('Parent',handles.menu_go,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_go_gt_frames_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','GT frames',...
  'Tag','menu_go_gt_frames',...
  'Separator','on',...
  'Checked','off');

% Evaluate menu
handles.menu_evaluate = uimenu('Parent',handles.figure,'Position',6,'Label','Evaluate');
handles.menu_evaluate_crossvalidate = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_crossvalidate_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Cross validate',...
  'Tag','menu_evaluate_crossvalidate',...
  'Separator','off',...
  'Checked','off');
handles.menu_evaluate_gtmode = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_gtmode_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Ground-Truthing Mode',...
  'Tag','menu_evaluate_gtmode',...
  'Separator','off',...
  'Checked','off');

handles.menu_evaluate_gtloadsuggestions = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_gtloadsuggestions_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Load GT suggestions',...
  'Tag','menu_evaluate_gtloadsuggestions',...
  'Separator','on');
handles.menu_evaluate_gtsetsuggestions = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_gtsetsuggestions_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set GT suggestions to current GT labels',...
  'Tag','menu_evaluate_gtsetsuggestions',...
  'Separator','off');
handles.menu_evaluate_gtcomputeperf = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_gtcomputeperf_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Compute GT performance',...
  'Tag','menu_evaluate_gtcomputeperf',...
  'Separator','on');
handles.menu_evaluate_gtcomputeperfimported = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_gtcomputeperfimported_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Compute GT performance (imported predictions)',...
  'Tag','menu_evaluate_gtcomputeperfimported',...
  'Separator','off');
handles.menu_evaluate_gtexportresults = uimenu('Parent',handles.menu_evaluate,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_evaluate_gtexportresults_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Export GT performance results',...
  'Tag','menu_evaluate_gtexportresults',...
  'Separator','off');


handles.menu_go.Position = 4;
handles.menu_track.Position = 5;
handles.menu_evaluate.Position = 6;
handles.menu_help.Position = 7;
% hCMenu = uicontextmenu('parent',handles.figure);
% uimenu('Parent',hCMenu,'Label','Freeze to current main window',...
%   'Callback',@(src,evt)cbkFreezePrevAxesToMainWindow(src,evt));
% uimenu('Parent',hCMenu,'Label','Display last frame seen in main window',...
%   'Callback',@(src,evt)cbkUnfreezePrevAxes(src,evt));
% handles.axes_prev.UIContextMenu = hCMenu;

% misc labelmode/Setup menu
LABELMODE_SETUPMENU_MAP = ...
  {LabelMode.NONE '';
   LabelMode.SEQUENTIAL 'menu_setup_sequential_mode';
   LabelMode.TEMPLATE 'menu_setup_template_mode';
   LabelMode.HIGHTHROUGHPUT 'menu_setup_highthroughput_mode';
   LabelMode.MULTIVIEWCALIBRATED2 'menu_setup_multiview_calibrated_mode_2'; 
   LabelMode.MULTIANIMAL 'menu_setup_multianimal_mode';
   LabelMode.SEQUENTIALADD 'menu_setup_sequential_add_mode'};
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

handles.image_curr = imagesc(0,'Parent',handles.axes_curr,'Tag','image_curr');
set(handles.image_curr,'PickableParts','none');
hold(handles.axes_curr,'on');
set(handles.axes_curr,'Color',[0 0 0],'Tag','axes_curr');
set(handles.axes_curr.Toolbar,'Visible','off');
handles.image_prev = imagesc(0,'Parent',handles.axes_prev,'Tag','image_prev');
set(handles.image_prev,'PickableParts','none');
hold(handles.axes_prev,'on');
set(handles.axes_prev,'Color',[0 0 0],'Tag','axes_prev');
%set(hObject,'WindowScrollWheelFcn',@scroll_callback);
% set(hObject,'WindowbuttonDownFcn',@dragstart_callback);
% set(hObject,'Windowbuttonmotionfcn',@drag_callback);
% set(hObject,'WindowbuttonUpFcn',@dragend_callback);

handles.figs_all = handles.figure;
handles.axes_all = handles.axes_curr;
handles.images_all = handles.image_curr;
handles.cropHRect = [];
handles.tbAdjustCropSizeString0 = 'Adjust Size';
handles.tbAdjustCropSizeString1 = 'Done Adjusting';
handles.tbAdjustCropSizeBGColor0 = handles.tbAdjustCropSize.BackgroundColor;
handles.tbAdjustCropSizeBGColor1 = [1 0 0];

pumTrack = handles.pumTrack;
pumTrack.Value = 1;
pumTrack.String = {'All frames'};
%set(pumTrack,'FontUnits','points','FontSize',6.5);
%pumTrack.FontUnits = 'normalized';
aptResize = APTResize(handles);
handles.figure.SizeChangedFcn = @(src,evt)aptResize.resize(src,evt);
aptResize.resize(handles.figure,[]);
handles.pumTrack.Callback = ...
  @(hObj,edata)LabelerGUI('pumTrack_Callback',hObj,edata,guidata(hObj));

lObj = handles.labelerObj;

handles.labelTLInfo = InfoTimeline(lObj,handles.axes_timeline_manual,...
  handles.axes_timeline_islabeled);

set(handles.pumInfo,'String',handles.labelTLInfo.getPropsDisp(),...
  'Value',handles.labelTLInfo.curprop);
set(handles.pumInfo_labels,'String',handles.labelTLInfo.getPropTypesDisp(),...
  'Value',handles.labelTLInfo.curproptype);

% this is currently not used - KB made space here for training status
%set(handles.txProjectName,'String','');

listeners = cell(0,1);
listeners{end+1,1} = addlistener(handles.slider_frame,'ContinuousValueChange',@slider_frame_Callback);
listeners{end+1,1} = addlistener(handles.sldZoom,'ContinuousValueChange',@sldZoom_Callback);
listeners{end+1,1} = addlistener(handles.axes_curr,'XLim','PostSet',@(s,e)axescurrXLimChanged(s,e,handles));
listeners{end+1,1} = addlistener(handles.axes_curr,'XDir','PostSet',@(s,e)axescurrXDirChanged(s,e,handles));
listeners{end+1,1} = addlistener(handles.axes_curr,'YDir','PostSet',@(s,e)axescurrYDirChanged(s,e,handles));
listeners{end+1,1} = addlistener(lObj,'didSetProjname',@cbkProjNameChanged);
listeners{end+1,1} = addlistener(lObj,'didSetCurrTarget',@cbkCurrTargetChanged);
listeners{end+1,1} = addlistener(lObj,'didSetLastLabelChangeTS',@cbkLastLabelChangeTS);
listeners{end+1,1} = addlistener(lObj,'didSetTrackParams',@cbkParameterChange);
listeners{end+1,1} = addlistener(lObj,'didSetLabelMode',@cbkLabelModeChanged);
listeners{end+1,1} = addlistener(lObj,'didSetLabels2Hide',@cbkLabels2HideChanged);
listeners{end+1,1} = addlistener(lObj,'didSetLabels2ShowCurrTargetOnly',@cbkLabels2ShowCurrTargetOnlyChanged);
listeners{end+1,1} = addlistener(lObj,'didSetProjFSInfo',@cbkProjFSInfoChanged);
listeners{end+1,1} = addlistener(lObj,'didSetShowTrx',@cbkShowTrxChanged);
listeners{end+1,1} = addlistener(lObj,'didSetShowOccludedBox',@cbkShowOccludedBoxChanged);
listeners{end+1,1} = addlistener(lObj,'didSetShowTrxCurrTargetOnly',@cbkShowTrxCurrTargetOnlyChanged);
listeners{end+1,1} = addlistener(lObj,'didSetTrackersAll',@cbkTrackersAllChanged);
listeners{end+1,1} = addlistener(lObj,'didSetCurrTracker',@cbkCurrTrackerChanged);
listeners{end+1,1} = addlistener(lObj,'didSetTrackModeIdx',@cbkTrackModeIdxChanged);
listeners{end+1,1} = addlistener(lObj,'didSetTrackNFramesSmall',@cbkTrackerNFramesChanged);
listeners{end+1,1} = addlistener(lObj,'didSetTrackNFramesLarge',@cbkTrackerNFramesChanged);    
listeners{end+1,1} = addlistener(lObj,'didSetTrackNFramesNear',@cbkTrackerNFramesChanged);
listeners{end+1,1} = addlistener(lObj,'didSetMovieCenterOnTarget',@cbkMovieCenterOnTargetChanged);
listeners{end+1,1} = addlistener(lObj,'didSetMovieRotateTargetUp',@cbkMovieRotateTargetUpChanged);
listeners{end+1,1} = addlistener(lObj,'didSetMovieForceGrayscale',@cbkMovieForceGrayscaleChanged);
listeners{end+1,1} = addlistener(lObj,'didSetMovieInvert',@cbkMovieInvertChanged);
listeners{end+1,1} = addlistener(lObj,'didSetMovieViewBGsubbed',@cbkMovieViewBGsubbedChanged);
listeners{end+1,1} = addlistener(lObj,'didSetLblCore',@(src,evt)(handles.controller.didSetLblCore()));
listeners{end+1,1} = addlistener(lObj,'gtIsGTModeChanged',@cbkGtIsGTModeChanged);
listeners{end+1,1} = addlistener(lObj,'cropIsCropModeChanged',@cbkCropIsCropModeChanged);
listeners{end+1,1} = addlistener(lObj,'cropUpdateCropGUITools',@cbkUpdateCropGUITools);
listeners{end+1,1} = addlistener(lObj,'cropCropsChanged',@cbkCropCropsChanged);
listeners{end+1,1} = addlistener(lObj,'newProject',@cbkNewProject);
listeners{end+1,1} = addlistener(lObj,'newMovie',@cbkNewMovie);
listeners{end+1,1} = addlistener(lObj,'projLoaded',@cbkProjLoaded);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'selectOn','PostSet',@cbklabelTLInfoSelectOn);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_tracker','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_allframes','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'proptypes','PostSet',@cbklabelTLInfoPropTypesUpdated);
listeners{end+1,1} = addlistener(lObj,'startAddMovie',@cbkAddMovie);
listeners{end+1,1} = addlistener(lObj,'finishAddMovie',@cbkAddMovie);
listeners{end+1,1} = addlistener(lObj,'startSetMovie',@cbkSetMovie);
listeners{end+1,1} = addlistener(lObj,'dataImported',@cbkDataImported);
listeners{end+1,1} = addlistener(lObj,'didSetShowSkeleton',@cbkShowSkeletonChanged);
listeners{end+1,1} = addlistener(lObj,'didSetShowMaRoi',@cbkShowMaRoiChanged);
listeners{end+1,1} = addlistener(lObj,'didSetShowMaRoiAux',@cbkShowMaRoiAuxChanged);

handles.listeners = listeners;
handles.listenersTracker = cell(0,1); % listeners added in cbkCurrTrackerChanged
handles.menu_track_trackers = cell(0,1); % menus added in cbkTrackersAllChanged

hZ = zoom(hObject);
hZ.ActionPostCallback = @cbkPostZoom;
hP = pan(hObject);
hP.ActionPostCallback = @cbkPostPan;

% These Labeler properties need their callbacks fired to properly init UI.
% Labeler will read .propsNeedInit from the GUIData to comply.
handles.propsNeedInit = {
  'labelMode' 
  'suspScore' 
  'showTrx' 
  'showTrxCurrTargetOnly' %  'showPredTxtLbl'
  'trackersAll' % AL20190606: trackersAll cbk calls 'currTracker' cbk
  'trackNFramesSmall' % trackNFramesLarge, trackNframesNear currently share same callback
  'trackModeIdx'
  'movieCenterOnTarget'
  'movieForceGrayscale' 
  'movieInvert'
  'showOccludedBox'
  };

% handles that are only enabled in multi-view or single-view mode

handles.h_multiview_only = [...
  handles.menu_setup_multiview_calibrated_mode_2...
  ];
handles.h_singleview_only = [...
   handles.menu_setup_sequential_mode ...
   handles.menu_setup_template_mode ...
   handles.menu_setup_highthroughput_mode ...
   handles.menu_setup_multianimal_mode ...
   handles.menu_setup_sequential_add_mode ...
   ];
handles.h_ma_only = [...
  handles.menu_setup_multianimal_mode, ...
  %handles.menu_track_id ...
  ];
handles.h_nonma_only = [ ...
  handles.menu_setup_multiview_calibrated_mode_2...
  handles.menu_setup_sequential_mode ...
  handles.menu_setup_template_mode ...
  handles.menu_setup_highthroughput_mode ...
  handles.menu_setup_sequential_add_mode ...
  ];
handles.h_addpoints_only = [...
  handles.menu_setup_sequential_add_mode ...
  ];
  
  
set(handles.output,'Toolbar','figure');

handles = initTblTrx(handles);
%handles = initTblFrames(handles);

figSetPosAPTDefault(hObject);
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

% Add play-segment-reverse btn
SHRINKFAC = 0.7;
hps0 = handles.pbPlaySeg;
hps0right0 = hps0.Position(1)+hps0.Position(3);
hps0.Position(3) = hps0.Position(3)*SHRINKFAC;
btngap = hps0.Position(1)-handles.pbPlay.Position(1)-handles.pbPlay.Position(3);

hps1 = copyobj(hps0,hps0.Parent);
hps1.Position(1) = hps1.Position(1) + hps0.Position(3)+btngap/2;
set(hps1,...
  'CData',fliplr(hps1.CData),...
  'Callback',@(hObject,eventdata)LabelerGUI('pbPlaySegRev_Callback',hObject,eventdata,guidata(hObject)),...
  'Enable',hps0.Enable,...
  'Tag','pbPlaySegRev');
handles.pbPlaySegRev = hps1;
handles.pbPlaySegBoth = [hps0 hps1];

hps1right1 = hps1.Position(1)+hps1.Position(3);
dx = hps1right1 - hps0right0; % edit_frame, slider_frame shifted to right by this much
handles.edit_frame.Position(1) = handles.edit_frame.Position(1) + dx;
handles.slider_frame.Position([1 3]) = handles.slider_frame.Position([1 3]) + dx*[1 -1];

EnableControls(handles,'tooltipinit');
set(handles.figure,'Visible','on');
if handles.labelerObj.isgui,
  RefocusSplashScreen(hfigsplash,handles);
end

LabelerTooltips(handles);
if handles.labelerObj.isgui,
  RefocusSplashScreen(hfigsplash,handles);
  if ishandle(hfigsplash),
    delete(hfigsplash);
  end
end

% get rid of extra toolbars
h = findall(hObject,'-property','Toolbar');
for i = 1:numel(h),
  htool = get(h(i),'Toolbar');
  if ishandle(htool),
    set(htool,'Visible','off');
  end
end

lObj.clearStatus();
EnableControls(handles,'noproject');

if ismac % Change color of buttons 
 toChange = {'pbClear','tbAccept','pbTrain','pbTrack',...
     'pumTrack','tbTLSelectMode','pbClearSelection',...
     'pumInfo_labels','pumInfo'};
 for ndx = 1:numel(toChange)
     set(handles.(toChange{ndx}),'ForegroundColor',[1.0,0.0,1.0]);
 end
end

guidata(hObject, handles);

fprintf('Labeler GUI created.\n');

% UIWAIT makes LabelerGUI wait for user response (see UIRESUME)
% uiwait(handles.figure);

function EnableControls(handles,state)

switch lower(state),
  case 'init',
    
    set(handles.menu_file,'Enable','off');
    set(handles.menu_view,'Enable','off');
    set(handles.menu_labeling_setup,'Enable','off');
    set(handles.menu_track,'Enable','off');
    set(handles.menu_go,'Enable','off');
    set(handles.menu_evaluate,'Enable','off');
    set(handles.menu_help,'Enable','off');
    
    set(handles.tbAdjustCropSize,'Enable','off');
    set(handles.pbClearAllCrops,'Enable','off');
    set(handles.pushbutton_exitcropmode,'Enable','off');
    set(handles.uipanel_cropcontrols,'Visible','off');
    set(handles.text_trackerinfo,'Visible','on');
    
    set(handles.pbClearSelection,'Enable','off');
    set(handles.pumInfo,'Enable','off');
    set(handles.pumInfo_labels,'Enable','off');
    set(handles.tbTLSelectMode,'Enable','off');
    set(handles.pumTrack,'Enable','off');
    set(handles.pbTrack,'Enable','off');
    set(handles.pbTrain,'Enable','off');
    set(handles.pbClear,'Enable','off');
    set(handles.tbAccept,'Enable','off');
    set(handles.pbRecallZoom,'Enable','off');
    set(handles.pbSetZoom,'Enable','off');
    set(handles.pbResetZoom,'Enable','off');
    set(handles.sldZoom,'Enable','off');
    set(handles.pbPlaySegBoth,'Enable','off');
    set(handles.pbPlay,'Enable','off');
    set(handles.slider_frame,'Enable','off');
    set(handles.edit_frame,'Enable','off');
    set(handles.popupmenu_prevmode,'Enable','off');
    set(handles.pushbutton_freezetemplate,'Enable','off');
    set(handles.FigureToolBar,'Visible','off')

  case 'tooltipinit',
    
    set(handles.menu_file,'Enable','on');
    set(handles.menu_view,'Enable','on');
    set(handles.menu_labeling_setup,'Enable','on');
    set(handles.menu_track,'Enable','on');
    set(handles.menu_go,'Enable','on');
    set(handles.menu_evaluate,'Enable','on');
    set(handles.menu_help,'Enable','on');
    
    set(handles.tbAdjustCropSize,'Enable','off');
    set(handles.pbClearAllCrops,'Enable','off');
    set(handles.pushbutton_exitcropmode,'Enable','off');
    set(handles.uipanel_cropcontrols,'Visible','off');
    set(handles.text_trackerinfo,'Visible','off');
    
    set(handles.pbClearSelection,'Enable','off');
    set(handles.pumInfo,'Enable','off');
    set(handles.pumInfo_labels,'Enable','off');
    set(handles.tbTLSelectMode,'Enable','off');
    set(handles.pumTrack,'Enable','off');
    set(handles.pbTrack,'Enable','off');
    set(handles.pbTrain,'Enable','off');
    set(handles.pbClear,'Enable','off');
    set(handles.tbAccept,'Enable','off');
    set(handles.pbRecallZoom,'Enable','off');
    set(handles.pbSetZoom,'Enable','off');
    set(handles.pbResetZoom,'Enable','off');
    set(handles.sldZoom,'Enable','off');
    set(handles.pbPlaySegBoth,'Enable','off');
    set(handles.pbPlay,'Enable','off');
    set(handles.slider_frame,'Enable','off');
    set(handles.edit_frame,'Enable','off');
    set(handles.popupmenu_prevmode,'Enable','off');
    set(handles.pushbutton_freezetemplate,'Enable','off');
    set(handles.FigureToolBar,'Visible','off')
    
  case 'noproject',
    set(handles.menu_file,'Enable','on');
    set(handles.menu_view,'Enable','off');
    set(handles.menu_labeling_setup,'Enable','off');
    set(handles.menu_track,'Enable','off');
    set(handles.menu_evaluate,'Enable','off');
    set(handles.menu_go,'Enable','off');
    set(handles.menu_help,'Enable','off');

    set(handles.menu_file_quit,'Enable','on');
    set(handles.menu_file_crop_mode,'Enable','off');
    set(handles.menu_file_importexport,'Enable','off');
    set(handles.menu_file_managemovies,'Enable','off');
    set(handles.menu_file_load,'Enable','on');
    set(handles.menu_file_saveas,'Enable','off');
    set(handles.menu_file_save,'Enable','off');
    set(handles.menu_file_shortcuts,'Enable','off');
    set(handles.menu_file_new,'Enable','on');
    %set(handles.menu_file_quick_open,'Enable','on','Visible','on');
    
    set(handles.tbAdjustCropSize,'Enable','off');
    set(handles.pbClearAllCrops,'Enable','off');
    set(handles.pushbutton_exitcropmode,'Enable','off');
    set(handles.uipanel_cropcontrols,'Visible','off');    
    set(handles.text_trackerinfo,'Visible','off');

    
    set(handles.pbClearSelection,'Enable','off');
    set(handles.pumInfo,'Enable','off');
    set(handles.tbTLSelectMode,'Enable','off');
    set(handles.pumTrack,'Enable','off');
    set(handles.pbTrack,'Enable','off');
    set(handles.pbTrain,'Enable','off');
    set(handles.pbClear,'Enable','off');
    set(handles.tbAccept,'Enable','off');
    set(handles.pbRecallZoom,'Enable','off');
    set(handles.pbSetZoom,'Enable','off');
    set(handles.pbResetZoom,'Enable','off');
    set(handles.sldZoom,'Enable','off');
    set(handles.pbPlaySegBoth,'Enable','off');
    set(handles.pbPlay,'Enable','off');
    set(handles.slider_frame,'Enable','off');
    set(handles.edit_frame,'Enable','off');
    set(handles.popupmenu_prevmode,'Enable','off');
    set(handles.pushbutton_freezetemplate,'Enable','off');
    set(handles.FigureToolBar,'Visible','off')

  case 'projectloaded'

    set(findobj(handles.menu_file,'-property','Enable'),'Enable','on');
    set(handles.menu_view,'Enable','on');
    set(handles.menu_labeling_setup,'Enable','on');
    set(handles.menu_track,'Enable','on');
    set(handles.menu_evaluate,'Enable','on');
    set(handles.menu_go,'Enable','on');
    set(handles.menu_help,'Enable','on');
    
    % KB 20200504: I think this is confusing when a project is already open
    % AL 20220719: now always hiding
    % set(handles.menu_file_quick_open,'Visible','off');
    
    set(handles.tbAdjustCropSize,'Enable','on');
    set(handles.pbClearAllCrops,'Enable','on');
    set(handles.pushbutton_exitcropmode,'Enable','on');
    %set(handles.uipanel_cropcontrols,'Visible','on');

    set(handles.pbClearSelection,'Enable','on');
    set(handles.pumInfo,'Enable','on');
    set(handles.pumInfo_labels,'Enable','on');
    set(handles.tbTLSelectMode,'Enable','on');
    set(handles.pumTrack,'Enable','on');
    %set(handles.pbTrack,'Enable','on');
    %set(handles.pbTrain,'Enable','on');
    set(handles.pbClear,'Enable','on');
    set(handles.tbAccept,'Enable','on');
    set(handles.pbRecallZoom,'Enable','on');
    set(handles.pbSetZoom,'Enable','on');
    set(handles.pbResetZoom,'Enable','on');
    set(handles.sldZoom,'Enable','on');
    set(handles.pbPlaySegBoth,'Enable','on');
    set(handles.pbPlay,'Enable','on');
    set(handles.slider_frame,'Enable','on');
    set(handles.edit_frame,'Enable','on');
    set(handles.popupmenu_prevmode,'Enable','on');
    set(handles.pushbutton_freezetemplate,'Enable','on');
    set(handles.FigureToolBar,'Visible','on')
    
    lObj = handles.labelerObj;
    tObj = lObj.tracker;    
    tfTracker = ~isempty(tObj);
    onOff = onIff(tfTracker);
    handles.menu_track.Enable = onOff;
    handles.pbTrain.Enable = onOff;
    handles.pbTrack.Enable = onOff;
    handles.menu_view_hide_predictions.Enable = onOff;    
    
    tfGoTgts = ~lObj.gtIsGTMode;
    set(handles.menu_go_targets_summary,'Enable',onIff(tfGoTgts));
    
    if lObj.nview == 1,
      set(handles.h_multiview_only,'Enable','off');
    elseif lObj.nview > 1,
      set(handles.h_singleview_only,'Enable','off');
    else
      handles.labelerObj.lerror('Sanity check -- nview = 0');
    end
    if lObj.maIsMA
      set(handles.h_nonma_only,'Enable','off');
%      set(handles.menu_track_id,'Checked',handles.labelerObj.track_id,'Visible','on');
    else
      set(handles.h_ma_only,'Enable','off');
%      set(handles.menu_track_id,'Visible','off');
    end
    if lObj.nLabelPointsAdd == 0,
      set(handles.h_addpoints_only,'Visible','off');
    else
      set(handles.h_addpoints_only,'Visible','on');
    end

  otherwise
    fprintf('Not implemented\n');
end

function handles = initTblTrx(handles)
tbl0 = handles.tblTrx;
COLNAMES = {'Index' 'Labeled'};
set(tbl0,...
  'ColumnWidth',{100 100},...
  'ColumnName',COLNAMES,...
  'Data',cell(0,numel(COLNAMES)),...
  'CellSelectionCallback',@(src,evt)cbkTblTrxCellSelection(src,evt),...
  'FontUnits','points',...
  'FontSize',9.75,... % matches .tblTrx
  'BackgroundColor',[.3 .3 .3; .45 .45 .45]);  
% AL 20210209: jtable performance is too painful for larger projs (more 
% labels in any single movie). As of 2020x only cost to using regular 
% table is inability to set selected/hilite row.

function handles = initTblFrames(handles,isMA)
tbl0 = handles.tblFrames;
tbl0.Units = 'pixel';
tw = tbl0.Position(3);
if tw<50,  tw= 50; end
tbl0.Units = 'normalized';
if isMA
  COLNAMES = {'Frm' 'Tgts' 'Pts' 'ROIs'};
  COLWIDTH = {min(tw/4-1,80) min(tw/4-5,40) max(tw/4-7,10) max(tw/4-7,10)};
else
  COLNAMES = {'Frame' 'Tgts' 'Pts'};
  COLWIDTH = {100 50 'auto'};
end

% if 1 
set(tbl0,...
  'ColumnWidth',COLWIDTH,...
  'ColumnName',COLNAMES,...
  'Data',cell(0,numel(COLNAMES)),...
  'CellSelectionCallback',@(src,evt)cbkTblFramesCellSelection(src,evt),...
  'FontUnits','points',...
  'FontSize',9.75,... % matches .tblTrx
  'BackgroundColor',[.3 .3 .3; .45 .45 .45]);

% AL 20210209: jtable performance is too painful for larger projs (more
% labels in any single movie). As of 2020x only cost to using regular
% table is inability to set selected/hilite row.

% else
%   jt = uiextras.jTable.Table(...
%     'parent',tbl0.Parent,...
%     'Position',tbl0.Position,...
%     'SelectionMode','single',...
%     'Editable','off',...
%     'ColumnPreferredWidth',[100 50],...
%     'ColumnName',COLNAMES,... %  'ColumnFormat',{'integer' 'integer' 'integer'},...  'ColumnEditable',[false false false],...
%     'CellSelectionCallback',@(src,evt)cbkTblFramesCellSelection(src,evt));
%   set(jt,'Data',cell(0,numel(COLNAMES)));
%   cr = aptjava.StripedIntegerTableCellRenderer;
%   for i=0:2
%     jt.JColumnModel.getColumn(i).setCellRenderer(cr);
%   end
%   jt.JTable.Foreground = java.awt.Color.WHITE;
%   jt.hPanel.BackgroundColor = [0.3 0.3 0.3];
%   h = jt.JTable.getTableHeader;
%   h.setPreferredSize(java.awt.Dimension(225,22));
%   jt.JTable.repaint;
% 
%   delete(tbl0);
%   handles.tblFrames = jt;
% end

function varargout = LabelerGUI_OutputFcn(hObject, eventdata, handles) %#ok<*INUSL>
varargout{1} = handles.output;

function handles = clearDepHandles(handles)
deleteValidHandles(handles.depHandles);
handles.depHandles = gobjects(0,1);

function handles = addDepHandle(handles,h)
% GC dead handles
tfValid = arrayfun(@isvalid,handles.depHandles);
handles.depHandles = handles.depHandles(tfValid,:);
    
tfSame = arrayfun(@(x)x==h,handles.depHandles);
if ~any(tfSame)
  handles.depHandles(end+1,1) = h;
end

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
tfbatch = batchStartupOptionUsed; % ci
if tfbatch
  sel = CLOSESTR;
else
  sel = questdlg('This figure is required for your current multiview project.',...
    'Close Request Function',...
    DONTCLOSESTR,CLOSESTR,DONTCLOSESTR);
  if isempty(sel)
    sel = DONTCLOSESTR;
  end
end
switch sel
  case DONTCLOSESTR
    % none
  case CLOSESTR
    delete(gcf)
end

function cbkTrackerHideVizChanged(src,evt,hmenu_view_hide_predictions)
tracker = evt.AffectedObject;
hmenu_view_hide_predictions.Checked = onIff(tracker.hideViz);
function cbkTrackerShowPredsCurrTargetOnlyChanged(src,evt,hmenu)
tracker = evt.AffectedObject;
hmenu.Checked = onIff(tracker.showPredsCurrTargetOnly);


function tfKPused = cbkKPF(src,evt,lObj)
if ~lObj.isReady ,
  return
end

tfKPused = false;
isarrow = ismember(evt.Key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'});
if isarrow && ismember(src,lObj.gdata.h_ignore_arrows),
  return
end

% % first try user-defined KeyPressHandlers
% kph = lObj.keyPressHandlers ;
% for i = 1:numel(kph) ,
%   tfKPused = kph(i).handleKeyPress(evt, lObj) ;
%   if tfKPused ,
%     return
%   end
% end

tfShift = any(strcmp('shift',evt.Modifier));
tfCtrl = any(strcmp('control',evt.Modifier));

isMA = lObj.maIsMA;
handles = guidata(src);
% KB20160724: shortcuts from preferences
% skip this for MA projs where we need separate hotkey mappings
if ~isMA && all(isfield(handles,{'shortcutkeys','shortcutfns'}))
  % control key pressed?
  if tfCtrl && numel(evt.Modifier)==1 && any(strcmpi(evt.Key,handles.shortcutkeys))
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
  return
end

lcore = lObj.lblCore;
if ~isempty(lcore)
  tfKPused = lcore.kpf(src,evt);
  if tfKPused
    return
  end
end

%disp(evt);
if any(strcmp(evt.Key,{'leftarrow' 'rightarrow'}))
  switch evt.Key
    case 'leftarrow'
      if tfShift
        sam = lObj.movieShiftArrowNavMode;
        samth = lObj.movieShiftArrowNavModeThresh;
        samcmp = lObj.movieShiftArrowNavModeThreshCmp;
        [tffound,f] = sam.seekFrame(lObj,-1,samth,samcmp);
        if tffound
          lObj.setFrameProtected(f);
        end
      else
        lObj.frameDown(tfCtrl);
      end
      tfKPused = true;
    case 'rightarrow'
      if tfShift
        sam = lObj.movieShiftArrowNavMode;
        samth = lObj.movieShiftArrowNavModeThresh;
        samcmp = lObj.movieShiftArrowNavModeThreshCmp;
        [tffound,f] = sam.seekFrame(lObj,1,samth,samcmp);
        if tffound
          lObj.setFrameProtected(f);
        end
      else
        lObj.frameUp(tfCtrl);
      end
      tfKPused = true;
  end
  return
end

if lObj.gtIsGTMode && strcmp(evt.Key,{'r'})
  lObj.gtNextUnlabeledUI();
  return
end

      
function cbkWBMF(src,evt,lObj)
lcore = lObj.lblCore;
if ~isempty(lcore)
  lcore.wbmf(src,evt);
end

function cbkWBUF(src,evt,lObj)
if ~isempty(lObj.lblCore)
  lObj.lblCore.wbuf(src,evt);
end

function cbkWSWF(src,evt,lObj)

if ~lObj.hasMovie
  return;
end

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

handles = clearDepHandles(handles);

handles = initTblFrames(handles,lObj.maIsMA);

%curr_status_string=handles.txStatus.String;
%SetStatus(handles,curr_status_string,true);

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
  handles = addDepHandle(handles,figs(iView));
  
  ims(iView) = imagesc(0,'Parent',axs(iView));
  set(ims(iView),'PickableParts','none');
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

% AL 20191002 This is to enable labeling simple projs without the Image
% toolbox (crop init uses imrect)
try
  handles = cropInitImRects(handles);
catch ME
  fprintf(2,'Crop Mode initialization error: %s\n',ME.message);
end

if isfield(handles,'allAxHiliteMgr') && ~isempty(handles.allAxHiliteMgr)
  % Explicit deletion not supposed to be nec
  delete(handles.allAxHiliteMgr);
end
handles.allAxHiliteMgr = AxesHighlightManager(axs);

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
setGUIFigureNames(handles,lObj,figs);
setMainAxesName(handles,lObj);

arrayfun(@(x)zoom(x,'off'),handles.figs_all); % Cannot set KPF if zoom or pan is on
arrayfun(@(x)pan(x,'off'),handles.figs_all);
hTmp = findall(handles.figs_all,'-property','KeyPressFcn','-not','Tag','edit_frame');
set(hTmp,'KeyPressFcn',@(src,evt)cbkKPF(src,evt,lObj));
handles.h_ignore_arrows = [handles.slider_frame];
%set(handles.figs_all,'WindowButtonMotionFcn',@(src,evt)cbkWBMF(src,evt,lObj));
%set(handles.figs_all,'WindowButtonUpFcn',@(src,evt)cbkWBUF(src,evt,lObj));
% if ispc
%   set(handles.figs_all,'WindowScrollWheelFcn',@(src,evt)cbkWSWF(src,evt,lObj));
% end

% eg when going from proj-with-trx to proj-no-trx, targets table needs to
% be cleared
set(handles.tblTrx,'Data',cell(0,size(handles.tblTrx.ColumnName,2)));

handles = setShortcuts(handles);

handles.labelTLInfo.initNewProject();

if isfield(handles,'movieMgr') && isvalid(handles.movieMgr)
  delete(handles.movieMgr);
end
handles.movieMgr = MovieManagerController(handles.labelerObj);
drawnow; % 20171002 Without this, new tabbed MovieManager shows up with 
  % buttons clipped at bottom edge of UI (manually resizing UI then "snaps"
  % buttons/figure back into a good state)   
handles.movieMgr.setVisible(false);

handles.GTMgr = GTManager(handles.labelerObj);
handles.GTMgr.Visible = 'off';
handles = addDepHandle(handles,handles.GTMgr);

guidata(handles.figure,handles);

%handles.labelerObj.clearStatus();

function setGUIMainFigureName(lObj)

maxlength = 80;
if isempty(lObj.projectfile),
  projname = [lObj.projname,' (unsaved)'];
elseif numel(lObj.projectfile) <= maxlength,
  projname = lObj.projectfile;
else
  [~,projname,ext] = fileparts(lObj.projectfile);
  projname = [projname,ext];
end
lObj.gdata.figure.Name = sprintf('APT - Project %s',projname);

function setGUIFigureNames(handles,lObj,figs)

setGUIMainFigureName(lObj);

viewNames = lObj.viewNames;
for i=2:lObj.nview,
  vname = viewNames{i};
  if isempty(vname)
    str = sprintf('View %d',i);
  else
    str = sprintf('View: %s',vname);
  end
  if numel(lObj.movieInvert) >= i && lObj.movieInvert(i),
    str = [str,' (inverted)'];  %#ok<AGROW> 
  end  
  figs(i).Name = str;
  figs(i).NumberTitle = 'off';
end

function setMainAxesName(handles,lObj)

viewNames = lObj.viewNames;
if lObj.nview > 1,
  if isempty(viewNames{1}),
    str = 'View 1, ';
  else
    str = sprintf('View: %s, ',viewNames{1});
  end
else
  str = '';
end
mname = lObj.moviename;
if lObj.nview>1
  str = [str,sprintf('Movieset %d',lObj.currMovie)];  
else
  str = [str,sprintf('Movie %d',lObj.currMovie)];
end
if lObj.gtIsGTMode
  str = [str,' (GT)'];
end
str = [str,': ',mname];
if ~isempty(lObj.movieInvert) && lObj.movieInvert(1),
  str = [str,' (inverted)'];
end

set(handles.txMoviename,'String',str);


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

if lObj.hasMovie && evt.isFirstMovieOfProject,
  EnableControls(handles,'projectloaded');
end

if ~lObj.gtIsGTMode,
  set(handles.menu_go_targets_summary,'Enable','on');
else
  set(handles.menu_go_targets_summary,'Enable','off');
end

wbmf = @(src,evt)cbkWBMF(src,evt,lObj);
wbuf = @(src,evt)cbkWBUF(src,evt,lObj);
movnr = lObj.movienr;
movnc = lObj.movienc;
figs = lObj.gdata.figs_all;
if lObj.hasMovie
  % guard against callback during new proj creation etc; lObj.movienc/nr
  % are NaN which creates a badly-inited imgzoompan. Theoretically seems
  % this wouldn't matter as the next imgzoompan created (when movie
  % actually added) should be properly initted...
  for ivw=1:lObj.nview
    set(figs(ivw),'WindowScrollWheelFcn',@(src,evt)scroll_callback(src,evt,lObj));
    set(figs(ivw),'WindowButtonMotionFcn',wbmf,'WindowButtonUpFcn',wbuf);

    [hascrop,cropInfo] = lObj.cropGetCropCurrMovie();
    if hascrop
      xmax = cropInfo(2); xmin = cropInfo(1);
      ymax = cropInfo(4); ymin = cropInfo(3);
    else
      xmax = movnc(ivw); xmin = 0;
      ymax = movnr(ivw); ymin = 0;
    end
    imgzoompan(figs(ivw),'wbmf',wbmf,'wbuf',wbuf,...
      'ImgWidth',xmax,'ImgHeight',ymax,'PanMouseButton',2,...
      'ImgXMin',xmin,'ImgYMin',ymin);
  end
end

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
%   if tfResetCLims
%     hAxs(iView).CLimMode = 'auto';
%   end
end

handles.labelTLInfo.initNewMovie();
handles.labelTLInfo.setLabelsFull();

nframes = lObj.nframes;
sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
set(handles.slider_frame,'Value',0,'SliderStep',sliderstep);

tfHasMovie = lObj.currMovie>0;
if tfHasMovie
  minzoomrad = 10;
  maxzoomrad = (lObj.movienc(1)+lObj.movienr(1))/4;
  handles.sldZoom.UserData = log([minzoomrad maxzoomrad]);
end

TRX_MENUS = {...
  'menu_view_trajectories_centervideoontarget'
  'menu_view_rotate_video_target_up'
  'menu_view_hide_trajectories'
  'menu_view_plot_trajectories_current_target_only'};
%  'menu_setup_label_overlay_montage_trx_centered'};
tftblon = lObj.hasTrx || lObj.maIsMA;
onOff = onIff(tftblon);
cellfun(@(x)set(handles.(x),'Enable',onOff),TRX_MENUS);
hTbl = handles.tblTrx;
set(hTbl,'Enable',onOff);
guidata(handles.figure,handles);

setPUMTrackStrs(lObj);

% See note in AxesHighlightManager: Trx vs noTrx, Axes vs Panels
handles.allAxHiliteMgr.setHilitePnl(lObj.hasTrx);

hlpGTUpdateAxHilite(lObj);

if lObj.cropIsCropMode
  cropUpdateCropHRects(handles);
end
handles.menu_file_crop_mode.Enable = onIff(~lObj.hasTrx);

% update HUD, statusbar
mname = lObj.moviename;
if lObj.nview>1
  movstr = 'Movieset';
else
  movstr = 'Movie';
end
if lObj.gtIsGTMode
  str = sprintf('%s %d (GT): %s',movstr,lObj.currMovie,mname);  
else
  str = sprintf('%s %d: %s',movstr,lObj.currMovie,mname);
end
set(handles.txMoviename,'String',str);

% by default, use calibration if there is calibration for this movie
lc = lObj.lblCore;
if ~isempty(lc) && lc.supportsCalibration,
  handles.menu_setup_use_calibration.Checked = onIff(lc.isCalRig && lc.showCalibration);
end

if ~isempty(mname)
  %str = sprintf('new %s %s at %s',lower(movstr),mname,datestr(now,16));
  %setStatusBarTextWhenClear(handles,str);
  %SetStatus(handles,str,false); %in cbkNewMovie
  %SetStatus(handles,str,true);
  %set(handles.txStatus,'String',str);
  
  % Fragile behavior when loading projects; want project status update to
  % persist and not movie status update. This depends on detailed ordering in 
  % Labeler.projLoad
end

function cbkAddMovie(src,evt)
% lObj=src;
% %handles = lObj.gdata;
% 
% if strcmp(evt.EventName,'startAddMovie')
%     %SetStatus(handles,'Adding movie',true); 
% elseif strcmp(evt.EventName,'finishAddMovie')
%     %handles.labelerObj.clearStatus();        
% end

function cbkSetMovie(src,evt)
% lObj=src;
% handles = lObj.gdata;
% 
% if strcmp(evt.EventName,'startSetMovie')
%     %SetStatus(handles,'Setting first movie',true); 
% elseif strcmp(evt.EventName,'finishSetMovie')
%     %handles.labelerObj.clearStatus();        
% end

function cbkProjLoaded(src,evt)
lObj = src;
handles = lObj.gdata;
cbkCurrTargetChanged(src,struct('AffectedObject',lObj));
EnableControls(handles,'projectloaded');
% update tracker info when loading in new trackers
if ~isempty(lObj.tracker)
  lObj.tracker.updateTrackerInfo();
end

function cbkDataImported(src,evt)
lObj = src;
handles = lObj.gdata;
handles.labelTLInfo.newTarget(); % Using this as a "refresh" for now

function zoomOutFullView(hAx,hIm,resetCamUpVec)
if isequal(hIm,[])
  axis(hAx,'auto');
else
  xdata = hIm.XData;
  ydata = hIm.YData;
  set(hAx,...
    'XLim',[xdata(1)-0.5 xdata(end)+0.5],...
    'YLim',[ydata(1)-0.5 ydata(end)+0.5]);
end
axis(hAx,'image');
zoom(hAx,'reset');
if resetCamUpVec
  hAx.CameraUpVectorMode = 'auto';
end
hAx.CameraViewAngleMode = 'auto';
hAx.CameraPositionMode = 'auto';
hAx.CameraTargetMode = 'auto';

% function cbkCurrFrameChanged(src,evt) %#ok<*INUSD>
% ticinfo = tic;
% starttime = ticinfo;
% lObj = evt.AffectedObject;
% frm = lObj.currFrame;
% nfrm = lObj.nframes;
% handles = lObj.gdata;
% fprintf('cbkCurrFrameChanged 1 get stuff: %f\n',toc(ticinfo)); ticinfo = tic;
% set(handles.edit_frame,'String',num2str(frm));
% fprintf('cbkCurrFrameChanged 2 set edit_frame: %f\n',toc(ticinfo)); ticinfo = tic;
% sldval = (frm-1)/(nfrm-1);
% if isnan(sldval)
%   sldval = 0;
% end
% set(handles.slider_frame,'Value',sldval);
% fprintf('cbkCurrFrameChanged 3 slider_frame %f\n',toc(ticinfo)); ticinfo = tic;
% if ~lObj.isinit
%   %handles.labelTLInfo.newFrame(frm);
%   fprintf('cbkCurrFrameChanged 4 labelTLInfo.newFrame %f\n',toc(ticinfo)); ticinfo = tic;
%   hlpGTUpdateAxHilite(lObj);
%   fprintf('cbkCurrFrameChanged 5 axhilite %f\n',toc(ticinfo));
% end
% fprintf('->cbkCurrFrameChanged total time: %f\n',toc(starttime));

% function hlpGTUpdateAxHilite(lObj)
% if lObj.gtIsGTMode
%   tfHilite = lObj.gtCurrMovFrmTgtIsInGTSuggestions();
% else
%   tfHilite = false;
% end
% lObj.gdata.allAxHiliteMgr.setHighlight(tfHilite);

function hlpUpdateTblTrxHilite(lObj)

try
  if lObj.maIsMA
    lObj.gdata.tblTrx.SelectedRows = [];
  else
    i = find(lObj.currTarget == lObj.tblTrxData(:,1));
    assert(numel(i) == 1);
    lObj.gdata.tblTrx.SelectedRows = i;
  end
catch ME  %#ok<NASGU> 
  warningNoTrace('Error caught updating highlight row in Targets Table.');
end


function cbkCurrTargetChanged(src, ~)
lObj = src ;
if (lObj.hasTrx || lObj.maIsMA) && ~lObj.isinit ,
  iTgt = lObj.currTarget;
  lObj.currImHud.updateTarget(iTgt);
  lObj.gdata.labelTLInfo.newTarget();
  lObj.hlpGTUpdateAxHilite();
  %drawnow;
  %hlpUpdateTblTrxHilite(lObj);
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
lObj = src ;
handles = lObj.gdata;
lblMode = lObj.labelMode;
menuSetupLabelModeHelp(handles,lblMode);
switch lblMode
  case LabelMode.SEQUENTIAL
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
    handles.menu_setup_use_calibration.Visible = 'off';
    handles.menu_setup_ma_twoclick_align.Visible = 'off';
    handles.menu_view_zoom_toggle.Visible = 'off';
    handles.menu_view_pan_toggle.Visible = 'off';
    handles.menu_view_showhide_maroi.Visible = 'off';
    handles.menu_view_showhide_maroiaux.Visible = 'off';
  case LabelMode.SEQUENTIALADD
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
    handles.menu_setup_use_calibration.Visible = 'off';
    handles.menu_setup_ma_twoclick_align.Visible = 'off';
    handles.menu_view_zoom_toggle.Visible = 'off';
    handles.menu_view_pan_toggle.Visible = 'off';
    handles.menu_view_showhide_maroi.Visible = 'off';
    handles.menu_view_showhide_maroiaux.Visible = 'off';
  case LabelMode.MULTIANIMAL
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
    handles.menu_setup_use_calibration.Visible = 'off';
    handles.menu_setup_ma_twoclick_align.Visible = 'on';
    handles.menu_setup_ma_twoclick_align.Checked = lObj.isTwoClickAlign;
    handles.menu_view_zoom_toggle.Visible = 'on';
    handles.menu_view_pan_toggle.Visible = 'on';
    handles.menu_view_showhide_maroi.Visible = 'on';
    handles.menu_view_showhide_maroiaux.Visible = 'on';
  case LabelMode.TEMPLATE
%     handles.menu_setup_createtemplate.Visible = 'on';
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
    handles.menu_setup_use_calibration.Visible = 'off';
    handles.menu_setup_ma_twoclick_align.Visible = 'off';
    handles.menu_view_zoom_toggle.Visible = 'off';
    handles.menu_view_pan_toggle.Visible = 'off';
    handles.menu_view_showhide_maroi.Visible = 'off';
    handles.menu_view_showhide_maroiaux.Visible = 'off';
  case LabelMode.HIGHTHROUGHPUT
%     handles.menu_setup_createtemplate.Visible = 'off';
    handles.menu_setup_set_labeling_point.Visible = 'on';
    handles.menu_setup_set_nframe_skip.Visible = 'on';
    handles.menu_setup_streamlined.Visible = 'off';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'off';
    handles.menu_setup_use_calibration.Visible = 'off';
    handles.menu_setup_ma_twoclick_align.Visible = 'off';
    handles.menu_view_zoom_toggle.Visible = 'off';
    handles.menu_view_pan_toggle.Visible = 'off';
    handles.menu_view_showhide_maroi.Visible = 'off';
    handles.menu_view_showhide_maroiaux.Visible = 'off';
  case LabelMode.MULTIVIEWCALIBRATED2
    handles.menu_setup_set_labeling_point.Visible = 'off';
    handles.menu_setup_set_nframe_skip.Visible = 'off';
    handles.menu_setup_streamlined.Visible = 'on';
    handles.menu_setup_unlock_all_frames.Visible = 'off';
    handles.menu_setup_lock_all_frames.Visible = 'off';
    handles.menu_setup_load_calibration_file.Visible = 'on';
    handles.menu_setup_use_calibration.Visible = 'on';
    handles.menu_setup_ma_twoclick_align.Visible = 'off';
    handles.menu_view_zoom_toggle.Visible = 'off';
    handles.menu_view_pan_toggle.Visible = 'off';
    handles.menu_view_showhide_maroi.Visible = 'off';
    handles.menu_view_showhide_maroiaux.Visible = 'off'; 
end

%lc = lObj.lblCore;
%tfShow3DAxes = ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration;
% handles.menu_view_show_3D_axes.Enable = onIff(tfShow3DAxes);

% function hlpUpdateTxProjectName(lObj)
% projname = lObj.projname;
% info = lObj.projFSInfo;
% if isempty(info)
%   str = projname;
% else
%   [~,projfileS] = myfileparts(info.filename);  
%   str = sprintf('%s / %s',projfileS,projname);
% end
% hTX = lObj.gdata.txProjectName;
% hTX.String = str;

function cbkProjNameChanged(src,evt)
lObj = src ;
%handles = lObj.gdata;
%pname = lObj.projname;
str = sprintf('Project $PROJECTNAME created (unsaved) at %s',datestr(now,16));
%setStatusBarTextWhenClear(handles,str);
lObj.setRawStatusStringWhenClear_(str) ;
%SetStatus(handles,str,false);
% set(handles.txStatus,'String',str);
%hlpUpdateTxProjectName(lObj);
setGUIMainFigureName(lObj);

function cbkProjFSInfoChanged(src,evt)
lObj = src ;
info = lObj.projFSInfo;
if ~isempty(info)  
  str = sprintf('Project $PROJECTNAME %s at %s',info.action,datestr(info.timestamp,16));
  %set(lObj.gdata.txStatus,'String',str);
  %setStatusBarTextWhenClear(lObj.gdata,str);
  lObj.setRawStatusStringWhenClear_(str) ;
  %SetStatus(lObj.gdata,str,false);
end
%hlpUpdateTxProjectName(lObj);
setGUIMainFigureName(lObj);

function cbkMovieForceGrayscaleChanged(src,evt)
lObj = src ;
tf = lObj.movieForceGrayscale;
mnu = lObj.gdata.menu_view_converttograyscale;
mnu.Checked = onIff(tf);

function cbkMovieInvertChanged(src,evt)
lObj = src ;
figs = lObj.gdata.figs_all;
setGUIFigureNames(lObj.gdata,lObj,figs);
setMainAxesName(lObj.gdata,lObj);

% movInvert = lObj.movieInvert;
% viewNames = lObj.viewNames;
% for i=1:lObj.nview
%   name = viewNames{i};
%   if isempty(name)
%     name = ''; 
%   else
%     name = sprintf('View: %s',name);
%   end
%   if movInvert(i)
%     name = [name ' (movie inverted)']; %#ok<AGROW>
%   end
%   figs(i).Name = name;
% end

function cbkMovieViewBGsubbedChanged(src,evt)
lObj = src ;
tf = lObj.movieViewBGsubbed;
mnu = lObj.gdata.menu_view_show_bgsubbed_frames;
mnu.Checked = onIff(tf);

% function cbkSuspScoreChanged(src,evt)
% lObj = evt.AffectedObject;
% ss = lObj.suspScore;
% lObj.currImHud.updateReadoutFields('hasSusp',~isempty(ss));
% 
% assert(~lObj.gtIsGTMode,'Unsupported in GT mode.');
% 
% handles = lObj.gdata;
% pnlSusp = handles.pnlSusp;
% tblSusp = handles.tblSusp;
% tfDoSusp = ~isempty(ss) && lObj.hasMovie && ~lObj.isinit;
% if tfDoSusp 
%   nfrms = lObj.nframes;
%   ntgts = lObj.nTargets;
%   [tgt,frm] = meshgrid(1:ntgts,1:nfrms);
%   ss = ss{lObj.currMovie};
%   
%   frm = frm(:);
%   tgt = tgt(:);
%   ss = ss(:);
%   tfnan = isnan(ss);
%   frm = frm(~tfnan);
%   tgt = tgt(~tfnan);
%   ss = ss(~tfnan);
%   
%   [ss,idx] = sort(ss,1,'descend');
%   frm = frm(idx);
%   tgt = tgt(idx);
%   
%   mat = [frm tgt ss];
%   tblSusp.Data = mat;
%   pnlSusp.Visible = 'on';
%   
%   if verLessThan('matlab','R2015b') % findjobj doesn't work for >=2015b
%     
%     % make tblSusp column-sortable. 
%     % AL 201510: Tried putting this in opening_fcn but
%     % got weird behavior (findjobj couldn't find jsp)
%     jscrollpane = findjobj(tblSusp);
%     jtable = jscrollpane.getViewport.getView;
%     jtable.setSortable(true);		% or: set(jtable,'Sortable','on');
%     jtable.setAutoResort(true);
%     jtable.setMultiColumnSortable(true);
%     jtable.setPreserveSelectionsAfterSorting(true);
%     % reset ColumnWidth, jtable messes it up
%     cwidth = tblSusp.ColumnWidth;
%     cwidth{end} = cwidth{end}-1;
%     tblSusp.ColumnWidth = cwidth;
%     cwidth{end} = cwidth{end}+1;
%     tblSusp.ColumnWidth = cwidth;
%   
%     tblSusp.UserData = struct('jtable',jtable);   
%   else
%     % none
%   end
%   lObj.updateCurrSusp();
% else
%   tblSusp.Data = cell(0,3);
%   pnlSusp.Visible = 'off';
% end

% function cbkCurrSuspChanged(src,evt)
% lObj = evt.AffectedObject;
% ss = lObj.currSusp;
% if ~isequal(ss,[])
%   lObj.currImHud.updateSusp(ss);
% end

function handles = setupAvailTrackersMenu(handles,tObjs)
% set up menus and put in handles.menu_track_trackers (cell arr)

cellfun(@delete,handles.menu_track_trackers);

nTrker = numel(tObjs);
menuTrks = cell(nTrker,1);
for i=1:nTrker  
  algName = tObjs{i}.algorithmName;
  algLabel = tObjs{i}.algorithmNamePretty;
  enable = onIff(~strcmp(algName,'dpk'));
  mnu = uimenu( ...
    'Parent',handles.menu_track_tracking_algorithm,...
    'Label',algLabel,...
    'Callback',@cbkTrackerMenu,...
    'Tag',sprintf('menu_track_%s',algName),...
    'UserData',i,...
    'enable',enable,...
    'Position',i);
  menuTrks{i} = mnu;
end
handles.menu_track_trackers = menuTrks;

if ~isfield(handles,'menu_track_backend_config')
  % set up first time only, should not change
  handles.menu_track_backend_config = uimenu( ...
    'Parent',handles.menu_track,...
    'Label','GPU/Backend Configuration',...
    'Visible','off',...
    'Tag','menu_track_backend_config');
  moveMenuItemAfter(handles.menu_track_backend_config,handles.menu_track_tracking_algorithm);
  handles.menu_track_backend_config_jrc = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','JRC Cluster',...
    'Callback',@cbkTrackerBackendMenu,...
    'Tag','menu_track_backend_config_jrc',...
    'userdata',DLBackEnd.Bsub);
  handles.menu_track_backend_config_aws = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','AWS Cloud',...
    'Callback',@cbkTrackerBackendMenu,...
    'Tag','menu_track_backend_config_aws',...
    'userdata',DLBackEnd.AWS);
  handles.menu_track_backend_config_docker = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','Docker',...
    'Callback',@cbkTrackerBackendMenu,...
    'Tag','menu_track_backend_config_docker',...
    'userdata',DLBackEnd.Docker);  
  handles.menu_track_backend_config_conda = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','Conda',...
    'Callback',@cbkTrackerBackendMenu,...
    'Tag','menu_track_backend_config_conda',...
    'userdata',DLBackEnd.Conda,...
    'Visible',true);
%   if false %ispc,
%     handles.menu_track_backend_config_docker.Enable = 'off';
%   else
%     handles.menu_track_backend_config_conda.Enable = 'off';
%   end
  handles.menu_track_backend_config_conda.Enable = 'on';
  % KB added menu item to get more info about how to set up
  handles.menu_track_backend_config_moreinfo = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','on',...
    'Label','More information...',...
    'Callback',@cbkTrackerBackendMenuMoreInfo,...
    'Tag','menu_track_backend_config_moreinfo');  
  handles.menu_track_backend_config_test = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','Test backend configuration',...
    'Callback',@cbkTrackerBackendTest,...
    'Tag','menu_track_backend_config_test');
  
  % JRC Cluster 'submenu'
  handles.menu_track_backend_config_jrc_setconfig = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','on',...
    'Label','(JRC) Set number of slots for training...',...
    'Callback',@cbkTrackerBackendSetJRCNSlots,...
    'Tag','menu_track_backend_config_jrc_setconfig');  

  handles.menu_track_backend_config_jrc_setconfig_track = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','off',...
    'Label','(JRC) Set number of slots for tracking...',...
    'Callback',@cbkTrackerBackendSetJRCNSlotsTrack,...
    'Tag','menu_track_backend_config_jrc_setconfig_track');  

  handles.menu_track_backend_config_jrc_additional_bsub_args = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','(JRC) Additional bsub arguments...',...
    'Callback',@cbkTrackerBackendAdditionalBsubArgs,...
    'Tag','menu_track_backend_config_jrc_additional_bsub_args');  

  handles.menu_track_backend_config_jrc_set_singularity_image = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','off',...
    'Label','(JRC) Set Singularity image...',...
    'Callback',@cbkTrackerBackendSetSingularityImage,...
    'Tag','menu_track_backend_config_jrc_set_singularity_image');  

  % AWS submenu (enabled when backend==AWS)
  handles.menu_track_backend_config_aws_setinstance = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','on',...
    'Label','(AWS) Set EC2 instance',...
    'Callback',@cbkTrackerBackendAWSSetInstance,...
    'Tag','menu_track_backend_config_aws_setinstance');  
  
  handles.menu_track_backend_config_aws_configure = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','(AWS) Configure...',...
    'Callback',@cbkTrackerBackendAWSConfigure,...
    'Tag','menu_track_backend_config_aws_configure');  
  
  % Docker 'submenu' (added by KB)
  handles.menu_track_backend_config_setdockerssh = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','on',...
    'Label','(Docker) Set remote host...',...
    'Callback',@cbkTrackerBackendSetDockerSSH,...
    'Tag','menu_track_backend_config_setdockerssh');  

  handles.menu_track_backend_config_docker_image_spec = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Label','(Docker) Set image spec...',...
    'Callback',@cbkTrackerBackendSetDockerImageSpec,...
    'Tag','menu_track_backend_config_docker_image_spec');  

  % Conda submenu
  handles.menu_track_backend_set_conda_env = uimenu( ...
    'Parent',handles.menu_track_backend_config,...
    'Separator','on', ...
    'Label','(Conda) Set environment...',...
    'Callback',@cbkTrackerBackendSetCondaEnv,...
    'Tag','menu_track_backend_set_conda_env');  

end

function handles = setupTrackerMenusListeners(handles,tObj,iTrker)
% Configure listerers-on-current-tracker obj; tracker-specific menus

% delete all existing listeners-to-trackers
cellfun(@delete,handles.listenersTracker);
handles.listenersTracker = cell(0,1);

% UI, is a tracker available
tfTracker = ~isempty(tObj);
onOff = onIff(tfTracker&&handles.labelerObj.isReady);
handles.menu_track.Enable = onOff;
handles.pbTrain.Enable = onOff;
handles.pbTrack.Enable = onOff;
handles.menu_view_hide_predictions.Enable = onOff;
handles.menu_view_show_preds_curr_target_only.Enable = onOff;

menuTrkers = handles.menu_track_trackers;
for i=1:numel(menuTrkers)
  mnu = menuTrkers{i};
  if i==iTrker
    mnu.Checked = 'on';
  else
    mnu.Checked = 'off';
  end
end

listenersNew = cell(0,1);

if tfTracker
  % UI, tracker-specific
  iscpr = strcmp('cpr',tObj.algorithmName);
  onOffCpr = onIff(iscpr);
  handles.menu_track_cpr_show_replicates.Visible = onOffCpr;
  handles.menu_track_cpr_storefull.Visible = onOffCpr;
  handles.menu_track_cpr_view_diagnostics.Visible = onOffCpr;
  
  % FUTURE TODO, enable for DL
  % I think these are obsolete, semi-removing them
  handles.menu_track_training_data_montage.Visible = onOffCpr;
  handles.menu_track_track_and_export.Visible = onOffCpr;
  isDL = ~iscpr;
  onOffDL = onIff(isDL);
  handles.menu_track_backend_config.Visible = onOffDL;
  if isDL
    updateTrackBackendConfigMenuChecked(handles,tObj.lObj);
  end
  
  listenersNew{end+1,1} = tObj.addlistener('trackerInfo','PostSet',@(src1,evt1) cbkTrackerInfoChanged(src1,evt1));
  
  % Listeners, general tracker
  listenersNew{end+1,1} = tObj.addlistener('hideViz','PostSet',...
    @(src1,evt1) cbkTrackerHideVizChanged(src1,evt1,handles.menu_view_hide_predictions)); 
  listenersNew{end+1,1} = tObj.addlistener('showPredsCurrTargetOnly','PostSet',...
    @(src1,evt1) cbkTrackerShowPredsCurrTargetOnlyChanged(src1,evt1,handles.menu_view_show_preds_curr_target_only)); 
  

  % Listeners, algo-specific
  switch tObj.algorithmName
    case 'cpr'
      %  tObj.addlistener('trnDataDownSamp','PostSet',@(src1,evt1) cbkTrackerTrnDataDownSampChanged(src1,evt1,handles));
      
      % NOTE: handles here can get out-of-date but that is ok for now
      listenersNew{end+1,1} = tObj.addlistener('showVizReplicates','PostSet',...
        @(src1,evt1) cbkTrackerShowVizReplicatesChanged(src1,evt1,handles));
      listenersNew{end+1,1} = tObj.addlistener('storeFullTracking','PostSet',...
        @(src1,evt1) cbkTrackerStoreFullTrackingChanged(src1,evt1,handles));
    otherwise
      listenersNew{end+1,1} = tObj.addlistener('trainStart',...
        @(src1,evt1) cbkTrackerTrainStart(src1,evt1,handles));
      listenersNew{end+1,1} = tObj.addlistener('trainEnd',...
        @(src1,evt1) cbkTrackerTrainEnd(src1,evt1,handles));
      listenersNew{end+1,1} = tObj.lObj.addlistener('didSetTrackDLBackEnd', @(src,evt) (cbkTrackerBackEndChanged(src,evt,handles)) );      
      listenersNew{end+1,1} = tObj.addlistener('trackStart',...
        @(src1,evt1) cbkTrackerStart(src1,evt1,handles));
      listenersNew{end+1,1} = tObj.addlistener('trackEnd',...
        @(src1,evt1) cbkTrackerEnd(src1,evt1,handles));
  end
end

handles.listenersTracker = listenersNew;

function updateTrackBackendConfigMenuChecked(handles,lObj)

beType = lObj.trackDLBackEnd.type;
oiBsub = onIff(beType==DLBackEnd.Bsub);
oiDckr = onIff(beType==DLBackEnd.Docker);
oiCnda = onIff(beType==DLBackEnd.Conda);
oiAWS = onIff(beType==DLBackEnd.AWS);
set(handles.menu_track_backend_config_jrc,'checked',oiBsub);
set(handles.menu_track_backend_config_docker,'checked',oiDckr);
set(handles.menu_track_backend_config_conda,'checked',oiCnda, 'Enable', onIff(~ispc()));
set(handles.menu_track_backend_config_aws,'checked',oiAWS);
set(handles.menu_track_backend_config_aws_setinstance,'Enable',oiAWS);
set(handles.menu_track_backend_config_aws_configure,'Enable',oiAWS);
set(handles.menu_track_backend_config_setdockerssh,'Enable',oiDckr);
set(handles.menu_track_backend_config_docker_image_spec,'Enable',oiDckr);
set(handles.menu_track_backend_config_jrc_setconfig,'Enable',oiBsub);
set(handles.menu_track_backend_config_jrc_setconfig_track,'Enable',oiBsub);
set(handles.menu_track_backend_config_jrc_additional_bsub_args,'Enable',oiBsub);
set(handles.menu_track_backend_config_jrc_set_singularity_image,'Enable',oiBsub);
set(handles.menu_track_backend_set_conda_env,'Enable',onIff(beType==DLBackEnd.Conda&&~ispc())) ;

% m = handles.menu_track_backend_config;
% % Menu item ordering seems very buggy. Setting .Position on menu items
% % doesn't work; setting Children once doesn't work, maybe bc of AbortSet. 
% %handles.menu_track_backend_config_aws_setinstance.Separator = 'on';
% drawnow;
% m.Children = [ ...
%   handles.menu_track_backend_config_jrc ...
%   handles.menu_track_backend_config_aws...
%   handles.menu_track_backend_config_docker...
%   handles.menu_track_backend_config_conda...
%   handles.menu_track_backend_config_moreinfo... 
%   handles.menu_track_backend_config_aws_setinstance...
%   handles.menu_track_backend_config_aws_configure...
%   handles.menu_track_backend_config_setdockerssh...
%   handles.menu_track_backend_config_jrc_setconfig...
%   handles.menu_track_backend_config_jrc_setconfig_track...
%   handles.menu_track_backend_config_test...
%   ];
% m.Children = m.Children(end:-1:1);
% handles.menu_track_viz_dataaug.Enable = oiDckr;


function cbkTrackerMenu(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
iTracker = src.UserData;
lObj.trackSetCurrentTracker(iTracker);

function cbkTrackerBackendMenu(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
beType = src.UserData;
be = DLBackEndClass(beType,lObj.trackGetDLBackend());
lObj.trackSetDLBackend(be);

function cbkTrackerBackendMenuMoreInfo(src,evt)

handles = guidata(src);
lObj = handles.labelerObj;

res = web(lObj.DLCONFIGINFOURL,'-new');
if res ~= 0,
  msgbox({'Information on configuring Deep Learning GPU/Backends can be found at'
    'https://github.com/kristinbranson/APT/wiki/Deep-Neural-Network-Tracking.'},...
    'Deep Learning GPU/Backend Information','replace');
end

function cbkTrackerBackendTest(src,evt)

handles = guidata(src);
lObj = handles.labelerObj;

cacheDir = lObj.DLCacheDir; 
assert(exist(cacheDir,'dir')>0,...
  'Deep Learning cache directory ''%s'' does not exist.',cacheDir);

be = lObj.trackDLBackEnd;
be.testConfigUI(cacheDir);

% % is APTCache set?
% hedit.String{end+1} = ''; drawnow;
% hedit.String{end+1} = '** Testing that Deep Track->Saving->CacheDir parameter is set...'; drawnow;
% if isempty(cacheDir),
%   hedit.String{end+1} = 'Deep Track->Saving->CacheDir tracking parameter is not set. Please go to Track->Configure tracking parameters menu to set this.'; drawnow;
%   return;
% end
% % does APTCache exist?
% if ~exist(cacheDir,'dir'),
%   hedit.String{end+1} = sprintf('Deep Track->CacheDir %s did not exist, trying to create it...',cacheDir); drawnow;
%   [tfsucc1,msg1] = mkdir(cacheDir);
%   if ~tfsucc1 || ~exist(cacheDir,'dir'),
%     hedit.String{end+1} = sprintf('Deep Track->CacheDir %s could not be created: %s. Make sure you have access to %s, and/or set CacheDir to a different directory.',cacheDir,msg1,cacheDir); drawnow;
%     return;
%   end
% end
% hedit.String{end+1} = sprintf('Deep Track->Saving->CacheDir set to %s, and exists.',cacheDir); drawnow;
% hedit.String{end+1} = 'SUCCESS!'; drawnow;
      
      
function cbkTrackerBackendAWSSetInstance(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
%be = lObj.trackDLBackEnd;
assert(lObj.trackDLBackEnd.type==DLBackEnd.AWS);

% aws = be.awsec2;
% if ~isempty(aws)
  %[tfsucc,instanceID,pemFile] = lObj.trackDLBackEnd.awsec2.respecifyInstance();
  [tfsucc,~,~,~] = lObj.trackDLBackEnd.awsec2.selectInstance();
% else
%   [tfsucc,instanceID,pemFile] = AWSec2.specifyInstanceUIStc();
% end

if tfsucc
%   lObj.trackDLBackEnd.awsec2.setInstanceID(instanceID);
%   lObj.trackDLBackEnd.awsec2.setPemFile(pemFile);
%   aws = AWSec2(pemFile,'instanceID',instanceID);
%   be.awsec2 = aws;
  %aws.checkInstanceRunning('throwErrs',false);
  %lObj.trackSetDLBackend(be);
end

function cbkTrackerBackendAWSConfigure(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
%be = lObj.trackDLBackEnd;
assert(lObj.trackDLBackEnd.type==DLBackEnd.AWS);

%aws = be.awsec2;
%if ~isempty(aws)
  [tfsucc,~,~,reason] = lObj.trackDLBackEnd.awsec2.selectInstance('canlaunch',1,...
    'canconfigure',2,'forceSelect',1);
  if ~tfsucc,
    warning('Problem configuring: %s',reason);
  end
%else
%  [tfsucc,keyName,pemFile] = AWSec2.specifySSHKeyUIStc();
%end

if tfsucc  
%   aws = AWSec2(pemFile,'keyName',keyName);
%   be.awsec2 = aws;
  %aws.checkInstanceRunning('throwErrs',false);
%   lObj.trackSetDLBackend(be);
end

function cbkTrackerBackendSetJRCNSlots(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
n = inputdlg('Number of cluster slots for training','a',1,{num2str(lObj.tracker.jrcnslots)});
if isempty(n)
  return;
end
n = str2double(n{1});
if isnan(n)
  return;
end
lObj.tracker.setJrcnslots(n);

function cbkTrackerBackendSetJRCNSlotsTrack(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
n = inputdlg('Number of cluster slots for tracking','a',1,{num2str(lObj.tracker.jrcnslotstrack)});
if isempty(n)
  return;
end
n = str2double(n{1});
if isnan(n)
  return;
end
lObj.tracker.setJrcnslotstrack(n);


function cbkTrackerBackendAdditionalBsubArgs(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
original_value = lObj.get_backend_property('jrcAdditionalBsubArgs') ;
dialog_result = inputdlg({'Addtional bsub arguments:'},'Additional bsub arguments...',1,{original_value});
if isempty(dialog_result)
  return
end
new_value = dialog_result{1};
try
  lObj.set_backend_property('jrcAdditionalBsubArgs', new_value) ;
catch exception
  if strcmp(exception.identifier, 'APT:invalidValue') ,
    uiwait(errordlg(exception.message));
  else
    rethrow(exception);
  end
end


function cbkTrackerBackendSetDockerSSH(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
%be = lObj.trackDLBackEnd;
assert(lObj.trackDLBackEnd.type==DLBackEnd.Docker);
drh = lObj.trackDLBackEnd.dockerremotehost;
if isempty(drh),
  defans = 'Local';
else
  defans = 'Remote';
end

res = questdlg('Run docker on your Local machine, or SSH to a Remote machine?',...
  'Set Docker Remote Host','Local','Remote','Cancel',defans);
if strcmpi(res,'Cancel'),
  return;
end
if strcmpi(res,'Remote'),
  res = inputdlg({'Remote Host Name:'},'Set Docker Remote Host',1,{drh});
  if isempty(res) || isempty(res{1}),
    return;
  end
  lObj.trackDLBackEnd.dockerremotehost = res{1};
else
  lObj.trackDLBackEnd.dockerremotehost = '';
end

ischange = ~strcmp(drh,lObj.trackDLBackEnd.dockerremotehost);

if ischange,
  res = questdlg('Test new Docker configuration now?','Test Docker configuration','Yes','No','Yes');
  if strcmpi(res,'Yes'),
    try
      tfsucc = lObj.trackDLBackEnd.testDockerConfig();
    catch ME,
      tfsucc = false;
      disp(getReport(ME));
    end
    if ~tfsucc,
      res = questdlg('Test failed. Revert to previous Docker settings?','Backend test failed','Yes','No','Yes');
      if strcmpi(res,'Yes'),
        lObj.trackDLBackEnd.dockerremotehost = drh;
      end
    end
  end
end


function cbkTrackerBackendSetDockerImageSpec(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
original_full_image_spec = lObj.get_backend_property('dockerimgfull') ;
dialog_result = inputdlg({'Docker Image Spec:'},'Set image spec...',1,{original_full_image_spec});
if isempty(dialog_result)
  return
end
new_full_image_spec = dialog_result{1};
try
  lObj.set_backend_property('dockerimgfull', new_full_image_spec) ;
catch exception
  if strcmp(exception.identifier, 'APT:invalidValue') ,
    uiwait(errordlg(exception.message));
  else
    rethrow(exception);
  end
end


function cbkTrackerBackendSetCondaEnv(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
original_value = lObj.get_backend_property('condaEnv') ;
dialog_result = inputdlg({'Conda environment:'},'Set environment...',1,{original_value});
if isempty(dialog_result)
  return
end
new_value = dialog_result{1};
try
  lObj.set_backend_property('condaEnv', new_value) ;
catch exception
  if strcmp(exception.identifier, 'APT:invalidValue') ,
    uiwait(errordlg(exception.message));
  else
    rethrow(exception);
  end
end


function cbkTrackerBackendSetSingularityImage(src,evt)
handles = guidata(src);
lObj = handles.labelerObj;
original_value = lObj.get_backend_property('singularity_image_path') ;
filter_spec = {'*.sif','Singularity Images (*.sif)'; ...
              '*',  'All Files (*)'} ;
[file_name, path_name] = uigetfile(filter_spec, 'Set Singularity Image...', original_value) ;
if isnumeric(file_name)
  return
end
new_value = fullfile(path_name, file_name) ;
try
  lObj.set_backend_property('singularity_image_path', new_value) ;
catch exception
  if strcmp(exception.identifier, 'APT:invalidValue') ,
    uiwait(errordlg(exception.message));
  else
    rethrow(exception);
  end
end


function cbkTrackersAllChanged(src, evt)
lObj = src ;
if ~lObj.isinit ,
  handles = lObj.gdata ;
  handles = setupAvailTrackersMenu(handles, lObj.trackersAll) ;
  guidata(handles.figure, handles) ;
  cellfun(@deactivate, lObj.trackersAll) ;
  cbkCurrTrackerChanged(src, evt) ;  % current tracker object depends on lObj.trackersAll
end


function cbkCurrTrackerPreChanged(src,evt)
lObj = src ;
if lObj.isinit ,
  return
end 
tObj = lObj.tracker ;
if ~isempty(tObj) ,
  tObj.deactivate() ;
end


function cbkCurrTrackerChanged(src, evt)
lObj = src ;
if lObj.isinit ,
  return
end 
handles = lObj.gdata ;
tObj = lObj.tracker ;
iTrker = lObj.currTracker ;
handles = setupTrackerMenusListeners(handles, tObj, iTrker) ;
% tracker changed, update tracker info
if ~isempty(tObj),
  tObj.activate();
  tObj.updateTrackerInfo();
end
handles.labelTLInfo.setTracker(tObj);
guidata(handles.figure,handles);

function cbkTrackModeIdxChanged(src,evt)
lObj = src ;
if lObj.isinit ,
  return
end
hPUM = lObj.gdata.pumTrack;
hPUM.Value = lObj.trackModeIdx;
try %#ok<TRYNC>
  fullstrings = getappdata(hPUM,'FullStrings');
  set(lObj.gdata.text_framestotrackinfo,'String',fullstrings{hPUM.Value});
end
% Edge case: conceivably, pumTrack.Strings may not be updated (eg for a
% noTrx->hasTrx transition before this callback fires). In this case,
% hPUM.Value (trackModeIdx) will be out of bounds and a warning till be
% thrown, PUM will not be displayed etc. However when hPUM.value is
% updated, this should resolve.

function cbkTrackerNFramesChanged(src,evt)
lObj = src ;
if lObj.isinit ,
  return
end
setPUMTrackStrs(lObj) ;

function setPUMTrackStrs(lObj)
if lObj.hasTrx
  mfts = MFTSetEnum.TrackingMenuTrx;
else
  mfts = MFTSetEnum.TrackingMenuNoTrx;
end
menustrs = arrayfun(@(x)x.getPrettyStr(lObj),mfts,'uni',0);
if ispc || ismac
  menustrs_compact = arrayfun(@(x)x.getPrettyStrCompact(lObj),mfts,'uni',0);
else
  % iss #161
  menustrs_compact = arrayfun(@(x)x.getPrettyStrMoreCompact(lObj),mfts,'uni',0);
end
hPUM = lObj.gdata.pumTrack;
hPUM.String = menustrs_compact;
setappdata(hPUM,'FullStrings',menustrs);
if lObj.trackModeIdx>numel(menustrs)
  lObj.trackModeIdx = 1;
end

hFig = lObj.gdata.figure;
hFig.SizeChangedFcn(hFig,[]);

function pumTrack_Callback(hObj,edata,handles)
lObj = handles.labelerObj;
lObj.trackModeIdx = hObj.Value;

function mftset = getTrackMode(handles)
idx = handles.pumTrack.Value;
% Note, .TrackingMenuNoTrx==.TrackingMenuTrx(1:K), so we can just index
% .TrackingMenuTrx.
mfts = MFTSetEnum.TrackingMenuTrx;
mftset = mfts(idx);

function cbkMovieCenterOnTargetChanged(src,~)
lObj = src ;
tf = lObj.movieCenterOnTarget;
mnu = lObj.gdata.menu_view_trajectories_centervideoontarget;
mnu.Checked = onIff(tf);
if tf,
  lObj.videoZoom(lObj.targetZoomRadiusDefault);
end

function cbkMovieRotateTargetUpChanged(src,evt)
lObj = src ;
tf = lObj.movieRotateTargetUp;
if tf
  ax = lObj.gdata.axes_curr;
  warnst = warning('off','LabelerGUI:axDir');
  % When axis is in image mode, ydir should be reversed!
  ax.XDir = 'normal';
  ax.YDir = 'reverse';

%   for f={'XDir' 'YDir'},f=f{1}; %#ok<FXSET>
%     if strcmp(ax.(f),'reverse')
%       warningNoTrace('LabelerGUI:ax','Setting main axis .%s to ''normal''.',f);
%       ax.(f) = 'normal';
%     end
%   end

  warning(warnst);
end
mnu = lObj.gdata.menu_view_rotate_video_target_up;
mnu.Checked = onIff(tf);
lObj.UpdatePrevAxesDirections();

function slider_frame_Callback(hObject,evt,varargin)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

debugtiming = false;
if debugtiming,
  starttime = tic() ;
end

handles = guidata(hObject);
lObj = handles.labelerObj;

if ~lObj.hasProject
  set(hObject,'Value',0);  
  return;
end
if ~lObj.hasMovie
  set(hObject,'Value',0);  
  msgbox('There is no movie open.');
  return;
end

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

if debugtiming,
  fprintf('Slider callback setting to frame %d took %f seconds\n',f,toc(starttime));
end

function slider_frame_CreateFcn(hObject,~,~)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edit_frame_Callback(hObject,~,handles)
if ~checkProjAndMovieExist(handles)
  return;
end

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
if ~checkProjAndMovieExist(handles)
  return;
end
if handles.labelerObj.doesNeedSave ,
  res = questdlg('Project has unsaved changes. Save before training?','Save Project','Save As','No','Cancel','Save As');
  if strcmp(res,'Cancel')
    return
  elseif strcmp(res,'Save As')
    menu_file_saveas_Callback(hObject, eventdata, handles)
  end    
end

handles.labelerObj.setStatus('Training...');
drawnow;
[tfCanTrain,reason] = handles.labelerObj.trackCanTrain();
if ~tfCanTrain,
  errordlg(['Error training tracker: ',reason],'Error training tracker');
  handles.labelerObj.clearStatus();
  return;
end

%handles.labelerObj.trackSetAutoParams();

fprintf('Training started at %s...\n',datestr(now));
oc1 = onCleanup(@()(handles.labelerObj.clearStatus()));
wbObj = WaitBarWithCancel('Training');
oc2 = onCleanup(@()delete(wbObj));
centerOnParentFigure(wbObj.hWB,handles.figure);
try
  handles.labelerObj.trackRetrain('retrainArgs',{'wbObj',wbObj});
catch ME
  errordlg(ME.message,'Error while training','modal');
  disp(getReport(ME));
  return;
end
if wbObj.isCancel
  msg = wbObj.cancelMessage('Training canceled');
  msgbox(msg,'Train');
end
  
function pbTrack_Callback(hObject, eventdata, handles)
if ~checkProjAndMovieExist(handles)
  return;
end
handles.labelerObj.setStatus('Tracking...');
tm = getTrackMode(handles);
tblMFT = tm.getMFTable(handles.labelerObj,'istrack',true);
if isempty(tblMFT),
  msgbox('All frames tracked.','Track');
  handles.labelerObj.clearStatus() ;
  return;
end
[tfCanTrack,reason] = handles.labelerObj.trackCanTrack(tblMFT);
if ~tfCanTrack,
  errordlg(['Error tracking: ',reason],'Error tracking');
  handles.labelerObj.clearStatus();
  return;
end
fprintf('Tracking started at %s...\n',datestr(now));
wbObj = WaitBarWithCancel('Tracking');
centerOnParentFigure(wbObj.hWB,handles.figure);
oc = onCleanup(@()delete(wbObj));
if handles.labelerObj.maIsMA
  handles.labelerObj.track(tblMFT,'wbObj',wbObj,'track_type','detect');
else
  handles.labelerObj.track(tblMFT,'wbObj',wbObj);
end
if wbObj.isCancel
  msg = wbObj.cancelMessage('Tracking canceled');
  msgbox(msg,'Track');
end
handles.labelerObj.clearStatus();

function pbClear_Callback(hObject, eventdata, handles)

if ~checkProjAndMovieExist(handles)
  return;
end
handles.labelerObj.lblCore.clearLabels();
handles.labelerObj.CheckPrevAxesTemplate();

function tbAccept_Callback(hObject, eventdata, handles)
% debugtiming = false;
% if debugtiming,
%   starttime = tic;
% end

if ~checkProjAndMovieExist(handles)
  return;
end
lc = handles.labelerObj.lblCore;
switch lc.state
  case LabelState.ADJUST
    lc.acceptLabels();
    %handles.labelerObj.InitializePrevAxesTemplate();
  case LabelState.ACCEPTED
    lc.unAcceptLabels();    
    %handles.labelerObj.CheckPrevAxesTemplate();
  otherwise
    assert(false);
end

% if debugtiming,
%   fprintf('toggleAccept took %f seconds\n',toc(starttime));
% end

function cbkTblTrxCellSelection(src,evt) %#ok<*DEFNU>
% Current/last row selection is maintained in hObject.UserData

handles = guidata(src.Parent);
lObj = handles.labelerObj;
if ~(lObj.hasTrx || lObj.maIsMA)
  return;
end

rows = evt.Indices;
rows = rows(:,1); % AL20210514: rows is nx2; columns are {rowidxs,colidxs} at least in 2020b
%rowsprev = src.UserData;
src.UserData = rows;
dat = get(src,'Data');

if isscalar(rows)
  idx = dat{rows(1),1};
  lObj.setTarget(idx);
  %lObj.labelsOtherTargetHideAll();
else
  % 20210514 Skipping this for now; possible performance hit
  
  % addon to existing selection
  %rowsnew = setdiff(rows,rowsprev);  
  %idxsnew = cell2mat(dat(rowsnew,1));
  %lObj.labelsOtherTargetShowIdxs(idxsnew);
end

hlpRemoveFocus(src,handles);

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

function cbkTblFramesCellSelection(src,evt)
handles = guidata(src.Parent);
lObj = handles.labelerObj;
row = evt.Indices;
if ~isempty(row)
  row = row(1);
  dat = get(src,'Data');
  lObj.setFrame(dat{row,1},'changeTgtsIfNec',true);
end

hlpRemoveFocus(src,handles);

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
if (strcmp(ax.XDir,'reverse') || strcmp(ax.YDir,'normal')) && ...
    handles.labelerObj.movieRotateTargetUp
  warningNoTrace('LabelerGUI:axDir',...
    'Main axis ''XDir'' or ''YDir'' is set to be flipped and .movieRotateTargetUp is set. Graphics behavior may be unexpected; proceed at your own risk.');
end

function scroll_callback(hObject,eventdata,lObj)
gdata = lObj.gdata;
ivw = find(hObject==gdata.figs_all);
ax = gdata.axes_all(ivw);
curp = get(ax,'CurrentPoint');
xlim = get(ax,'XLim');
ylim = get(ax,'YLim');
if (curp(1,1)< xlim(1)) || (curp(1,1)>xlim(2))
  return
end
if (curp(1,2)< ylim(1)) || (curp(1,2)>ylim(2))
  return
end
scrl = 1.2;
% scrl = scrl^eventdata.VerticalScrollAmount;
if eventdata.VerticalScrollCount>0
  scrl = 1/scrl;
end
him = gdata.images_all(ivw);
imglimx = get(him,'XData');
imglimy = get(him,'YData');
xlim(1) = max(imglimx(1),curp(1,1)-(curp(1,1)-xlim(1))/scrl);
xlim(2) = min(imglimx(2),curp(1,1)+(xlim(2)-curp(1,1))/scrl);
ylim(1) = max(imglimy(1),curp(1,2)-(curp(1,2)-ylim(1))/scrl);
ylim(2) = min(imglimy(2),curp(1,2)+(ylim(2)-curp(1,2))/scrl);
axis(ax,[xlim(1),xlim(2),ylim(1),ylim(2)]);
% fprintf('Scrolling %d!!\n',eventdata.VerticalScrollAmount)

% function dragstart_callback(hObject,eventdata,~)
% fprintf('Drag on\n')
% h = guidata(hObject);
% curp = get(h.axes_curr,'CurrentPoint');
% xlim = get(h.axes_curr,'XLim');
% ylim = get(h.axes_curr,'YLim');
% if (curp(1,1)< xlim(1)) || (curp(1,1)>xlim(2))
%   return
% end
% if (curp(1,2)< ylim(1)) || (curp(1,2)>ylim(2))
%   return
% end
% h.labelerObj.drag = true;
% h.labelerObj.drag_pt = [curp(1,1),curp(1,2)];
% set(hObject,'Windowbuttonmotionfcn',@drag_callback);
% set(hObject,'WindowbuttonUpFcn',@dragend_callback);

% function drag_callback(hObject,evendata,~)
% h = guidata(hObject);
% if ~h.labelerObj.drag
%   return
% end
% fprintf('Dragging\n');
% xlim = get(h.axes_curr,'XLim');
% ylim = get(h.axes_curr,'YLim');
% curp = get(h.axes_curr,'CurrentPoint');
% dx = curp(1,1)-h.labelerObj.drag_pt(1);
% dy = curp(1,2) - h.labelerObj.drag_pt(2);
% imglimx = get(h.image_curr,'XData');
% imglimy = get(h.image_curr,'YData');
% xlim(1) = max(imglimx(1),xlim(1)-dx);
% xlim(2) = min(imglimx(2),xlim(2)-dx);
% ylim(1) = max(imglimy(1),ylim(1)-dy);
% ylim(2) = min(imglimy(2),ylim(2)-dy);
% axis(h.axes_curr,[xlim(1),xlim(2),ylim(1),ylim(2)]);


% function dragend_callback(hObject,eventdata,~)
% h = guidata(hObject);
% h.labelerObj.unsetdrag();
% fprintf('Drag off\n')


function sldZoom_Callback(hObject, eventdata, ~)
% log(zoomrad) = logzoomradmax + sldval*(logzoomradmin-logzoomradmax)
handles = guidata(hObject);

if ~checkProjAndMovieExist(handles)
  return;
end

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
handles = guidata(src);
if evt.Axes == handles.axes_prev,
  handles.labelerObj.UpdatePrevAxesLimits();
end

function cbkPostPan(src,evt)
handles = guidata(src);
if evt.Axes == handles.axes_prev,
  handles.labelerObj.UpdatePrevAxesLimits();
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
% TODO this is broken!!
lObj.videoCenterOnCurrTarget();
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
if ~checkProjAndMovieExist(handles)
  return;
end
tl = handles.labelTLInfo;
tl.selectOn = hObject.Value;

function pbClearSelection_Callback(hObject, eventdata, handles)
if ~checkProjAndMovieExist(handles)
  return;
end
tl = handles.labelTLInfo;
tl.selectClearSelection();

function cbklabelTLInfoSelectOn(src,evt)
lblTLObj = evt.AffectedObject;
tb = lblTLObj.lObj.gdata.tbTLSelectMode;
tb.Value = lblTLObj.selectOn;

function cbklabelTLInfoPropsUpdated(src,evt)
% Update the props dropdown menu and timeline.
labelTLInfo = evt.AffectedObject;
props = labelTLInfo.getPropsDisp();
set(labelTLInfo.lObj.gdata.pumInfo,'String',props);

function cbklabelTLInfoPropTypesUpdated(src,evt)
% Update the props dropdown menu and timeline.
labelTLInfo = evt.AffectedObject;
proptypes = labelTLInfo.getPropTypesDisp();
set(labelTLInfo.lObj.gdata.pumInfo_labels,'String',proptypes);

function cbkFreezePrevAxesToMainWindow(src,evt)
handles = guidata(src);
handles.labelerObj.setPrevAxesMode(PrevAxesMode.FROZEN);
function cbkUnfreezePrevAxes(src,evt)
handles = guidata(src);
handles.labelerObj.setPrevAxesMode(PrevAxesMode.LASTSEEN);

%% menu
function menu_file_quick_open_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
controller = handles.controller ;
if controller.raiseUnsavedChangesDialogIfNeeded() ,
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
% 
function menu_file_new_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.setStatus('Starting New Project');
controller = handles.controller ;
if controller.raiseUnsavedChangesDialogIfNeeded() ,
  cfg = ProjectSetup(handles.figure);
  if ~isempty(cfg)    
    lObj.setStatus('Configuring New Project') ;
    lObj.initFromConfig(cfg);
    lObj.projNew(cfg.ProjectName);
    lObj.setStatus('Adding Movies') ;
    handles = lObj.gdata; % initFromConfig, projNew have updated handles
    menu_file_managemovies_Callback([],[],handles);  %all this does is make the movie manager visible
  end  
end
handles.labelerObj.clearStatus();

function menu_file_save_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj ;
lObj.setStatus('Saving project...');
handles.labelerObj.projSaveSmart();
handles.labelerObj.projAssignProjNameFromProjFileIfAppropriate();
handles.labelerObj.clearStatus()

function menu_file_saveas_Callback(hObject, eventdata, handles)
handles.labelerObj.setStatus('Saving project...');
handles.labelerObj.projSaveAs();
handles.labelerObj.projAssignProjNameFromProjFileIfAppropriate();
handles.labelerObj.clearStatus()

function menu_file_load_Callback(hObject, eventdata, handles)

handles.labelerObj.setStatus('Loading Project...') ;
%EnableControls(handles,'projectloaded');
lObj = handles.labelerObj;
controller = handles.controller ;
if controller.raiseUnsavedChangesDialogIfNeeded() ,
  currMovInfo = lObj.projLoad();
  if ~isempty(currMovInfo)
    handles = lObj.gdata; % projLoad updated stuff
    handles.movieMgr.setVisible(true);
    wstr = sprintf('Could not find file for movie(set) %d: %s.\n\nProject opened with no movie selected. Double-click a row in the MovieManager or use the ''Switch to Movie'' button to start working on a movie.',...
      currMovInfo.iMov,currMovInfo.badfile);
    warndlg(wstr,'Movie not found','modal');
  end
end
handles.labelerObj.clearStatus()

function tfcontinue = hlpSave(labelerObj)
tfcontinue = true;

if ~verLessThan('matlab','9.6') && batchStartupOptionUsed
  return;
end

OPTION_SAVE = 'Save first';
OPTION_PROC = 'Proceed without saving';
OPTION_CANC = 'Cancel';
if labelerObj.doesNeedSave ,
  res = questdlg('You have unsaved changes to your project. If you proceed without saving, your changes will be lost.',...
    'Unsaved changes',OPTION_SAVE,OPTION_PROC,OPTION_CANC,OPTION_SAVE);
  switch res
    case OPTION_SAVE
      labelerObj.projSaveSmart();
      labelerObj.projAssignProjNameFromProjFileIfAppropriate();
    case OPTION_CANC
      tfcontinue = false;
    case OPTION_PROC
      % none
  end
end

function menu_file_managemovies_Callback(~,~,handles)
if isfield(handles,'movieMgr')
  handles.movieMgr.setVisible(true);
else
  handles.labelerObj.lerror('LabelerGUI:movieMgr','Please create or load a project.');
end

function menu_file_import_labels_trk_curr_mov_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if ~lObj.hasMovie
  handles.labelerObj.lerror('LabelerGUI:noMovie','No movie is loaded.');
end
lObj.gtThrowErrIfInGTMode();
iMov = lObj.currMovie;
haslbls1 = lObj.labelPosMovieHasLabels(iMov); % TODO: method should be unnec
haslbls2 = lObj.movieFilesAllHaveLbls(iMov)>0;
assert(haslbls1==haslbls2);
if haslbls1
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
handles.labelerObj.labelImportTrkPromptGenericSimple(iMov,...
  'labelImportTrk','gtok',false);

function menu_file_import_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if ~lObj.hasMovie
  handles.labelerObj.lerror('LabelerGUI:noMovie','No movie is loaded.');
end
iMov = lObj.currMovie; % gt-aware
handles.labelerObj.setStatus('Importing tracking results...');
lObj.labelImportTrkPromptGenericSimple(iMov,'labels2ImportTrk','gtok',true);
handles.labelerObj.clearStatus();

function menu_file_export_labels_trks_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
[tfok,rawtrkname] = lObj.getExportTrkRawnameUI('labels',true);
if ~tfok
  return;
end
handles.labelerObj.setStatus('Exporting tracking results...');
lObj.labelExportTrk(1:lObj.nmoviesGTaware,'rawtrkname',rawtrkname);
handles.labelerObj.clearStatus();

function menu_file_export_labels_table_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
fname = lObj.getDefaultFilenameExportLabelTable();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return;
end
fname = fullfile(p,f);  
VARNAME = 'tblLbls';
s = struct();
s.(VARNAME) = lObj.labelGetMFTableLabeled('useMovNames',true); 
save(fname,'-mat','-struct','s');
fprintf('Saved table ''%s'' to file ''%s''.\n',VARNAME,fname);

function menu_file_import_labels_table_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lastFile = RC.getprop('lastLabelMatfile');
if isempty(lastFile)
  lastFile = pwd;
end
[fname,pth] = uigetfile('*.mat','Load Labels',lastFile);
if isequal(fname,0)
  return;
end
fname = fullfile(pth,fname);
t = loadSingleVariableMatfile(fname);
lObj.labelPosBulkImportTbl(t);
fprintf('Loaded %d labeled frames from file ''%s''.\n',height(t),fname);

function menu_file_export_stripped_lbl_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
fname = lObj.getDefaultFilenameExportStrippedLbl();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return;
end
fname = fullfile(p,f);
handles.labelerObj.setStatus(sprintf('Exporting training data to %s',fname));
lObj.projExportTrainData(fname)
fprintf('Saved training data to file ''%s''.\n',fname);
handles.labelerObj.clearStatus();

function menu_file_crop_mode_Callback(hObject,evtdata,handles)

lObj = handles.labelerObj;

if ~isempty(lObj.tracker) && ~lObj.gtIsGTMode && lObj.labelPosMovieHasLabels(lObj.currMovie),
  res = questdlg('Frames of the current movie are labeled. Editing the crop region for this movie will cause trackers to be reset. Continue?');
  if ~strcmpi(res,'Yes'),
    return;
  end
end

handles.labelerObj.setStatus('Switching crop mode...');
lObj.cropSetCropMode(~lObj.cropIsCropMode);
handles.labelerObj.clearStatus();

function menu_file_clean_tempdir_Callback(hObject,evtdata,handles)

handles.labelerObj.setStatus('Deleting temp directories...');
handles.labelerObj.projRemoveOtherTempDirs();
handles.labelerObj.clearStatus();

function menu_file_bundle_tempdir_Callback(hObject,evtdata,handles)
handles.labelerObj.setStatus('Bundling the temp directory...');
handles.labelerObj.projBundleTempDir();
handles.labelerObj.clearStatus();


function menu_help_Callback(hObject, eventdata, handles)

function menu_help_labeling_actions_Callback(hObject, eventdata, handles)
lblCore = handles.labelerObj.lblCore;
if isempty(lblCore)
  h = 'Please open a movie first.';
else
  h = lblCore.getLabelingHelp();
end
msgbox(h,'Labeling Actions','help',struct('Interpreter','tex','WindowStyle','replace'));

function menu_help_about_Callback(hObject, eventdata, handles)
about(handles.labelerObj);

function menu_setup_sequential_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_sequential_add_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_template_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_highthroughput_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_multiview_calibrated_mode_2_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menu_setup_multianimal_mode_Callback(hObject,eventdata,handles)
menuSetupLabelModeCbkGeneric(hObject,handles);
function menuSetupLabelModeCbkGeneric(hObject,handles)
lblMode = handles.setupMenu2LabelMode.(hObject.Tag);
handles.labelerObj.labelingInit('labelMode',lblMode);

function menu_setup_label_overlay_montage_Callback(hObject,evtdata,handles)
handles.labelerObj.setStatus('Plotting all labels on one axes to visualize label distribution...');
lObj = handles.labelerObj;
if lObj.hasTrx
  lObj.labelOverlayMontage();
  lObj.labelOverlayMontage('ctrMeth','trx');
  lObj.labelOverlayMontage('ctrMeth','trx','rotAlignMeth','trxtheta');
  % could also use headtail for centering/alignment but skip for now.  
else % lObj.maIsMA, or SA-non-trx
  lObj.labelOverlayMontage();
  if ~lObj.isMultiView
    lObj.labelOverlayMontage('ctrMeth','centroid');
    tfHTdefined = ~isempty(lObj.skelHead) && ~isempty(lObj.skelTail);
    if tfHTdefined  
      lObj.labelOverlayMontage('ctrMeth','centroid','rotAlignMeth','headtail');
    else
      warningNoTrace('For aligned overlays, define head/tail points in Track>Landmark Paraneters.');
    end
  end
end
handles.labelerObj.clearStatus();

% function menu_setup_label_overlay_montage_trx_centered_Callback(hObject,evtdata,handles)
% 
% handles.labelerObj.setStatus('Plotting all labels on one axes to visualize label distribution...');
% lObj = handles.labelerObj;
% hFig(1) = lObj.labelOverlayMontage('ctrMeth','trx','rotAlignMeth','none'); 
% try
%   hFig(2) = lObj.labelOverlayMontage('ctrMeth','trx',...
%     'rotAlignMeth','headtail','hFig0',hFig(1)); 
% catch ME
%   warningNoTrace('Could not create head-tail aligned montage: %s',ME.message);
%   hFig(2) = figurecascaded(hFig(1));
% end
% hFig(3) = lObj.labelOverlayMontage('ctrMeth','trx',...
%   'rotAlignMeth','trxtheta','hFig0',hFig(2)); %#ok<NASGU>
% handles.labelerObj.clearStatus();

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

function menu_setup_ma_twoclick_align_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lc = lObj.lblCore;
tftc = ~lc.tcOn;
lObj.isTwoClickAlign = tftc; % store the state
lc.setTwoClickOn(tftc);
hObject.Checked = onIff(tftc); % skip listener business for now

function menu_setup_set_labeling_point_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
ipt = lObj.lblCore.iPoint;
ret = inputdlg('Select labeling point','Point number',1,{num2str(ipt)});
if isempty(ret)
  return;
end
ret = str2double(ret{1});
lObj.lblCore.setIPoint(ret);

function set_use_calibration(handles,v)

lObj = handles.labelerObj;
lc = lObj.lblCore;
if lc.supportsCalibration,
  lc.setShowCalibration(v);
end
handles.menu_setup_use_calibration.Checked = onIff(lc.showCalibration);

function menu_setup_use_calibration_Callback(hObject, eventdata, handles)

lObj = handles.labelerObj;
lc = lObj.lblCore;
if lc.supportsCalibration,
  lc.toggleShowCalibration();
  hObject.Checked = onIff(lc.showCalibration);
else
  hObject.Checked = 'off';
end

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

crObj = CalRig.loadCreateCalRigObjFromFile(fname);

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
  lObj.viewCalSetProjWide(crObj);%,'tfSetViewSizes',tfSetViewSizes);
else
  lObj.viewCalSetCurrMovie(crObj);%,'tfSetViewSizes',tfSetViewSizes);
end
set_use_calibration(handles,true);

RC.saveprop('lastCalibrationFile',fname);

% function menu_setup_unlock_all_frames_Callback(hObject, eventdata, handles)
% handles.labelerObj.labelPosSetAllMarked(false);
% function menu_setup_lock_all_frames_Callback(hObject, eventdata, handles)
% handles.labelerObj.labelPosSetAllMarked(true);

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
  lObj.clim_manual = clim;
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
  fignames{1} = handles.txMoviename.String;
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

function menu_view_show_bgsubbed_frames_Callback(hObject,evtdata,handles)
tf = ~strcmp(hObject.Checked,'on');
lObj = handles.labelerObj;
lObj.movieViewBGsubbed = tf;

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
controller = handles.controller ;
controller.quitRequested() ;
%CloseGUI(handles);

% function cbkShowPredTxtLblChanged(src,evt)
% lObj = evt.AffectedObject;
% handles = lObj.gdata;
% onOff = onIff(~lObj.showPredTxtLbl);
% handles.menu_view_showhide_advanced_hidepredtxtlbls.Checked = onOff;

function cbkShowSkeletonChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
hasSkeleton = ~isempty(lObj.skeletonEdges) ;
isChecked = onIff(hasSkeleton && lObj.showSkeleton) ;
set(handles.menu_view_showhide_skeleton, 'Enable', hasSkeleton, 'Checked', isChecked) ;

function cbkShowMaRoiChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
onOff = onIff(lObj.showMaRoi);
handles.menu_view_showhide_maroi.Checked = onOff;

function cbkShowMaRoiAuxChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
onOff = onIff(lObj.showMaRoiAux);
handles.menu_view_showhide_maroiaux.Checked = onOff;

function cbkShowTrxChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
onOff = onIff(~lObj.showTrx);
handles.menu_view_hide_trajectories.Checked = onOff;

function cbkShowOccludedBoxChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
onOff = onIff(lObj.showOccludedBox);
handles.menu_view_occluded_points_box.Checked = onOff;
set([handles.text_occludedpoints,handles.axes_occ],'Visible',onOff);

function cbkShowTrxCurrTargetOnlyChanged(src,evt)
lObj = src ;
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
  handles.labelerObj.UpdatePrevAxesDirections();
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
    handles.labelerObj.UpdatePrevAxesDirections();
  end
end
function menu_view_show_axes_toolbar_Callback(hObject, eventdata, handles)
ax = handles.axes_curr;
if strcmp(hObject.Checked,'on')
  onoff = 'off';
else
  onoff = 'on';
end
ax.Toolbar.Visible = onoff;
hObject.Checked = onoff;
% For now not listening to ax.Toolbar.Visible for cmdline changes


function menu_view_fit_entire_image_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms,true(1,numel(hAxs)));
handles.labelerObj.movieCenterOnTarget = false;


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
tfAxLimsSpecifiedInCfg = ViewConfig.setCfgOnViews(viewCfg,...
  handles.figs_all,axs,handles.images_all,handles.axes_prev);
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
function menu_view_show_preds_curr_target_only_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
tracker = lObj.tracker;
if ~isempty(tracker)
  tracker.showPredsCurrTargetOnlyToggle();
end

function menu_view_hide_imported_predictions_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.labels2VizToggle();

function menu_view_show_imported_preds_curr_target_only_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.labels2VizSetShowCurrTargetOnly(~lObj.labels2ShowCurrTargetOnly);

function cbkTrackerShowVizReplicatesChanged(hObject, eventdata, handles)
handles.menu_track_cpr_show_replicates.Checked = ...
  onIff(handles.labelerObj.tracker.showVizReplicates);

function cbkTrackerTrainStart(hObject, eventdata, handles)
lObj = handles.labelerObj;
algName = lObj.tracker.algorithmName;
%algLabel = lObj.tracker.algorithmNamePretty;
backend = lObj.trackDLBackEnd.prettyName;
handles.txBGTrain.String = sprintf('%s training on %s (started %s)',algName,backend,datestr(now(),'HH:MM'));
handles.txBGTrain.ForegroundColor = handles.busystatuscolor;
handles.txBGTrain.FontWeight = 'normal';
handles.txBGTrain.Visible = 'on';

function cbkTrackerTrainEnd(hObject, eventdata, handles)
handles.txBGTrain.Visible = 'off';
handles.txBGTrain.String = 'Idle';
handles.txBGTrain.ForegroundColor = handles.idlestatuscolor;

lObj = handles.labelerObj;
val = true;
str = 'Tracker trained';
%lObj.needsSave = true;
%cbkSaveNeeded(lObj,val,str);
lObj.setDoesNeedSave(val, str) ;

function cbkTrackerStart(hObject, eventdata, handles)
lObj = handles.labelerObj;
algName = lObj.tracker.algorithmName;
%algLabel = lObj.tracker.algorithmNamePretty;
backend = lObj.trackDLBackEnd.prettyName;
handles.txBGTrain.String = sprintf('%s tracking on %s (started %s)',algName,backend,datestr(now,'HH:MM'));
handles.txBGTrain.ForegroundColor = handles.busystatuscolor;
handles.txBGTrain.FontWeight = 'normal';
handles.txBGTrain.Visible = 'on';

function cbkTrackerEnd(hObject, eventdata, handles)
handles.txBGTrain.Visible = 'off';
handles.txBGTrain.String = 'Idle';
handles.txBGTrain.ForegroundColor = handles.idlestatuscolor;

lObj = handles.labelerObj;
val = true;
str = 'New frames tracked';
%lObj.needsSave = true;
%cbkSaveNeeded(lObj,val,str);
lObj.setDoesNeedSave(val, str) ;

function cbkTrackerBackEndChanged(src, evt, handles)
lObj = src ;
updateTrackBackendConfigMenuChecked(handles, lObj) ;

function menu_track_cpr_show_replicates_Callback(hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
vsr = tObj.showVizReplicates;
vsrnew = ~vsr;
sft = tObj.storeFullTracking;
if vsrnew && sft==StoreFullTrackingType.NONE
  warningNoTrace('Tracker will store replicates for final CPR iterations.');
  tObj.storeFullTracking = StoreFullTrackingType.FINALITER;
end
tObj.showVizReplicates = vsrnew;

function cbkLabels2HideChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
handles.menu_view_hide_imported_predictions.Checked = onIff(lObj.labels2Hide);

function cbkLabels2ShowCurrTargetOnlyChanged(src,evt)
lObj = src ;
handles = lObj.gdata;
handles.menu_view_show_imported_preds_curr_target_only.Checked = ...
    onIff(lObj.labels2ShowCurrTargetOnly);

% when trackerInfo is updated, update the tracker info text in the main APT window
function cbkTrackerInfoChanged(src,evt)

tObj = evt.AffectedObject;
tObj.lObj.gdata.text_trackerinfo.String = tObj.getTrackerInfoString();

% when lastLabelChangeTS is updated, update the tracker info text in the main APT window
function cbkLastLabelChangeTS(src, evt)
lObj = src ;
if ~isempty(lObj.trackersAll) && ~isempty(lObj.tracker),
  lObj.gdata.text_trackerinfo.String = lObj.tracker.getTrackerInfoString() ;
end

function cbkParameterChange(src, evt)
lObj = src ;
if isempty(lObj.trackersAll) || isempty(lObj.tracker) ,
  return
end
lObj.gdata.text_trackerinfo.String = lObj.tracker.getTrackerInfoString() ;

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

% AL20180205 LEAVE ME good functionality just currently dormant. CalRigs
% need to be updated, /reconstruct2d()
%
% function menu_view_show_3D_axes_Callback(hObject,eventdata,handles)
% if isfield(handles,'hShow3D')
%   deleteValidHandles(handles.hShow3D);
% end
% handles.hShow3D = gobjects(0,1);
% 
% tfHide = strcmp(hObject.Checked,'on');
% 
% if tfHide
%   hObject.Checked = 'off';
% else
%   lObj = handles.labelerObj;
%   lc = lObj.lblCore;
%   if ~( ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration )
%     handles.labelerObj.lerror('LabelerGUI:multiView',...
%       'Labeling mode must support multiple, calibrated views.');
%   end
%   vcd = lObj.viewCalibrationDataCurrent;
%   if isempty(vcd)
%     handles.labelerObj.lerror('LabelerGUI:vcd','No view calibration data set.');
%   end
%   % Hmm, is this weird, getting the vcd off Labeler not LabelCore. They
%   % should match however
%   assert(isa(vcd,'CalRig'));
%   crig = vcd;
% 
%   nview = lObj.nview;
%   for iview=1:nview
%     ax = handles.axes_all(iview);
% 
%     VIEWDISTFRAC = 5;
% 
%     % Start from where we want the 3D axes to be located in the view
%     xl = ax.XLim;
%     yl = ax.YLim;
%     x0 = diff(xl)/VIEWDISTFRAC+xl(1);
%     y0 = diff(yl)/VIEWDISTFRAC+yl(1);
% 
%     % Project out into 3D; pick a pt
%     [u_p,v_p,w_p] = crig.reconstruct2d(x0,y0,iview);
%     RECON_T = 5; % don't know units here
%     u0 = u_p(1)+RECON_T*u_p(2);
%     v0 = v_p(1)+RECON_T*v_p(2);
%     w0 = w_p(1)+RECON_T*w_p(2);
% 
%     % Loop and find the scale where the the maximum projected length is ~
%     % 1/8th the current view
%     SCALEMIN = 0;
%     SCALEMAX = 20;
%     SCALEN = 300;
%     avViewSz = (diff(xl)+diff(yl))/2;
%     tgtDX = avViewSz/VIEWDISTFRAC*.8;  
%     scales = linspace(SCALEMIN,SCALEMAX,SCALEN);
%     for iScale = 1:SCALEN
%       % origin is (u0,v0,w0) in 3D; (x0,y0) in 2D
% 
%       s = scales(iScale);    
%       [x1,y1] = crig.project3d(u0+s,v0,w0,iview);
%       [x2,y2] = crig.project3d(u0,v0+s,w0,iview);
%       [x3,y3] = crig.project3d(u0,v0,w0+s,iview);
%       d1 = sqrt( (x1-x0).^2 + (y1-y0).^2 );
%       d2 = sqrt( (x2-x0).^2 + (y2-y0).^2 );
%       d3 = sqrt( (x3-x0).^2 + (y3-y0).^2 );
%       if d1>tgtDX || d2>tgtDX || d3>tgtDX
%         fprintf(1,'Found scale for t=%.2f: %.2f\n',RECON_T,s);
%         break;
%       end
%     end
% 
%     LINEWIDTH = 2;
%     FONTSIZE = 12;
%     handles.hShow3D(end+1,1) = plot(ax,[x0 x1],[y0 y1],'r-','LineWidth',LINEWIDTH);
%     handles.hShow3D(end+1,1) = text(x1,y1,'x','Color',[1 0 0],...
%       'fontweight','bold','fontsize',FONTSIZE,'parent',ax);
%     handles.hShow3D(end+1,1) = plot(ax,[x0 x2],[y0 y2],'g-','LineWidth',LINEWIDTH);
%     handles.hShow3D(end+1,1) = text(x2,y2,'y','Color',[0 1 0],...
%       'fontweight','bold','fontsize',FONTSIZE,'parent',ax);
%     handles.hShow3D(end+1,1) = plot(ax,[x0 x3],[y0 y3],'y-','LineWidth',LINEWIDTH);
%     handles.hShow3D(end+1,1) = text(x3,y3,'z','Color',[1 1 0],...
%       'fontweight','bold','fontsize',FONTSIZE,'parent',ax);
%   end
%   hObject.Checked = 'on';
% end
% guidata(hObject,handles);

function menu_track_setparametersfile_Callback(hObject, eventdata, handles)
% Really, "configure parameters"

lObj = handles.labelerObj;
if any(lObj.trackBGTrnIsRunning),
  warndlg('Cannot change training parameters while trackers are training.','Training in progress','modal');
  return;
end
handles.labelerObj.setStatus('Setting training parameters...');


% tObj = lObj.tracker;
% assert(~isempty(tObj));

% KB 20190214 - don't delete trackers anymore!
% tfCanTrack = lObj.trackAllCanTrack();
% if any(tfCanTrack),
%   nTrackers = nnz(tfCanTrack);
%   res = questdlg(sprintf('%d trackers have been trained. Updating parameters will result in one or more of them being deleted, and they will need to be retrained.',nTrackers),...
%     'Update tracking parameters','Continue','Cancel','Continue');
%   if strcmpi(res,'Cancel'),
%     return;
%   end
% end

[tPrm,do_update] = lObj.trackSetAutoParams();

sPrmNew = ParameterSetup(handles.figure,tPrm,'labelerObj',lObj); % modal

if isempty(sPrmNew)
  if do_update
    RC.saveprop('lastCPRAPTParams',sPrmNew);
    %cbkSaveNeeded(lObj,true,'Parameters changed');
    lObj.setDoesNeedSave(true,'Parameters changed') ;
  end
  % user canceled; none
else
  lObj.trackSetParams(sPrmNew);
  RC.saveprop('lastCPRAPTParams',sPrmNew);
  %cbkSaveNeeded(lObj,true,'Parameters changed');
  lObj.setDoesNeedSave(true,'Parameters changed') ;
end

handles.labelerObj.clearStatus();

function menu_track_settrackparams_Callback(hObject, eventdata, handles)

lObj = handles.labelerObj;
handles.labelerObj.setStatus('Setting tracking parameters...');

[tPrm] = lObj.trackGetTrackParams();

sPrmTrack = ParameterSetup(handles.figure,tPrm,'labelerObj',lObj); % modal

if ~isempty(sPrmTrack),
  sPrmNew = lObj.trackSetTrackParams(sPrmTrack);
  RC.saveprop('lastCPRAPTParams',sPrmNew);
  %cbkSaveNeeded(lObj,true,'Parameters changed');
  lObj.setDoesNeedSave(true,'Parameters changed') ;
end

handles.labelerObj.clearStatus();


function menu_track_auto_params_update_Callback(hObject,eventdata,handles)

checked = get(hObject,'Checked');
set(hObject,'Checked',~checked);
handles.labelerObj.trackAutoSetParams = ~checked;

function menu_track_use_all_labels_to_train_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
tObj = lObj.tracker;
if isempty(tObj)
  handles.labelerObj.lerror('LabelerGUI:tracker','No tracker for this project.');
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

% function menu_track_select_training_data_Callback(hObject, eventdata, handles)
% tObj = handles.labelerObj.tracker;
% if tObj.hasTrained
%   resp = questdlg('A tracker has already been trained. Downsampling training data will clear all previous trained/tracked results. Proceed?',...
%     'Clear Existing Tracker','Yes, clear previous tracker','Cancel','Cancel');
%   if isempty(resp)
%     resp = 'Cancel';
%   end
%   switch resp
%     case 'Yes, clear previous tracker'
%       % none
%     case 'Cancel'
%       return;
%   end
% end
% tObj.trnDataSelect();

% function menu_track_set_landmark_matches_Callback(hObject,eventdata,handles)
% handles.labelerObj.setStatus('Defining landmark matches...');
% lObj = handles.labelerObj;
% instructions = ['These part matches are used for data augmentation when training using deep learning. ' ...
%                 'To create more training data, we can flip the original images. This requires knowing ' ...
%                 'which parts on the left of the animal correspond to the same parts on the right side. ' ...
%                 'Use this GUI to select these pairings of parts. Each part can only belong to at most ' ...
%                 'one pair. Some parts will not be part of any pair, e.g. parts that go down the ' ...
%                 'mid-line of the animal should not have a mate.'];
% matches = defineLandmarkMatches(lObj,'edges',lObj.flipLandmarkMatches,'instructions',instructions);
% lObj.setFlipLandmarkMatches(matches);
% handles.labelerObj.clearStatus();

function menu_track_training_data_montage_Callback(hObject,eventdata,handles)
handles.labelerObj.setStatus('Plotting training examples...');
lObj = handles.labelerObj;
lObj.tracker.trainingDataMontage();
handles.labelerObj.clearStatus();

function menu_track_trainincremental_Callback(hObject, eventdata, handles)
handles.labelerObj.trackTrain();

function menu_go_targets_summary_Callback(hObject, eventdata, handles)
if handles.labelerObj.maIsMA
  TrkInfoUI(handles.labelerObj);
else
  handles.labelerObj.targetsTableUI();
end

function menu_go_nav_prefs_Callback(hObject, eventdata, handles)
handles.labelerObj.navPrefsUI();

function menu_go_gt_frames_Callback(hObject, eventdata, handles)
handles.labelerObj.gtShowGTManager();

function menu_evaluate_crossvalidate_Callback(hObject, eventdata, handles)

lObj = handles.labelerObj;

tbl = lObj.labelGetMFTableLabeled;  
if lObj.maIsMA
  tbl = tbl(:,1:2);
  tbl = unique(tbl);
  str = 'frames';
else
  tbl = tbl(:,1:3);
  str = 'targets';
end
n = height(tbl);

inputstr = sprintf('This project has %d labeled %s.\nNumber of folds for k-fold cross validation:',...
  n,str);
resp = inputdlg(inputstr,'Cross Validation',1,{'3'});
if isempty(resp)
  return;
end
nfold = str2double(resp{1});
if round(nfold)~=nfold || nfold<=1
  handles.labelerObj.lerror('LabelerGUI:xvalid','Number of folds must be a positive integer greater than 1.');
end

tbl.split = ceil(nfold*rand(n,1));

t = lObj.tracker;
t.trainsplit(tbl);

function cbkTrackerStoreFullTrackingChanged(hObject, eventdata, handles)
sft = handles.labelerObj.tracker.storeFullTracking;
switch sft
  case StoreFullTrackingType.NONE
    handles.menu_track_cpr_storefull_dont_store.Checked = 'on';
    handles.menu_track_cpr_storefull_store_final_iteration.Checked = 'off';
    handles.menu_track_cpr_storefull_store_all_iterations.Checked = 'off';
    handles.menu_track_cpr_view_diagnostics.Enable = 'off';
  case StoreFullTrackingType.FINALITER
    handles.menu_track_cpr_storefull_dont_store.Checked = 'off';
    handles.menu_track_cpr_storefull_store_final_iteration.Checked = 'on';
    handles.menu_track_cpr_storefull_store_all_iterations.Checked = 'off';
    handles.menu_track_cpr_view_diagnostics.Enable = 'on';
  case StoreFullTrackingType.ALLITERS
    handles.menu_track_cpr_storefull_dont_store.Checked = 'off';
    handles.menu_track_cpr_storefull_store_final_iteration.Checked = 'off';
    handles.menu_track_cpr_storefull_store_all_iterations.Checked = 'on';
    handles.menu_track_cpr_view_diagnostics.Enable = 'on';
  otherwise
    assert(false);
end

function menu_track_clear_tracking_results_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
% legacy behavior not sure why; maybe b/c the user is prob wanting to increase avail mem
%lObj.preProcInitData(); 
res = questdlg('Are you sure you want to clear tracking results?');
if ~strcmpi(res,'yes'),
  return;
end
handles.labelerObj.setStatus('Clearing tracking results...');
tObj = lObj.tracker;
tObj.clearTrackingResults();
handles.labelerObj.clearStatus();
%msgbox('Tracking results cleared.','Done');

function menu_track_clear_tracker_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
res = questdlg('Clear current tracker or all trackers? This will clear your trained tracker(s), along with all tracking results. Hit cancel if you do not want to do this.',...
  'Clear Models','Current only','All','Cancel','Cancel');
if strcmpi(res,'Cancel'),
  return;
elseif strcmpi(res,'Current only'),
  handles.labelerObj.setStatus('Clearing current trained tracker and all tracking results...');
  lObj.clearCurrentTracker();
  handles.labelerObj.clearStatus();
elseif strcmpi(res,'All'),
  handles.labelerObj.setStatus('Clearing trained trackers and all tracking results...');
  lObj.clearAllTrackers();
  handles.labelerObj.clearStatus();
end


function menu_track_cpr_storefull_dont_store_Callback(hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
svr = tObj.showVizReplicates;
if svr
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
tObj.storeFullTracking = StoreFullTrackingType.NONE;

function menu_track_cpr_storefull_store_final_iteration_Callback(...
  hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
tObj.storeFullTracking = StoreFullTrackingType.FINALITER;

function menu_track_cpr_storefull_store_all_iterations_Callback(...
  hObject, eventdata, handles)
tObj = handles.labelerObj.tracker;
tObj.storeFullTracking = StoreFullTrackingType.ALLITERS;

function menu_track_cpr_view_diagnostics_Callback(...
  hObject, eventdata, handles)
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
handles = addDepHandle(handles,hVizGUI);
guidata(handles.figure,handles);

function menu_track_track_and_export_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
tm = getTrackMode(handles);
[tfok,rawtrkname] = lObj.getExportTrkRawnameUI();
if ~tfok
  return;
end
handles.labelerObj.setStatus('Tracking...');
handles.labelerObj.trackAndExport(tm,'rawtrkname',rawtrkname);
handles.labelerObj.clearStatus();

function menu_track_batch_track_Callback(hObject,eventdata,handles)

lObj = handles.labelerObj;
tbobj = TrackBatchGUI(lObj);
tbobj.run();

% 
% persistent jsonfile;
% if isempty(jsonfile),
%   jsonfile = '';
% end
% 
% [filename,pathname] = uigetfile('*.json','Select json batch tracking file',jsonfile);
% if ~ischar(filename),
%   return;
% end
% jsonfile1 = fullfile(pathname,filename);
% if ~exist(jsonfile1,'file'),
%   warndlg(sprintf('File %s does not exist',jsonfile1));
%   return;
% end
% jsonfile = jsonfile1;
% handles.labelerObj.setStatus('Tracking a batch of videos...');
% trackBatch('lObj',handles.labelerObj,'jsonfile',jsonfile);
% handles.labelerObj.clearStatus();

function menu_track_all_movies_Callback(hObject,eventdata,handles)

lObj = handles.labelerObj;
mIdx = lObj.allMovIdx();
toTrackIn = lObj.mIdx2TrackList(mIdx);
tbobj = TrackBatchGUI(lObj,'toTrack',toTrackIn);
[toTrackOut] = tbobj.run(); %#ok<NASGU>
% todo: import predictions

function menu_track_current_movie_Callback(hObject,eventdata,handles)

lObj = handles.labelerObj;
mIdx = lObj.currMovIdx;
toTrackIn = lObj.mIdx2TrackList(mIdx);
mdobj = SpecifyMovieToTrackGUI(lObj,lObj.gdata.figure,toTrackIn);
[toTrackOut,dostore] = mdobj.run();
if ~dostore,
  return;
end
trackBatch('lObj',lObj,'toTrack',toTrackOut);

function menu_track_id_Callback(hObject,eventdata,handles)

lObj = handles.labelerObj;
lObj.track_id = ~lObj.track_id;
set(handles.menu_track_id,'checked',lObj.track_id);

% function menu_track_export_current_movie_Callback(hObject,eventdata,handles)
% lObj = handles.labelerObj;
% iMov = lObj.currMovie;
% if iMov==0
%   handles.labelerObj.lerror('LabelerGUI:noMov','No movie currently set.');
% end
% [tfok,rawtrkname] = lObj.getExportTrkRawnameUI();
% if ~tfok
%   return;
% end
% lObj.trackExportResults(iMov,'rawtrkname',rawtrkname);

function menu_file_clear_imported_Callback(hObject,evtdata,handles)
lObj = handles.labelerObj;
lObj.labels2Clear();

function menu_file_export_all_movies_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
nMov = lObj.nmoviesGTaware;
if nMov==0
  handles.labelerObj.lerror('LabelerGUI:noMov','No movies in project.');
end
iMov = 1:nMov;
[tfok,rawtrkname] = lObj.getExportTrkRawnameUI();
if ~tfok
  return;
end
lObj.trackExportResults(iMov,'rawtrkname',rawtrkname);

function menu_track_set_labels_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
tObj = lObj.tracker;
if lObj.gtIsGTMode
  handles.labelerObj.lerror('LabelerGUI:gt','Unsupported in GT mode.');
end

if ~isempty(tObj) && tObj.getHasTrained() && (~lObj.maIsMA)
  % single animal. Use prediction if available else use imported below
  [tfhaspred,xy,tfocc] = tObj.getTrackingResultsCurrFrm(); %#ok<ASGLU>
  itgt = lObj.currTarget;
  if ~tfhaspred(itgt)
    msgbox('No predictions for current frame.');
    return;    
  end
  xy = xy(:,:,itgt); % "targets" treatment differs from below
  disp(xy);
  
  % AL20161219: possibly dangerous, assignLabelCoords prob was intended
  % only as a util method for subclasses rather than public API. This may
  % not do the right thing for some concrete LabelCores.
  lObj.lblCore.assignLabelCoords(xy);
else
  iMov = lObj.currMovie;
  frm = lObj.currFrame;
  if iMov==0
    handles.labelerObj.lerror('LabelerGUI:setLabels','No movie open.');
  end
  
  if lObj.maIsMA
    % We need to be smart about which to use. 
    % If only one of imported or prediction exist for the current frame then use whichever exists
    % If both exist for current frame, then don't do anything and error.
    
    useImported = true;
    usePred = true;
    % getting imported info old sytle. Doesn't work anymore
    
%     s = lObj.labels2{iMov};
%     itgtsImported = Labels.isLabeledF(s,frm);
%     ntgtsImported = numel(itgtsImported);
 
    % check if we can use imported
    imp_trk = lObj.labeledpos2trkViz;
    if isempty(imp_trk)
      useImported=false;
    elseif isnan(imp_trk.currTrklet)
      useImported=false;
    else
      s = lObj.labels2{iMov};
      iTgtImp = imp_trk.currTrklet;
      if isnan(iTgtImp)
        useImported = false;
      else
        [tfhaspred,~,~] = s.getPTrkFrame(frm,'collapse',true);
        if ~tfhaspred(iTgtImp)
          useImported = false;
        end
      end
    end
      
    % check if we can use pred
    if isempty(tObj)
      usePred = false;
    elseif isempty(tObj.trkVizer)
      usePred = false;
    else
      [tfhaspred,xy,tfocc] = tObj.getTrackingResultsCurrFrm(); %#ok<ASGLU>
      iTgtPred = tObj.trkVizer.currTrklet;
      if isnan(iTgtPred)
        usePred = false;
      elseif ~tfhaspred(iTgtPred)
        usePred = false;      
      end
    end
    
    if usePred && useImported
      msgbox('Both imported and prediction exist for current frame. Cannot decide which to use. Skipping');
      return
    end
    
    if (~usePred) && (~useImported)
      msgbox('No predictions for current frame or no valid tracklet selected. Nothing to use as a label');
      return
    end

    if useImported
      s = lObj.labels2{iMov};
      iTgt = imp_trk.currTrklet;
      [~,xy,tfocc] = s.getPTrkFrame(frm,'collapse',true);      
    else
      iTgt = tObj.trkVizer.currTrklet;
      [~,xy,tfocc] = tObj.getTrackingResultsCurrFrm();
    end
    xy = xy(:,:,iTgt); % "targets" treatment differs from below
    occ = tfocc(:,iTgt);
    ntgts = lObj.labelNumLabeledTgts();
    lObj.setTargetMA(ntgts+1);
    lObj.labelPosSet(xy,occ);
    lObj.updateTrxTable();
    iTgt = lObj.currTarget;
    lObj.lblCore.tv.updateTrackResI(xy,occ,iTgt);

  else
    if lObj.nTrx>1
      handles.labelerObj.lerror('LabelerGUI:setLabels','Unsupported for multiple targets.');
    end
    %lpos2 = lObj.labeledpos2{iMov};
    p = Labels.getLabelsF(lObj.labels2{iMov},frm,1);
    lpos2xy = reshape(p,lObj.nLabelPoints,2);
    %assert(size(lpos2,4)==1); % "targets" treatment differs from above
    %lpos2xy = lpos2(:,:,frm);
    lObj.labelPosSet(lpos2xy);
    
    lObj.lblCore.newFrame(frm,frm,1);
  end
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

function menu_evaluate_gtmode_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;

handles.labelerObj.setStatus('Switching between Labeling and Ground Truth Mode...');

gt = lObj.gtIsGTMode;
gtNew = ~gt;
lObj.gtSetGTMode(gtNew);
% hGTMgr = lObj.gdata.GTMgr;
if gtNew
  hMovMgr = lObj.gdata.movieMgr;
  hMovMgr.setVisible(true);
  figure(hMovMgr.hFig);
end
handles.labelerObj.clearStatus();

function menu_evaluate_gtloadsuggestions_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
LabelerGT.loadSuggestionsUI(lObj);

function menu_evaluate_gtsetsuggestions_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
LabelerGT.setSuggestionsToLabeledUI(lObj);

function menu_evaluate_gtcomputeperf_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
assert(lObj.gtIsGTMode);
% next three lines identical to GTManager:pbComputeGT_Callback
lObj.gtComputeGTPerformance();

function menu_evaluate_gtcomputeperfimported_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;
assert(lObj.gtIsGTMode);
% next three lines identical to GTManager:pbComputeGT_Callback
lObj.gtComputeGTPerformance('useLabels2',true);

function menu_evaluate_gtexportresults_Callback(hObject,eventdata,handles)
lObj = handles.labelerObj;

tblRes = lObj.gtTblRes;
if isempty(tblRes)
  errordlg('No GT results are currently available.','Export GT Results');
  return;
end

%assert(lObj.gtIsGTMode);
fname = lObj.getDefaultFilenameExportGTResults();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return;
end
fname = fullfile(p,f);  
VARNAME = 'tblGT';
s = struct();
s.(VARNAME) = tblRes;
save(fname,'-mat','-struct','s');
fprintf('Saved table ''%s'' to file ''%s''.\n',VARNAME,fname);
  
function cbkGtIsGTModeChanged(src,evt)
lObj = src;
handles = lObj.gdata;
gt = lObj.gtIsGTMode;
onIffGT = onIff(gt);
handles.menu_go_gt_frames.Visible = onIffGT;
handles.menu_evaluate_gtmode.Checked = onIffGT;
handles.menu_evaluate_gtloadsuggestions.Visible = onIffGT;
handles.menu_evaluate_gtsetsuggestions.Visible = onIffGT;
handles.menu_evaluate_gtcomputeperf.Visible = onIffGT;
handles.menu_evaluate_gtcomputeperfimported.Visible = onIffGT;
handles.menu_evaluate_gtexportresults.Visible = onIffGT;
handles.txGTMode.Visible = onIffGT;
handles.GTMgr.Visible = onIffGT;
hlpGTUpdateAxHilite(lObj);

function figure_CloseRequestFcn(hObject, eventdata, handles)
%CloseGUI(handles);
controller = handles.controller ;
controller.quitRequested() ;

% function CloseGUI(handles)
% if hlpSave(handles.labelerObj)
%   handles = clearDepHandles(handles);
%   if isfield(handles,'movieMgr') && ~isempty(handles.movieMgr) ...
%       && isvalid(handles.movieMgr)
%     delete(handles.movieMgr);
%   end  
%   delete(findall(0,'tag','TrkInfoUI'));
%   delete(handles.figure);
%   delete(handles.labelerObj);
% end

function pumInfo_Callback(hObject, eventdata, handles)
cprop = get(hObject,'Value');
handles.labelTLInfo.setCurProp(cprop);
cpropNew = handles.labelTLInfo.getCurProp();
if cpropNew ~= cprop,
  set(hObject,'Value',cpropNew);
end
hlpRemoveFocus(hObject,handles);

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
if ~checkProjAndMovieExist(handles)
  return;
end
play(hObject,handles,'playsegment','videoPlaySegFwdEnding');
function pbPlaySegRev_Callback(hObject, eventdata, handles)
if ~checkProjAndMovieExist(handles)
  return;
end
play(hObject,handles,'playsegmentrev','videoPlaySegRevEnding');

function pbPlay_Callback(hObject, eventdata, handles)
if ~checkProjAndMovieExist(handles)
  return;
end
play(hObject,handles,'play','videoPlay');

function tfok = checkProjAndMovieExist(handles)
tfok = false;
lObj = handles.labelerObj;
if ~lObj.hasProject
  return;
end
if ~lObj.hasMovie
  msgbox('There is no movie open.');
  return;
end
tfok = true;

%% Cropping
function handles = cropInitImRects(handles)
deleteValidHandles(handles.cropHRect);
handles.cropHRect = ...
  arrayfun(@(x)imrect(x,[nan nan nan nan]),handles.axes_all,'uni',0); %#ok<IMRECT> 
handles.cropHRect = cat(1,handles.cropHRect{:}); % ML 2016a ish can't concat imrects in arrayfun output
arrayfun(@(x)set(x,'Visible','off','PickableParts','none','UserData',true),...
  handles.cropHRect); % userdata: see cropImRectSetPosnNoPosnCallback
for ivw=1:numel(handles.axes_all)
  posnCallback = @(zpos)cbkCropPosn(zpos,ivw,handles.figure);
  handles.cropHRect(ivw).addNewPositionCallback(posnCallback);
end

function cbkCropIsCropModeChanged(src,evt)
lObj = src;
lObj.setStatus('Switching crop mode...');
cropReactNewCropMode(lObj.gdata,lObj.cropIsCropMode);
if lObj.hasMovie
  lObj.setFrame(lObj.currFrame,'tfforcereadmovie',true);
end
lObj.clearStatus();

function cbkUpdateCropGUITools(src,evt)
lObj = src;
cropReactNewCropMode(lObj.gdata,lObj.cropIsCropMode);

function cbkCropCropsChanged(src,evt)
lObj = src;
cropUpdateCropHRects(lObj.gdata);

function cropReactNewCropMode(handles,tf)

% CROPCONTROLS = {
%   'pushbutton_exitcropmode'
%   'tbAdjustCropSize'
%   'pbClearAllCrops'
%   'txCropMode'
%   };
REGCONTROLS = {
  'pbClear'
  'tbAccept'
  'pbTrain'
  'pbTrack'
  'pumTrack'};

onIfTrue = onIff(tf);
offIfTrue = onIff(~tf);

%cellfun(@(x)set(handles.(x),'Visible',onIfTrue),CROPCONTROLS);
set(handles.uipanel_cropcontrols,'Visible',onIfTrue);
set(handles.text_trackerinfo,'Visible',offIfTrue);

cellfun(@(x)set(handles.(x),'Visible',offIfTrue),REGCONTROLS);
handles.menu_file_crop_mode.Checked = onIfTrue;

cropUpdateCropHRects(handles);
cropUpdateCropAdjustingCropSize(handles,false);

function cropUpdateCropHRects(handles)
% Update handles.cropHRect from lObj.cropIsCropMode, lObj.currMovie and
% lObj.movieFilesAll*cropInfo
%
% rect props set:
% - position
% - visibility, pickableparts
%
% rect props NOT set:
% - resizeability. 

lObj = handles.labelerObj;
tfCropMode = lObj.cropIsCropMode;
[tfHasCrop,roi] = lObj.cropGetCropCurrMovie();
if tfCropMode && tfHasCrop
  nview = lObj.nview;
  imnc = lObj.movierawnc;
  imnr = lObj.movierawnr;
  szassert(roi,[nview 4]);
  szassert(imnc,[nview 1]);
  szassert(imnr,[nview 1]);
  for ivw=1:nview
    h = handles.cropHRect(ivw);
    cropImRectSetPosnNoPosnCallback(h,CropInfo.roi2RectPos(roi(ivw,:)));
    set(h,'Visible','on','PickableParts','all');
    fcn = makeConstrainToRectFcn('imrect',[1 imnc(ivw)],[1 imnr(ivw)]);
    h.setPositionConstraintFcn(fcn);
  end
else
  arrayfun(@(x)cropImRectSetPosnNoPosnCallback(x,[nan nan nan nan]),...
    handles.cropHRect);
  arrayfun(@(x)set(x,'Visible','off','PickableParts','none'),handles.cropHRect);
end

function cropImRectSetPosnNoPosnCallback(hRect,pos)
% Set the hRect's graphics position without triggering its
% PositionCallback. Works in concert with cbkCropPosn
tfSetPosnLabeler0 = get(hRect,'UserData');
set(hRect,'UserData',false);
hRect.setPosition(pos);
set(hRect,'UserData',tfSetPosnLabeler0);

function cropUpdateCropAdjustingCropSize(handles,tfAdjust)
% cropUpdateCropAdjustingCropSize(handles) --
%   update .cropHRects.resizeable based on tbAdjustCropSize
% cropUpdateCropAdjustingCropSize(handles,tfCropMode) --
%   update .cropHRects.resizeable and tbAdjustCropSize based on tfAdjust

tb = handles.tbAdjustCropSize;
if nargin<2
  tfAdjust = tb.Value==tb.Max; % tb depressed
end

if tfAdjust
  tb.Value = tb.Max;
  tb.String = handles.tbAdjustCropSizeString1;
  tb.BackgroundColor = handles.tbAdjustCropSizeBGColor1;
else
  tb.Value = tb.Min;
  tb.String = handles.tbAdjustCropSizeString0;
  tb.BackgroundColor = handles.tbAdjustCropSizeBGColor0;
end
arrayfun(@(x)x.setResizable(tfAdjust),handles.cropHRect);

function cbkCropPosn(posn,iview,hFig)
handles = guidata(hFig);
tfSetPosnLabeler = get(handles.cropHRect(iview),'UserData');
if tfSetPosnLabeler
  [roi,roiw,roih] = CropInfo.rectPos2roi(posn);
  tb = handles.tbAdjustCropSize;
  if tb.Value==tb.Max % tbAdjustCropSizes depressed; using as proxy for, imrect is resizable
    fprintf(1,'roi (width,height): (%d,%d)\n',roiw,roih);
  end
  handles.labelerObj.cropSetNewRoiCurrMov(iview,roi);
end

function tbAdjustCropSize_Callback(hObject, eventdata, handles)
cropUpdateCropAdjustingCropSize(handles);
tb = handles.tbAdjustCropSize;
if tb.Value==tb.Min
  % user clicked "Done Adjusting"
  warningNoTrace('All movies in a given view must share the same crop size. The sizes of all crops have been updated as necessary.'); 
elseif tb.Value==tb.Max
  % user clicked "Adjust Crop Size"
  lObj = handles.labelerObj;
  if ~lObj.cropProjHasCrops
    lObj.cropInitCropsAllMovies;
    fprintf(1,'Default crop initialized for all movies.\n');
    cropUpdateCropHRects(handles);
  end
end
function pbClearAllCrops_Callback(hObject, eventdata, handles)
handles.labelerObj.cropClearAllCrops();

% % -------------------------------------------------------------------------
% function SetStatus(handles, str, isbusy)
% 
% if nargin<3 ,
%   isbusy = true;
% end
% istemp = false;
% if isempty(isbusy)
%   color = get(handles.txStatus,'ForegroundColor');
% elseif isbusy
%   color = handles.busystatuscolor;
%   if isfield(handles,'figs_all') && any(ishandle(handles.figs_all)),
%     set(handles.figs_all(ishandle(handles.figs_all)),'Pointer','watch');
%   else
%     set(handles.figure,'Pointer','watch');
%   end
% else
%   color = handles.idlestatuscolor;
%   if isfield(handles,'figs_all') && any(ishandle(handles.figs_all)),
%     set(handles.figs_all(ishandle(handles.figs_all)),'Pointer','arrow');
%   else
%     set(handles.figure,'Pointer','arrow');
%   end
% end
% set(handles.txStatus,'ForegroundColor',color);
% SetStatusText(handles,str);
% drawnow('limitrate');
% if ~isbusy && ~istemp,  syncStatusBarTextWhenClear(handles);
% end
% 
% function RefreshStatus(handles)
% 
% s = getappdata(handles.txStatus,'InputString');
% if ischar(s),
%   SetStatusText(handles,s);
% end
% 
% function SetStatusText(handles,s)
% 
% setappdata(handles.txStatus,'InputString',s);
% isprojname = contains(s,'$PROJECTNAME');
% if isprojname && isfield(handles,'labelerObj') && handles.labelerObj.hasProject,
%   if ~ischar(handles.labelerObj.projectfile),
%     projfile = '';
%   else
%     projfile = handles.labelerObj.projectfile;
%     if numel(projfile) > 100,
%       projfile = ['..',projfile(end-97:end)];
%     end
%   end
%   s1 = strrep(s,'$PROJECTNAME',projfile);
%   [~,n,ext] = fileparts(projfile);
%   n = [n,ext];
%   s2 = strrep(s,'$PROJECTNAME',n);
%   if ~isempty(handles.jtxStatus),
%     set(handles.txStatus,'String',s1);
%     drawnow;
%     pos1 = get(handles.jtxStatus,'PreferredSize');
%     w = get(handles.jtxStatus,'Width');
%     %fprintf('width = %f, preferredwidth = %f\n',w,pos1.width);
%     if pos1.width > w*.95,
%       set(handles.txStatus,'String',s2);
%     end
%   else
%     set(handles.txStatus,'String',s2);
%   end
% else
%   set(handles.txStatus,'String',s);
% end
% 
% % -------------------------------------------------------------------------
% function ClearStatus(handles)
% 
% cleartext = getStatusBarTextWhenClear(handles);
% set(handles.txStatus, ...
%     'ForegroundColor',handles.idlestatuscolor);
% SetStatusText(handles,cleartext);
% if isfield(handles,'figs_all') && any(ishandle(handles.figs_all))
%   set(handles.figs_all(ishandle(handles.figs_all)),'Pointer','arrow');
% else
%   set(handles.figure,'Pointer','arrow');
% end
% drawnow('limitrate');
% 
% function syncStatusBarTextWhenClear(handles,s)
% 
% try
%   s = getappdata(handles.txStatus,'InputString');
%   if isempty(s),
%     s = get(handles.txStatus,'string');
%   end
%   setStatusBarTextWhenClear(handles,s);
% catch
% end
% 
% function setStatusBarTextWhenClear(handles,s)
% 
% try
%   setappdata(handles.txStatus,'text_when_clear',s);
% catch
% end
% 
% function s = getStatusBarTextWhenClear(handles)
% 
% try
%   s = getappdata(handles.txStatus,'text_when_clear');
% catch
%   warning('Could not get text_when_clear appdata for status bar');
%   s = '';
% end
% 
% function refreshStatus(handles)
% s = getappdata(handles.txStatus,'InputString');
% if isempty(s),
%   s = get(handles.txStatus,'string');
% end
% SetStatusText(handles,s);

% --------------------------------------------------------------------
function menu_file_export_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_export_labels2_trk_curr_mov (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

lObj = handles.labelerObj;
iMov = lObj.currMovie;
if iMov==0
  handles.labelerObj.lerror('LabelerGUI:noMov','No movie currently set.');
end
[tfok,rawtrkname] = lObj.getExportTrkRawnameUI();
if ~tfok
  return;
end
lObj.trackExportResults(iMov,'rawtrkname',rawtrkname);


% --------------------------------------------------------------------
function menu_file_import_export_advanced_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_import_export_advanced (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_track_tracking_algorithm_Callback(hObject, eventdata, handles)
% hObject    handle to menu_track_tracking_algorithm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function menu_view_landmark_colors_Callback(hObject, eventdata)
handles = guidata(hObject);
lObj = handles.labelerObj;
cbkApply = @(varargin)hlpApplyCosmetics(lObj,varargin{:});
LandmarkColors(lObj,cbkApply);
% AL 20220217: changes now applied immediately
% if ischange
%   cbkApply(savedres.colorSpecs,savedres.markerSpecs,savedres.skelSpecs);
% end

function hlpApplyCosmetics(lObj,colorSpecs,mrkrSpecs,skelSpecs)
lObj.updateLandmarkColors(colorSpecs);
lObj.updateLandmarkCosmetics(mrkrSpecs);
lObj.updateSkeletonCosmetics(skelSpecs);

function menu_track_edit_skeleton_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
landmark_specs('lObj',lObj);
%hasSkeleton = ~isempty(lObj.skeletonEdges) ;
%lObj.setShowSkeleton(hasSkeleton) ;

function menu_track_viz_dataaug_Callback(hObject,evtdata,handles)
lObj = handles.labelerObj;
t = lObj.tracker;
t.retrain('augOnly',true);

function menu_view_showhide_skeleton_Callback(hObject, eventdata, handles)
if strcmpi(get(hObject,'Checked'),'off'),
  hObject.Checked = 'on';
  handles.labelerObj.setShowSkeleton(true);
else
  hObject.Checked = 'off';
  handles.labelerObj.setShowSkeleton(false);
end

function menu_view_showhide_maroi_Callback(hObject, eventdata, handles)
if strcmpi(get(hObject,'Checked'),'off'),
  handles.labelerObj.setShowMaRoi(true);
else
  handles.labelerObj.setShowMaRoi(false);
end

function menu_view_showhide_maroiaux_Callback(hObject, eventdata, handles)
tf = strcmpi(get(hObject,'Checked'),'off');
handles.labelerObj.setShowMaRoiAux(tf);

% --- Executes on selection change in popupmenu_prevmode.
function popupmenu_prevmode_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_prevmode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
mode = contents{get(hObject,'Value')};
if strcmpi(mode,'Reference'),
  handles.labelerObj.setPrevAxesMode(PrevAxesMode.FROZEN,handles.labelerObj.prevAxesModeInfo);
else
  handles.labelerObj.setPrevAxesMode(PrevAxesMode.LASTSEEN);
end

% --- Executes during object creation, after setting all properties.
function popupmenu_prevmode_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_prevmode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_freezetemplate.
function pushbutton_freezetemplate_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_freezetemplate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.labelerObj.setPrevAxesMode(PrevAxesMode.FROZEN);

function hfig = splashScreen(handles)

%hparent = handles.figure;
hfig = nan;
p = APT.Root; %fileparts(mfilename('fullpath'));
splashimfilename = fullfile(p,'gfx','SplashScreen.png');
if ~exist(splashimfilename,'file'),
  return;
end

% oldunits = get(hparent,'Units');
% set(hparent,'Units','pixels');
% pos0 = get(hparent,'Position');
% set(hparent,'Units',oldunits);

warnst = warning('off','MATLAB:imagesci:png:libraryWarning');
im = imread(splashimfilename);
warning(warnst);
sz = size(im);
sz = sz(1:2);

s = {'APT: The Animal Part Tracker'
  'http://kristinbranson.github.io/APT/'
  ''
  'Developed and tested by Allen Lee, Mayank Kabra,'
  'Alice Robie, Felipe Rodriguez, Stephen Huston,'
  'Roian Egnor, Austin Edwards, Caroline Maloney,'
  'and Kristin Branson'};

border = 20;
w0 = 400;
texth1 = 25;
w = w0+2*border;
texth2 = (numel(s)-1)*texth1;
textskip = 5;
h0 = w0*sz(1)/sz(2);
h = 2*border+h0+border+texth2+textskip+texth1;

r = [w,h]/2;

center = get(0,'ScreenSize');
center = center(3:4)/2;
%center = pos0([1,2])+pos0([3,4])/2;
pos1 = [center-r,2*r];

hfig = figure('Name','Starting APT...','Color','k','Units','pixels','Position',pos1,'ToolBar','none','NumberTitle','off','MenuBar','none','Pointer','watch');%'Visible','off',
hax = axes('Parent',hfig,'Units','pixels','Position',[border,border,w0,h0]);
him = image(im,'Parent',hax,'Tag','image_SplashScreen'); axis(hax,'image','off');  %#ok<NASGU> 
htext = uicontrol('Style','text','String',s{1},'Units','pixels','Position',[border,h-border-texth1,w0,texth1],...
  'BackgroundColor','k','HorizontalAlignment','center',...
  'Parent',hfig,'ForegroundColor','c','FontUnits','pixels','FontSize',texth1*.9,'FontWeight','b',...
  'Tag','text1_SplashScreen'); %#ok<NASGU> 
htext = uicontrol('Style','text','String',s(2:end),'Units','pixels','Position',[border,border+h0+border,w0,texth2],...
  'BackgroundColor','k','HorizontalAlignment','center',...
  'Parent',hfig,'ForegroundColor','c','FontUnits','pixels','FontSize',14,...
  'Tag','text2_SplashScreen'); %#ok<NASGU> 
set(hfig,'Visible','on');
drawnow;

function RefocusSplashScreen(hfigsplash,handles)

if ~ishandle(hfigsplash),
  return;
end

hparent = handles.figure;
oldunits = get(hparent,'Units');
set(hparent,'Units','pixels');
pos0 = get(hparent,'OuterPosition');
set(hparent,'Units',oldunits);

%topleft = [pos0(1),pos0(2)+pos0(4)+30];
center = pos0([1,2])+pos0([3,4])/2;
pos1 = get(hfigsplash,'Position');
%pos2 = [topleft(1),topleft(2)-pos1(4),pos1(3),pos1(4)];
pos2 = [center-pos1(3:4)/2,pos1(3:4)];
set(hfigsplash,'Position',pos2);
figure(hfigsplash);
drawnow;
% --- Executes on button press in pushbutton_exitcropmode.
function pushbutton_exitcropmode_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_exitcropmode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

lObj = handles.labelerObj;
lObj.cropSetCropMode(false);


% --------------------------------------------------------------------
function menu_view_occluded_points_box_Callback(hObject, eventdata, handles)
% hObject    handle to menu_view_occluded_points_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

lObj = handles.labelerObj;
lObj.setShowOccludedBox(~lObj.showOccludedBox);
if lObj.showOccludedBox,
  lObj.lblCore.showOcc();
else
  lObj.lblCore.hideOcc();
end

% --- Executes on selection change in pumInfo_labels.
function pumInfo_labels_Callback(hObject, eventdata, handles)
% hObject    handle to pumInfo_labels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pumInfo_labels contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pumInfo_labels

ipropType = get(hObject,'Value');
% see also InfoTimeline/enforcePropConsistencyWithUI
iprop = get(handles.pumInfo,'Value');
props = handles.labelTLInfo.getPropsDisp(ipropType);
if iprop > numel(props),
  iprop = 1;
end
set(handles.pumInfo,'String',props,'Value',iprop);
handles.labelTLInfo.setCurPropType(ipropType,iprop);


% --- Executes during object creation, after setting all properties.
function pumInfo_labels_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pumInfo_labels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function menu_file_shortcuts_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_shortcuts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

lObj = handles.labelerObj;
while true,
  [~,newShortcuts] = propertiesGUI([],lObj.projPrefs.Shortcuts);
  shs = struct2cell(newShortcuts);
  % everything should just be one character
  % no repeats
  uniqueshs = unique(shs);
  isproblem = any(cellfun(@numel,shs) ~= 1) || numel(uniqueshs) < numel(shs);
  if ~isproblem,
    break;
  end
  res = questdlg('All shortcuts must be unique, single-character letters','Error setting shortcuts','Try again','Cancel','Try again');
  if strcmpi(res,'Cancel'),
    return;
  end  
end
%oldShortcuts = lObj.projPrefs.Shortcuts;
lObj.projPrefs.Shortcuts = newShortcuts;
handles = setShortcuts(handles);
guidata(hObject,handles);
