function dispatchMainFigureCallback(callbackName, varargin)
feval(callbackName, varargin{:}) ;



function initialize(hObject,eventdata,handles,varargin) 

% if verLessThan('matlab','8.4')
%   labeler.lerror('LabelerGUI:ver','LabelerGUI requires MATLAB version R2014b or later.');
% end

% if labeler.isgui ,
%   hfigsplash = splashScreen(handles);
% end

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

% Do a poor-man's labeler.setStatus()
if isfield(handles,'figs_all') && any(isgraphics(handles.figs_all)) ,
  set(handles.figs_all(isgraphics(handles.figs_all)),'Pointer','watch') ;
end
set(handles.txStatus,'ForegroundColor',handles.busystatuscolor) ;
set(handles.txStatus,'String','Initializing APT...') ;

PURPLE = [80 31 124]/256;
handles.tbTLSelectMode.BackgroundColor = PURPLE;

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
handles.menu_setup_label_outliers = uimenu('Parent',handles.menu_labeling_setup,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_setup_label_outliers_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Find Suspicious Labels...',...
  'Tag','menu_setup_label_outliers',...
  'Visible','on');
%moveMenuItemAfter(handles.menu_setup_label_outliers,handles.menu_setup_label_overlay_montage);

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
   'menu_setup_label_outliers'
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

deleteValidGraphicsHandles(handles.menu_view_landmark_colors.Children);
set(handles.menu_view_landmark_colors,'Callback',@menu_view_landmark_colors_Callback);

handles.menu_view_showhide_skeleton = uimenu('Parent',handles.menu_view,...
  'Label','Show skeleton',...
  'Tag','menu_view_showhide_skeleton',...
  'Checked','off',...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_showhide_skeleton_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_view_showhide_skeleton,handles.menu_view_landmark_colors);

handles.menu_view_hide_trajectories = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_hide_trajectories_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Hide trajectories',...
  'Tag','menu_view_hide_trajectories',...
  'Checked','off');
handles.menu_view_plot_trajectories_current_target_only = uimenu('Parent',handles.menu_view,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_view_plot_trajectories_current_target_only_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Plot trajectories only for current target',...
  'Tag','menu_view_plot_trajectories_current_target_only',...
  'Checked','off');
moveMenuItemAfter(handles.menu_view_plot_trajectories_current_target_only,...
                  handles.menu_view_hide_trajectories);

delete(handles.menu_view_trajectories) ;

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

% Track menu
set(handles.menu_track_setparametersfile,...
    'Label','Set training parameters...',...
    'Callback',@(hObject,eventdata)LabelerGUI('menu_track_setparametersfile_Callback',hObject,eventdata,guidata(hObject)),...
    'Separator','on'); % separator b/c trackers are listed above

handles.menu_track_edit_skeleton = ...
  uimenu('Parent',handles.menu_track,...
         'Label','Landmark parameters...',...
         'Tag','menu_track_edit_skeleton',...
         'Callback',@(hObject,eventdata)LabelerGUI('menu_track_edit_skeleton_Callback',hObject,eventdata,guidata(hObject)));
moveMenuItemAfter(handles.menu_track_edit_skeleton, handles.menu_track_setparametersfile) ;

handles.menu_track_settrackparams = ...
  uimenu('Parent',handles.menu_track,...
         'Label','Set tracking parameters...',...
         'Tag','menu_track_settrackparams',...
         'Callback',@(hObject,eventdata)LabelerGUI('menu_track_settrackparams_Callback',hObject,eventdata,guidata(hObject))) ;
moveMenuItemAfter(handles.menu_track_settrackparams, handles.menu_track_edit_skeleton) ;

handles.menu_track_viz_dataaug = ...
  uimenu('Parent',handles.menu_track,...
         'Label','Visualize sample training images...',...
         'Tag','menu_track_viz_dataaug',...
         'Callback',@(hObject,eventdata)LabelerGUI('menu_track_viz_dataaug_Callback',hObject,eventdata,guidata(hObject))) ;
moveMenuItemAfter(handles.menu_track_viz_dataaug, handles.menu_track_settrackparams) ;

handles.menu_track_training_data_montage = ...
  uimenu('Parent',handles.menu_track,...
         'Label','Training Data Montage',...
         'Tag','menu_track_training_data_montage',...
         'Callback',@(h,evtdata)LabelerGUI('menu_track_training_data_montage_Callback',h,evtdata,guidata(h)));
moveMenuItemAfter(handles.menu_track_training_data_montage, handles.menu_track_viz_dataaug) ;
delete(handles.menu_track_select_training_data) ;

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


moveMenuItemAfter(handles.menu_track_track_and_export,handles.menu_track_retrain);

handles.menu_track_trainincremental = handles.menu_track_retrain;
handles = rmfield(handles,'menu_track_retrain');
handles.menu_track_trainincremental.Callback = @(h,edata)LabelerGUI('menu_track_trainincremental_Callback',h,edata,guidata(h));
handles.menu_track_trainincremental.Label = 'Incremental Train';
handles.menu_track_trainincremental.Tag = 'menu_track_trainincremental';
handles.menu_track_trainincremental.Visible = 'off';

handles.menu_track_clear_tracking_results = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_clear_tracking_results_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Clear tracking results',...
  'Tag','menu_track_clear_tracking_results',...
  'Separator','on');  
moveMenuItemAfter(handles.menu_track_clear_tracking_results,handles.menu_track_all_movies);

handles.menu_track_reset_current_tracker = ...
  uimenu('Parent',handles.menu_track,...
         'Callback',@(s,e)(handles.controller.controlActuated('menu_track_reset_current_tracker', s, e)), ...
         'Label','Reset current tracker',...
         'Tag','menu_track_reset_current_tracker',...
         'Separator','off');
moveMenuItemAfter(handles.menu_track_reset_current_tracker, handles.menu_track_clear_tracking_results) ;

handles.menu_track_delete_current_tracker = ...
  uimenu('Parent',handles.menu_track,...
         'Callback',@(s,e)(handles.controller.controlActuated('menu_track_delete_current_tracker', s, e)), ...
         'Label','Delete current tracker',...
         'Tag','menu_track_delete_current_tracker',...
         'Separator','off');
moveMenuItemAfter(handles.menu_track_delete_current_tracker, handles.menu_track_reset_current_tracker) ;

handles.menu_track_delete_old_trackers = ...
  uimenu('Parent',handles.menu_track,...
         'Callback',@(s,e)(handles.controller.controlActuated('menu_track_delete_old_trackers', s, e)), ...
         'Label','Delete old trackers',...
         'Tag','menu_track_delete_old_trackers',...
         'Separator','off');
moveMenuItemAfter(handles.menu_track_delete_old_trackers, handles.menu_track_delete_current_tracker) ;

handles.menu_track_set_labels = uimenu('Parent',handles.menu_track,...
  'Callback',@(hObject,eventdata)LabelerGUI('menu_track_set_labels_Callback',hObject,eventdata,guidata(hObject)),...
  'Label','Set manual labels to automatic prediction',...
  'Tag','menu_track_set_labels',...
  'Separator','on');  

handles.menu_track_cpr_storefull = uimenu('Parent',handles.menu_track,...
  'Label','(CPR) Store tracking replicates/iterations',...
  'Tag','menu_track_cpr_storefull',...
  'Separator','on');
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

handles.menu_track_auto_params_update = ...
  uimenu('Parent',handles.menu_track,...
         'Callback',@(hObject,eventdata)LabelerGUI('menu_track_auto_params_update_Callback',hObject,eventdata,guidata(hObject)),...
         'Label','Auto compute training parameters',...
         'Tag','menu_track_auto_params_update',...
         'Checked','on',...
         'Visible','on');
moveMenuItemAfter(handles.menu_track_auto_params_update,...
                  handles.menu_track_setparametersfile) ;

% Track > Tracker history submenu
handles.menu_track_tracker_history = ...
  uimenu('Parent',handles.menu_track, ...
         'Label','Tracker history', ...
         'Tag','menu_track_tracker_history', ...
         'Visible','on');
moveMenuItemAfter(handles.menu_track_tracker_history, ...
                  handles.menu_track_tracking_algorithm) ;



% Help menu
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
pumTrack.Tag = 'pumTrack' ;
pumTrack.Value = 1;
pumTrack.String = {'All frames'};
%set(pumTrack,'FontUnits','points','FontSize',6.5);
%pumTrack.FontUnits = 'normalized';
handles.pumTrack.Callback = ...
  @(hObj,edata)LabelerGUI('pumTrack_Callback',hObj,edata,guidata(hObj));

% handles.labelTLInfo = InfoTimeline(labeler,handles.axes_timeline_manual,...
%                                    handles.axes_timeline_islabeled);
% 
% set(handles.pumInfo,'String',handles.labelTLInfo.getPropsDisp(),...
%   'Value',handles.labelTLInfo.curprop);
% set(handles.pumInfo_labels,'String',handles.labelTLInfo.getPropTypesDisp(),...
%   'Value',handles.labelTLInfo.curproptype);
% 
% % this is currently not used - KB made space here for training status
% %set(handles.txProjectName,'String','');
% 
% listeners = cell(0,1);
% listeners{end+1,1} = addlistener(handles.slider_frame,'ContinuousValueChange',@slider_frame_Callback);
% listeners{end+1,1} = addlistener(handles.sldZoom,'ContinuousValueChange',@sldZoom_Callback);
% listeners{end+1,1} = addlistener(handles.axes_curr,'XLim','PostSet',@(s,e)axescurrXLimChanged(s,e,handles));
% listeners{end+1,1} = addlistener(handles.axes_curr,'XDir','PostSet',@(s,e)axescurrXDirChanged(s,e,handles));
% listeners{end+1,1} = addlistener(handles.axes_curr,'YDir','PostSet',@(s,e)axescurrYDirChanged(s,e,handles));
% % listeners{end+1,1} = addlistener(labeler,'didSetProjname',@cbkProjNameChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetCurrTarget',@cbkCurrTargetChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetLastLabelChangeTS',@cbkLastLabelChangeTS);
% % listeners{end+1,1} = addlistener(labeler,'didSetTrackParams',@cbkParameterChange);
% listeners{end+1,1} = addlistener(labeler,'didSetLabelMode',@cbkLabelModeChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLabels2Hide',@cbkLabels2HideChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLabels2ShowCurrTargetOnly',@cbkLabels2ShowCurrTargetOnlyChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetProjFSInfo',@cbkProjFSInfoChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowTrx',@cbkShowTrxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowOccludedBox',@cbkShowOccludedBoxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowTrxCurrTargetOnly',@cbkShowTrxCurrTargetOnlyChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetTrackersAll',@cbkTrackersAllChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetCurrTracker',@cbkCurrTrackerChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackModeIdx',@cbkTrackModeIdxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesSmall',@cbkTrackerNFramesChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesLarge',@cbkTrackerNFramesChanged);    
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesNear',@cbkTrackerNFramesChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieCenterOnTarget',@cbkMovieCenterOnTargetChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieRotateTargetUp',@cbkMovieRotateTargetUpChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieForceGrayscale',@cbkMovieForceGrayscaleChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetMovieInvert',@cbkMovieInvertChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieViewBGsubbed',@cbkMovieViewBGsubbedChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLblCore',@(src,evt)(handles.controller.didSetLblCore()));
% listeners{end+1,1} = addlistener(labeler,'gtIsGTModeChanged',@cbkGtIsGTModeChanged);
% listeners{end+1,1} = addlistener(labeler,'cropIsCropModeChanged',@cbkCropIsCropModeChanged);
% listeners{end+1,1} = addlistener(labeler,'cropUpdateCropGUITools',@cbkUpdateCropGUITools);
% listeners{end+1,1} = addlistener(labeler,'cropCropsChanged',@cbkCropCropsChanged);
% %listeners{end+1,1} = addlistener(labeler,'newProject',@cbkNewProject);
% listeners{end+1,1} = addlistener(labeler,'newMovie',@cbkNewMovie);
% %listeners{end+1,1} = addlistener(labeler,'projLoaded',@cbkProjLoaded);
% listeners{end+1,1} = addlistener(handles.labelTLInfo,'selectOn','PostSet',@cbklabelTLInfoSelectOn);
% listeners{end+1,1} = addlistener(handles.labelTLInfo,'props','PostSet',@cbklabelTLInfoPropsUpdated);
% listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_tracker','PostSet',@cbklabelTLInfoPropsUpdated);
% listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_allframes','PostSet',@cbklabelTLInfoPropsUpdated);
% listeners{end+1,1} = addlistener(handles.labelTLInfo,'proptypes','PostSet',@cbklabelTLInfoPropTypesUpdated);
% %listeners{end+1,1} = addlistener(labeler,'startAddMovie',@cbkAddMovie);
% %listeners{end+1,1} = addlistener(labeler,'finishAddMovie',@cbkAddMovie);
% %listeners{end+1,1} = addlistener(labeler,'startSetMovie',@cbkSetMovie);
% listeners{end+1,1} = addlistener(labeler,'dataImported',@cbkDataImported);
% listeners{end+1,1} = addlistener(labeler,'didSetShowSkeleton',@cbkShowSkeletonChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowMaRoi',@cbkShowMaRoiChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowMaRoiAux',@cbkShowMaRoiAuxChanged);
% 
% handles.listeners = listeners;

%handles.listenersTracker = cell(0,1); % listeners added in LabelerController::cbkCurrTrackerChanged
%handles.menu_track_trackers = cell(0,1); % menus added in LabelerController::cbkTrackersAllChanged()

hZ = zoom(hObject);
hZ.ActionPostCallback = @cbkPostZoom;
hP = pan(hObject);
hP.ActionPostCallback = @cbkPostPan;

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

%handles.depHandles = gobjects(0,1);  
% keep these in labelerController.satellites_ now

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

%handles.controller.enableControls_('tooltipinit');
set(handles.figure,'Visible','on');
% if labeler.isgui ,
%   RefocusSplashScreen(hfigsplash,handles);
% end

LabelerTooltips(handles);
% if labeler.isgui,
%   RefocusSplashScreen(hfigsplash,handles);
%   if ishandle(hfigsplash),
%     delete(hfigsplash);
%   end
% end

% get rid of extra toolbars
h = findall(hObject,'-property','Toolbar');
for i = 1:numel(h),
  htool = get(h(i),'Toolbar');
  if ishandle(htool),
    set(htool,'Visible','off');
  end
end

% Change some controls to use LabelerGUIControlActuated()
handles.menu_file_new.Callback = @LabelerGUIControlActuated ;
handles.menu_file_shortcuts.Callback = @LabelerGUIControlActuated ;
handles.menu_view_reset_views.Callback = @LabelerGUIControlActuated ;
handles.menu_file_quick_open.Callback = @LabelerGUIControlActuated ;
handles.pbTrain.Callback = @LabelerGUIControlActuated ;
handles.pbTrack.Callback = @LabelerGUIControlActuated ;

% Add the Debug menu, but leave hidden
handles.menu_debug = ...
  uimenu('Parent', hObject, ...
         'Label', 'Debug', ...
         'Tag', 'menu_debug', ...
         'Visible', 'off') ;
handles.menu_start_training_but_dont_call_python = ...
  uimenu('Parent', handles.menu_debug, ...
         'Label', 'Start Training, But Skip Python Call', ...
         'Tag', 'menu_start_training_but_dont_call_python', ...
         'Callback', @LabelerGUIControlActuated) ;
handles.menu_debug_generate_db = ...
  uimenu('Parent', handles.menu_debug, ...
         'Label', 'Start Training, But Just Generate DB', ...
         'Tag', 'menu_debug_generate_db', ...
         'Callback', @LabelerGUIControlActuated) ;
handles.menu_start_tracking_but_dont_call_python = ...
  uimenu('Parent', handles.menu_debug, ...
         'Label', 'Start Tracking, But Skip Python Call', ...
         'Tag', 'menu_start_tracking_but_dont_call_python', ...
         'Callback', @LabelerGUIControlActuated) ;
handles.menu_quit_but_dont_delete_temp_folder = ...
  uimenu('Parent', handles.menu_debug, ...
         'Label', 'Quit, But Don''t Delete Temp Folder', ...
         'Tag', 'menu_quit_but_dont_delete_temp_folder', ...
         'Callback', @LabelerGUIControlActuated) ;

if ismac()  % Change color of buttons 
 toChange = {'pbClear','tbAccept','pbTrain','pbTrack',...
     'pumTrack','tbTLSelectMode','pbClearSelection',...
     'pumInfo_labels','pumInfo'};
 for ndx = 1:numel(toChange)
     set(handles.(toChange{ndx}),'ForegroundColor',[1.0,0.0,1.0]);
 end
end

% Write the modified handles structure back to the figure guidata
guidata(hObject, handles);

fprintf('Labeler GUI created.\n');

% UIWAIT makes LabelerGUI wait for user response (see UIRESUME)
% uiwait(handles.figure);

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



function cbkAuxAxResize(src,data)
% AL 20160628: voodoo that may help make points more clickable. Sometimes
% pt clickability in MultiViewCalibrated mode is unstable (eg to anchor
% points etc)
ax = findall(src,'type','axes');
axis(ax,'image')
axis(ax,'auto');

function cbkWBMF(src,evt,labeler)
lcore = labeler.lblCore;
if ~isempty(lcore)
  lcore.wbmf(src,evt);
end

function cbkWBUF(src,evt,labeler)
if ~isempty(labeler.lblCore)
  labeler.lblCore.wbuf(src,evt);
end

function cbkWSWF(src,evt,labeler)

if ~labeler.hasMovie
  return;
end

scrollcnt = evt.VerticalScrollCount;
scrollamt = evt.VerticalScrollAmount;
fcurr = labeler.currFrame;
f = fcurr - round(scrollcnt*scrollamt); % scroll "up" => larger frame number
f = min(max(f,1),labeler.nframes);
cmod = labeler.gdata.figure.CurrentModifier;
tfMod = ~isempty(cmod) && any(strcmp(cmod{1},{'control' 'shift'}));
if tfMod
  if f>fcurr
    labeler.frameUp(true);
  else
    labeler.frameDown(true);
  end
else
  labeler.setFrameProtected(f);
end

% function setGUIMainFigureName(labeler)
% 
% maxlength = 80;
% if isempty(labeler.projectfile),
%   projname = [labeler.projname,' (unsaved)'];
% elseif numel(labeler.projectfile) <= maxlength,
%   projname = labeler.projectfile;
% else
%   [~,projname,ext] = fileparts(labeler.projectfile);
%   projname = [projname,ext];
% end
% labeler.gdata.figure.Name = sprintf('APT - Project %s',projname);



% function cbkCurrFrameChanged(src,evt) %#ok<*INUSD>
% ticinfo = tic;
% starttime = ticinfo;
% labeler = evt.AffectedObject;
% frm = labeler.currFrame;
% nfrm = labeler.nframes;
% handles = labeler.gdata;
% fprintf('cbkCurrFrameChanged 1 get stuff: %f\n',toc(ticinfo)); ticinfo = tic;
% set(handles.edit_frame,'String',num2str(frm));
% fprintf('cbkCurrFrameChanged 2 set edit_frame: %f\n',toc(ticinfo)); ticinfo = tic;
% sldval = (frm-1)/(nfrm-1);
% if isnan(sldval)
%   sldval = 0;
% end
% set(handles.slider_frame,'Value',sldval);
% fprintf('cbkCurrFrameChanged 3 slider_frame %f\n',toc(ticinfo)); ticinfo = tic;
% if ~labeler.isinit
%   %handles.labelTLInfo.newFrame(frm);
%   fprintf('cbkCurrFrameChanged 4 labelTLInfo.newFrame %f\n',toc(ticinfo)); ticinfo = tic;
%   hlpGTUpdateAxHilite(labeler);
%   fprintf('cbkCurrFrameChanged 5 axhilite %f\n',toc(ticinfo));
% end
% fprintf('->cbkCurrFrameChanged total time: %f\n',toc(starttime));

% function hlpGTUpdateAxHilite(labeler)
% if labeler.gtIsGTMode
%   tfHilite = labeler.gtCurrMovFrmTgtIsInGTSuggestions();
% else
%   tfHilite = false;
% end
% labeler.gdata.allAxHiliteMgr.setHighlight(tfHilite);

function hlpUpdateTblTrxHilite(labeler)

try
  if labeler.maIsMA
    labeler.gdata.tblTrx.SelectedRows = [];
  else
    i = find(labeler.currTarget == labeler.tblTrxData(:,1));
    assert(numel(i) == 1);
    labeler.gdata.tblTrx.SelectedRows = i;
  end
catch exception
  warningNoTrace('Error caught updating highlight row in Targets Table:\n%s\n', exception.getReport());
end



%lc = labeler.lblCore;
%tfShow3DAxes = ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration;
% handles.menu_view_show_3D_axes.Enable = onIff(tfShow3DAxes);

% function hlpUpdateTxProjectName(labeler)
% projname = labeler.projname;
% info = labeler.projFSInfo;
% if isempty(info)
%   str = projname;
% else
%   [~,projfileS] = myfileparts(info.filename);  
%   str = sprintf('%s / %s',projfileS,projname);
% end
% hTX = labeler.gdata.txProjectName;
% hTX.String = str;

% function cbkProjNameChanged(src,evt)
% labeler = src ;
% str = sprintf('Project $PROJECTNAME created (unsaved) at %s',datestr(now(),16));
% labeler.setRawStatusStringWhenClear_(str) ;
% controller = labeler.controller_ ;
% controller.updateMainFigureName() ;

% function cbkProjFSInfoChanged(src,evt)
% labeler = src ;
% info = labeler.projFSInfo;
% if ~isempty(info)  
%   str = sprintf('Project $PROJECTNAME %s at %s',info.action,datestr(info.timestamp,16));
%   labeler.setRawStatusStringWhenClear_(str) ;
% end
% setGUIMainFigureName(labeler);

% function cbkSuspScoreChanged(src,evt)
% labeler = evt.AffectedObject;
% ss = labeler.suspScore;
% labeler.currImHud.updateReadoutFields('hasSusp',~isempty(ss));
% 
% assert(~labeler.gtIsGTMode,'Unsupported in GT mode.');
% 
% handles = labeler.gdata;
% pnlSusp = handles.pnlSusp;
% tblSusp = handles.tblSusp;
% tfDoSusp = ~isempty(ss) && labeler.hasMovie && ~labeler.isinit;
% if tfDoSusp 
%   nfrms = labeler.nframes;
%   ntgts = labeler.nTargets;
%   [tgt,frm] = meshgrid(1:ntgts,1:nfrms);
%   ss = ss{labeler.currMovie};
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
%   labeler.updateCurrSusp();
% else
%   tblSusp.Data = cell(0,3);
%   pnlSusp.Visible = 'off';
% end

% function cbkCurrSuspChanged(src,evt)
% labeler = evt.AffectedObject;
% ss = labeler.currSusp;
% if ~isequal(ss,[])
%   labeler.currImHud.updateSusp(ss);
% end

function cbkTrackerBackendAWSSetInstance(src,evt)
error('Implemented elsewhere') ;

function cbkTrackerBackendAWSConfigure(src,evt)
error('Implemented elsewhere') ;

% function cbkCurrTrackerPreChanged(src,evt)
% labeler = src ;
% if labeler.isinit ,
%   return
% end 
% tObj = labeler.tracker ;
% if ~isempty(tObj) ,
%   tObj.deactivate() ;
% end


function pumTrack_Callback(hObj,edata,handles)
labeler = handles.labeler;
labeler.trackModeIdx = hObj.Value;

% function mftset = getTrackMode(handles)
% idx = handles.pumTrack.Value;
% % Note, .TrackingMenuNoTrx==.TrackingMenuTrx(1:K), so we can just index
% % .TrackingMenuTrx.
% mfts = MFTSetEnum.TrackingMenuTrx;
% mftset = mfts(idx);

function slider_frame_Callback(hObject,evt,varargin)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

debugtiming = false;
if debugtiming,
  starttime = tic() ;
end

handles = guidata(hObject);
labeler = handles.labeler;

if ~labeler.hasProject
  set(hObject,'Value',0);  
  return;
end
if ~labeler.hasMovie
  set(hObject,'Value',0);  
  msgbox('There is no movie open.');
  return;
end

v = get(hObject,'Value');
f = round(1 + v * (labeler.nframes - 1));

cmod = handles.figure.CurrentModifier;
if ~isempty(cmod) && any(strcmp(cmod{1},{'control' 'shift'}))
  if f>labeler.currFrame
    tfSetOccurred = labeler.frameUp(true);
  else
    tfSetOccurred = labeler.frameDown(true);
  end
else
  tfSetOccurred = labeler.setFrameProtected(f);
end
  
if ~tfSetOccurred
  sldval = (labeler.currFrame-1)/(labeler.nframes-1);
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

labeler = handles.labeler;

f = str2double(get(hObject,'String'));
if isnan(f)
  set(hObject,'String',num2str(labeler.currFrame));
  return;
end
f = min(max(1,round(f)),labeler.nframes);
if ~labeler.trxCheckFramesLive(f)
  set(hObject,'String',num2str(labeler.currFrame));
  warnstr = sprintf('Frame %d is out-of-range for current target.',f);
  warndlg(warnstr,'Out of range');
  return;
end
set(hObject,'String',num2str(f));
if f ~= labeler.currFrame
  labeler.setFrame(f)
end 
  
function edit_frame_CreateFcn(hObject,~,~)
if ispc && isequal(get(hObject,'BackgroundColor'), ...
                   get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


function pbTrain_Callback(hObject, eventdata, handles)
% Not used anymore.  See LabelerController::pbTrain_actuated()


function pbTrack_Callback(hObject, eventdata, handles)
% Not used anymore.  See LabelerController::pbTrain_actuated()
% if ~checkProjAndMovieExist(handles)
%   return;
% end
% handles.labeler.setStatus('Tracking...');
% tm = getTrackMode(handles);
% tblMFT = tm.getMFTable(handles.labeler,'istrack',true);
% if isempty(tblMFT),
%   msgbox('All frames tracked.','Track');
%   handles.labeler.clearStatus() ;
%   return;
% end
% [tfCanTrack,reason] = handles.labeler.trackCanTrack(tblMFT);
% if ~tfCanTrack,
%   errordlg(['Error tracking: ',reason],'Error tracking');
%   handles.labeler.clearStatus();
%   return;
% end
% fprintf('Tracking started at %s...\n',datestr(now));
% wbObj = WaitBarWithCancel('Tracking');
% centerOnParentFigure(wbObj.hWB,handles.figure);
% oc = onCleanup(@()delete(wbObj));
% if handles.labeler.maIsMA
%   handles.labeler.track(tblMFT,'wbObj',wbObj,'track_type','detect');
% else
%   handles.labeler.track(tblMFT,'wbObj',wbObj);
% end
% if wbObj.isCancel
%   msg = wbObj.cancelMessage('Tracking canceled');
%   msgbox(msg,'Track');
% end
% handles.labeler.clearStatus();


function pbClear_Callback(hObject, eventdata, handles)

if ~checkProjAndMovieExist(handles)
  return;
end
handles.labeler.lblCore.clearLabels();
handles.labeler.CheckPrevAxesTemplate();

function tbAccept_Callback(hObject, eventdata, handles)
% debugtiming = false;
% if debugtiming,
%   starttime = tic;
% end

if ~checkProjAndMovieExist(handles)
  return;
end
lc = handles.labeler.lblCore;
switch lc.state
  case LabelState.ADJUST
    lc.acceptLabels();
    %handles.labeler.InitializePrevAxesTemplate();
  case LabelState.ACCEPTED
    lc.unAcceptLabels();    
    %handles.labeler.CheckPrevAxesTemplate();
  otherwise
    assert(false);
end

% if debugtiming,
%   fprintf('toggleAccept took %f seconds\n',toc(starttime));
% end

function cbkTblTrxCellSelection(src,evt) %#ok<*DEFNU>
% Current/last row selection is maintained in hObject.UserData

handles = guidata(src.Parent);
labeler = handles.labeler;
if ~(labeler.hasTrx || labeler.maIsMA)
  return;
end

rows = evt.Indices;
rows = rows(:,1); % AL20210514: rows is nx2; columns are {rowidxs,colidxs} at least in 2020b
%rowsprev = src.UserData;
src.UserData = rows;
dat = get(src,'Data');

if isscalar(rows)
  idx = dat{rows(1),1};
  labeler.setTarget(idx);
  %labeler.labelsOtherTargetHideAll();
else
  % 20210514 Skipping this for now; possible performance hit
  
  % addon to existing selection
  %rowsnew = setdiff(rows,rowsprev);  
  %idxsnew = cell2mat(dat(rowsnew,1));
  %labeler.labelsOtherTargetShowIdxs(idxsnew);
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
labeler = handles.labeler;
row = evt.Indices;
if ~isempty(row)
  row = row(1);
  dat = get(src,'Data');
  labeler.setFrame(dat{row,1},'changeTgtsIfNec',true);
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
    handles.labeler.movieRotateTargetUp
  warningNoTrace('LabelerGUI:axDir',...
    'Main axis ''XDir'' or ''YDir'' is set to be flipped and .movieRotateTargetUp is set. Graphics behavior may be unexpected; proceed at your own risk.');
end

function scroll_callback(hObject,eventdata,labeler)
gdata = labeler.gdata;
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

labeler = handles.labeler;
v = hObject.Value;
userdata = hObject.UserData;
logzoomrad = userdata(2)+v*(userdata(1)-userdata(2));
zoomRad = exp(logzoomrad);
labeler.videoZoom(zoomRad);
hlpRemoveFocus(hObject,handles);

function cbkPostZoom(src,evt)
if verLessThan('matlab','R2016a')
  setappdata(src,'manualZoomOccured',true);
end
handles = guidata(src);
if evt.Axes == handles.axes_prev,
  handles.labeler.UpdatePrevAxesLimits();
end

function cbkPostPan(src,evt)
handles = guidata(src);
if evt.Axes == handles.axes_prev,
  handles.labeler.UpdatePrevAxesLimits();
end

function pbResetZoom_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms,false(1,numel(hIms)));

function pbSetZoom_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.targetZoomRadiusDefault = diff(handles.axes_curr.XLim)/2;

function pbRecallZoom_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
% TODO this is broken!!
labeler.videoCenterOnCurrTarget();
labeler.videoZoom(labeler.targetZoomRadiusDefault);

function tblSusp_CellSelectionCallback(hObject, eventdata, handles)
labeler = handles.labeler;
if verLessThan('matlab','R2015b')
  jt = labeler.gdata.tblSusp.UserData.jtable;
  row = jt.getSelectedRow; % 0 based
  frm = jt.getValueAt(row,0);
  iTgt = jt.getValueAt(row,1);
  if ~isempty(frm)
    frm = frm.longValueReal;
    iTgt = iTgt.longValueReal;
    labeler.setFrameAndTarget(frm,iTgt);
    hlpRemoveFocus(hObject,handles);
  end
else
  row = eventdata.Indices(1);
  dat = hObject.Data;
  frm = dat(row,1);
  iTgt = dat(row,2);
  labeler.setFrameAndTarget(frm,iTgt);
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
tb = lblTLObj.labeler.gdata.tbTLSelectMode;
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
handles.labeler.setPrevAxesMode(PrevAxesMode.FROZEN);

function cbkUnfreezePrevAxes(src,evt)
handles = guidata(src);
handles.labeler.setPrevAxesMode(PrevAxesMode.LASTSEEN);

%% menu
function menu_file_save_Callback(hObject, eventdata, handles)
labeler = handles.labeler ;
labeler.setStatus('Saving project...');
handles.labeler.projSaveSmart();
handles.labeler.projAssignProjNameFromProjFileIfAppropriate();
handles.labeler.clearStatus()

function menu_file_saveas_Callback(hObject, eventdata, handles)
handles.labeler.setStatus('Saving project...');
handles.labeler.projSaveAs();
handles.labeler.projAssignProjNameFromProjFileIfAppropriate();
handles.labeler.clearStatus()

function menu_file_load_Callback(hObject, eventdata, handles)

labeler = handles.labeler;
controller = handles.controller ;
labeler.setStatus('Loading Project...') ;
if controller.raiseUnsavedChangesDialogIfNeeded() ,
  currMovInfo = labeler.projLoad();
  if ~isempty(currMovInfo)
    controller.movieManagerController_.setVisible(true);
    wstr = ...
      sprintf(strcatg('Could not find file for movie(set) %d: %s.\n\nProject opened with no movie selected. ', ...
                      'Double-click a row in the MovieManager or use the ''Switch to Movie'' button to start working on a movie.'), ...
              currMovInfo.iMov, ...
              currMovInfo.badfile);
    warndlg(wstr,'Movie not found','modal');
  end
end
labeler.clearStatus()

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
if ~isempty(handles.controller.movieManagerController_) && isvalid(handles.controller.movieManagerController_) ,
  handles.controller.movieManagerController_.setVisible(true);
else
  handles.labeler.lerror('LabelerGUI:movieManagerController','Please create or load a project.');
end

function menu_file_import_labels_trk_curr_mov_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
if ~labeler.hasMovie
  handles.labeler.lerror('LabelerGUI:noMovie','No movie is loaded.');
end
labeler.gtThrowErrIfInGTMode();
iMov = labeler.currMovie;
haslbls1 = labeler.labelPosMovieHasLabels(iMov); % TODO: method should be unnec
haslbls2 = labeler.movieFilesAllHaveLbls(iMov)>0;
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
handles.labeler.labelImportTrkPromptGenericSimple(iMov,...
  'labelImportTrk','gtok',false);

function menu_file_import_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
if ~labeler.hasMovie
  handles.labeler.lerror('LabelerGUI:noMovie','No movie is loaded.');
end
iMov = labeler.currMovie; % gt-aware
handles.labeler.setStatus('Importing tracking results...');
labeler.labelImportTrkPromptGenericSimple(iMov,'labels2ImportTrk','gtok',true);
handles.labeler.clearStatus();

function menu_file_export_labels_trks_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI('labels',true);
if ~tfok
  return;
end
handles.labeler.setStatus('Exporting tracking results...');
labeler.labelExportTrk(1:labeler.nmoviesGTaware,'rawtrkname',rawtrkname);
handles.labeler.clearStatus();

function menu_file_export_labels_table_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
fname = labeler.getDefaultFilenameExportLabelTable();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return;
end
fname = fullfile(p,f);  
VARNAME = 'tblLbls';
s = struct();
s.(VARNAME) = labeler.labelGetMFTableLabeled('useMovNames',true); 
save(fname,'-mat','-struct','s');
fprintf('Saved table ''%s'' to file ''%s''.\n',VARNAME,fname);

function menu_file_import_labels_table_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
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
labeler.labelPosBulkImportTbl(t);
fprintf('Loaded %d labeled frames from file ''%s''.\n',height(t),fname);

function menu_file_export_stripped_lbl_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
fname = labeler.getDefaultFilenameExportStrippedLbl();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return
end
fname = fullfile(p,f);
handles.labeler.setStatus(sprintf('Exporting training data to %s',fname));
labeler.projExportTrainData(fname)
fprintf('Saved training data to file ''%s''.\n',fname);
handles.labeler.clearStatus();

function menu_file_crop_mode_Callback(hObject,evtdata,handles)

labeler = handles.labeler;

if ~isempty(labeler.tracker) && ~labeler.gtIsGTMode && labeler.labelPosMovieHasLabels(labeler.currMovie),
  res = questdlg('Frames of the current movie are labeled. Editing the crop region for this movie will cause trackers to be reset. Continue?');
  if ~strcmpi(res,'Yes'),
    return;
  end
end

handles.labeler.setStatus('Switching crop mode...');
labeler.cropSetCropMode(~labeler.cropIsCropMode);
handles.labeler.clearStatus();

function menu_file_clean_tempdir_Callback(hObject,evtdata,handles)

handles.labeler.setStatus('Deleting temp directories...');
handles.labeler.projRemoveOtherTempDirs();
handles.labeler.clearStatus();

function menu_file_bundle_tempdir_Callback(hObject,evtdata,handles)
handles.labeler.setStatus('Bundling the temp directory...');
handles.labeler.projBundleTempDir();
handles.labeler.clearStatus();


function menu_help_Callback(hObject, eventdata, handles)

function menu_help_labeling_actions_Callback(hObject, eventdata, handles)
lblCore = handles.labeler.lblCore;
if isempty(lblCore)
  h = 'Please open a movie first.';
else
  h = lblCore.getLabelingHelp();
end
msgbox(h,'Labeling Actions','help',struct('Interpreter','tex','WindowStyle','replace'));

function menu_help_about_Callback(hObject, eventdata, handles)
about(handles.labeler);

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
handles.labeler.labelingInit('labelMode',lblMode);

function menu_setup_label_overlay_montage_Callback(hObject,evtdata,handles)
handles.labeler.setStatus('Plotting all labels on one axes to visualize label distribution...');
labeler = handles.labeler;
if labeler.hasTrx
  labeler.labelOverlayMontage();
  labeler.labelOverlayMontage('ctrMeth','trx');
  labeler.labelOverlayMontage('ctrMeth','trx','rotAlignMeth','trxtheta');
  % could also use headtail for centering/alignment but skip for now.  
else % labeler.maIsMA, or SA-non-trx
  labeler.labelOverlayMontage();
  if ~labeler.isMultiView
    labeler.labelOverlayMontage('ctrMeth','centroid');
    tfHTdefined = ~isempty(labeler.skelHead) && ~isempty(labeler.skelTail);
    if tfHTdefined  
      labeler.labelOverlayMontage('ctrMeth','centroid','rotAlignMeth','headtail');
    else
      warningNoTrace('For aligned overlays, define head/tail points in Track>Landmark Paraneters.');
    end
  end
end
handles.labeler.clearStatus();

% function menu_setup_label_overlay_montage_trx_centered_Callback(hObject,evtdata,handles)
% 
% handles.labeler.setStatus('Plotting all labels on one axes to visualize label distribution...');
% labeler = handles.labeler;
% hFig(1) = labeler.labelOverlayMontage('ctrMeth','trx','rotAlignMeth','none'); 
% try
%   hFig(2) = labeler.labelOverlayMontage('ctrMeth','trx',...
%     'rotAlignMeth','headtail','hFig0',hFig(1)); 
% catch ME
%   warningNoTrace('Could not create head-tail aligned montage: %s',ME.message);
%   hFig(2) = figurecascaded(hFig(1));
% end
% hFig(3) = labeler.labelOverlayMontage('ctrMeth','trx',...
%   'rotAlignMeth','trxtheta','hFig0',hFig(2)); %#ok<NASGU>
% handles.labeler.clearStatus();

function menu_setup_label_outliers_Callback(hObject,evtdata,handles)
handles.labeler.setStatus('Finding outliers in labels...');
labeler = handles.labeler;
label_outlier_gui(labeler);
handles.labeler.clearStatus();

function menu_setup_set_nframe_skip_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lc = labeler.lblCore;
assert(isa(lc,'LabelCoreHT'));
nfs = lc.nFrameSkip;
ret = inputdlg('Select labeling frame increment','Set increment',1,{num2str(nfs)});
if isempty(ret)
  return;
end
val = str2double(ret{1});
lc.nFrameSkip = val;
labeler.labelPointsPlotInfo.HighThroughputMode.NFrameSkip = val;
% This state is duped between labelCore and lppi b/c the lifetimes are
% different. LabelCore exists only between movies etc, and is initted from
% lppi. Hmm

function menu_setup_streamlined_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lc = labeler.lblCore;
assert(isa(lc,'LabelCoreMultiViewCalibrated2'));
lc.streamlined = ~lc.streamlined;

function menu_setup_ma_twoclick_align_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lc = labeler.lblCore;
tftc = ~lc.tcOn;
labeler.isTwoClickAlign = tftc; % store the state
lc.setTwoClickOn(tftc);
hObject.Checked = onIff(tftc); % skip listener business for now

function menu_setup_set_labeling_point_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
ipt = labeler.lblCore.iPoint;
ret = inputdlg('Select labeling point','Point number',1,{num2str(ipt)});
if isempty(ret)
  return;
end
ret = str2double(ret{1});
labeler.lblCore.setIPoint(ret);

function set_use_calibration(handles,v)

labeler = handles.labeler;
lc = labeler.lblCore;
if lc.supportsCalibration,
  lc.setShowCalibration(v);
end
handles.menu_setup_use_calibration.Checked = onIff(lc.showCalibration);

function menu_setup_use_calibration_Callback(hObject, eventdata, handles)

labeler = handles.labeler;
lc = labeler.lblCore;
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

labeler = handles.labeler;
vcdPW = labeler.viewCalProjWide;
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

% Currently there is no UI for altering labeler.viewCalProjWide once it is set

if tfProjWide
  labeler.viewCalSetProjWide(crObj);%,'tfSetViewSizes',tfSetViewSizes);
else
  labeler.viewCalSetCurrMovie(crObj);%,'tfSetViewSizes',tfSetViewSizes);
end
set_use_calibration(handles,true);

RC.saveprop('lastCalibrationFile',fname);

% function menu_setup_unlock_all_frames_Callback(hObject, eventdata, handles)
% handles.labeler.labelPosSetAllMarked(false);
% function menu_setup_lock_all_frames_Callback(hObject, eventdata, handles)
% handles.labeler.labelPosSetAllMarked(true);

function CloseImContrast(labeler,iAxRead,iAxApply)
% ReadClim from axRead and apply to axApply

axAll = labeler.gdata.axes_all;
axRead = axAll(iAxRead);
axApply = axAll(iAxApply);
tfApplyAxPrev = any(iAxApply==1); % axes_prev mirrors axes_curr

clim = get(axRead,'CLim');
if isempty(clim)
	% none; can occur when Labeler is closed
else
  labeler.clim_manual = clim;
  warnst = warning('off','MATLAB:graphicsversion:GraphicsVersionRemoval');
	set(axApply,'CLim',clim);
	warning(warnst);
	if tfApplyAxPrev
		set(labeler.gdata.axes_prev,'CLim',clim);
	end
end		

function [tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(handles)
labeler = handles.labeler;
if ~labeler.isMultiView
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
labeler = handles.labeler;
labeler.movieViewBGsubbed = tf;

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
		@(s,e) CloseImContrast(handles.labeler,iAxRead,iAxApply));
end
  
function menu_view_converttograyscale_Callback(hObject, eventdata, handles)
tf = ~strcmp(hObject.Checked,'on');
labeler = handles.labeler;
labeler.movieForceGrayscale = tf;
if labeler.hasMovie
  % Pure convenience: update image for user rather than wait for next 
  % frame-switch. Could also put this in Labeler.set.movieForceGrayscale.
  labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
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
% labeler = evt.AffectedObject;
% handles = labeler.gdata;
% onOff = onIff(~labeler.showPredTxtLbl);
% handles.menu_view_showhide_advanced_hidepredtxtlbls.Checked = onOff;

function menu_view_hide_trajectories_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.setShowTrx(~labeler.showTrx);

function menu_view_plot_trajectories_current_target_only_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.setShowTrxCurrTargetOnly(~labeler.showTrxCurrTargetOnly);

function menu_view_trajectories_centervideoontarget_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.movieCenterOnTarget = ~labeler.movieCenterOnTarget;
function menu_view_rotate_video_target_up_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.movieRotateTargetUp = ~labeler.movieRotateTargetUp;
function menu_view_flip_flipud_movie_only_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  labeler = handles.labeler;
  labeler.movieInvert(iAxApply) = ~labeler.movieInvert(iAxApply);
  if labeler.hasMovie
    labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
  end
end
function menu_view_flip_flipud_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles);
if tfproceed
  for iAx = iAxApply(:)'
    ax = handles.axes_all(iAx);
    ax.YDir = toggleAxisDir(ax.YDir);
  end
  handles.labeler.UpdatePrevAxesDirections();
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
    handles.labeler.UpdatePrevAxesDirections();
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
handles.labeler.movieCenterOnTarget = false;


function menu_view_hide_labels_Callback(hObject, eventdata, handles)
lblCore = handles.labeler.lblCore;
if ~isempty(lblCore)
  lblCore.labelsHideToggle();
end

function menu_view_hide_predictions_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
tracker = labeler.tracker;
if ~isempty(tracker)
  tracker.hideVizToggle();
end

function menu_view_show_preds_curr_target_only_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
tracker = labeler.tracker;
if ~isempty(tracker)
  tracker.showPredsCurrTargetOnlyToggle();
end

function menu_view_hide_imported_predictions_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.labels2VizToggle();

function menu_view_show_imported_preds_curr_target_only_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.labels2VizSetShowCurrTargetOnly(~labeler.labels2ShowCurrTargetOnly);

function menu_track_cpr_show_replicates_Callback(hObject, eventdata, handles)
tObj = handles.labeler.tracker;
vsr = tObj.showVizReplicates;
vsrnew = ~vsr;
sft = tObj.storeFullTracking;
if vsrnew && sft==StoreFullTrackingType.NONE
  warningNoTrace('Tracker will store replicates for final CPR iterations.');
  tObj.storeFullTracking = StoreFullTrackingType.FINALITER;
end
tObj.showVizReplicates = vsrnew;

% % when trackerInfo is updated, update the tracker info text in the main APT window
% function cbkTrackerInfoChanged(src,evt)
% 
% tObj = evt.AffectedObject;
% tObj.labeler.gdata.text_trackerinfo.String = tObj.getTrackerInfoString();

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
%   deleteValidGraphicsHandles(handles.hShow3D);
% end
% handles.hShow3D = gobjects(0,1);
% 
% tfHide = strcmp(hObject.Checked,'on');
% 
% if tfHide
%   hObject.Checked = 'off';
% else
%   labeler = handles.labeler;
%   lc = labeler.lblCore;
%   if ~( ~isempty(lc) && lc.supportsMultiView && lc.supportsCalibration )
%     handles.labeler.lerror('LabelerGUI:multiView',...
%       'Labeling mode must support multiple, calibrated views.');
%   end
%   vcd = labeler.viewCalibrationDataCurrent;
%   if isempty(vcd)
%     handles.labeler.lerror('LabelerGUI:vcd','No view calibration data set.');
%   end
%   % Hmm, is this weird, getting the vcd off Labeler not LabelCore. They
%   % should match however
%   assert(isa(vcd,'CalRig'));
%   crig = vcd;
% 
%   nview = labeler.nview;
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

labeler = handles.labeler;
if any(labeler.bgTrnIsRunningFromTrackerIndex()),
  warndlg('Cannot change training parameters while trackers are training.','Training in progress','modal');
  return;
end
handles.labeler.setStatus('Setting training parameters...');

[tPrm,do_update] = labeler.trackSetAutoParams();

sPrmNew = ParameterSetup(handles.figure,tPrm,'labelerObj',labeler); % modal

if isempty(sPrmNew)
  if do_update
    RC.saveprop('lastCPRAPTParams',sPrmNew);
    %cbkSaveNeeded(labeler,true,'Parameters changed');
    labeler.setDoesNeedSave(true,'Parameters changed') ;
  end
  % user canceled; none
else
  labeler.trackSetParams(sPrmNew);
  RC.saveprop('lastCPRAPTParams',sPrmNew);
  %cbkSaveNeeded(labeler,true,'Parameters changed');
  labeler.setDoesNeedSave(true,'Parameters changed') ;
end

handles.labeler.clearStatus();


function menu_track_settrackparams_Callback(hObject, eventdata, handles)

labeler = handles.labeler;
handles.labeler.setStatus('Setting tracking parameters...');

[tPrm] = labeler.trackGetTrackParams();

sPrmTrack = ParameterSetup(handles.figure,tPrm,'labelerObj',labeler); % modal

if ~isempty(sPrmTrack),
  sPrmNew = labeler.trackSetTrackParams(sPrmTrack);
  RC.saveprop('lastCPRAPTParams',sPrmNew);
  %cbkSaveNeeded(labeler,true,'Parameters changed');
  labeler.setDoesNeedSave(true, 'Parameters changed') ;
end

handles.labeler.clearStatus();


function menu_track_auto_params_update_Callback(hObject,eventdata,handles)

checked = get(hObject,'Checked');
set(hObject,'Checked',~checked);
handles.labeler.trackAutoSetParams = ~checked;
labeler = handles.labeler;
labeler.setDoesNeedSave(true, 'Auto compute training parameters changed') ;


function menu_track_use_all_labels_to_train_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
tObj = labeler.tracker;
if isempty(tObj)
  handles.labeler.lerror('LabelerGUI:tracker','No tracker for this project.');
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
% tObj = handles.labeler.tracker;
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
% handles.labeler.setStatus('Defining landmark matches...');
% labeler = handles.labeler;
% instructions = ['These part matches are used for data augmentation when training using deep learning. ' ...
%                 'To create more training data, we can flip the original images. This requires knowing ' ...
%                 'which parts on the left of the animal correspond to the same parts on the right side. ' ...
%                 'Use this GUI to select these pairings of parts. Each part can only belong to at most ' ...
%                 'one pair. Some parts will not be part of any pair, e.g. parts that go down the ' ...
%                 'mid-line of the animal should not have a mate.'];
% matches = defineLandmarkMatches(labeler,'edges',labeler.flipLandmarkMatches,'instructions',instructions);
% labeler.setFlipLandmarkMatches(matches);
% handles.labeler.clearStatus();

function menu_track_training_data_montage_Callback(hObject,eventdata,handles)
handles.labeler.setStatus('Plotting training examples...');
labeler = handles.labeler;
labeler.tracker.trainingDataMontage();
handles.labeler.clearStatus();

function menu_track_trainincremental_Callback(hObject, eventdata, handles)
handles.labeler.trainIncremental();

function menu_go_targets_summary_Callback(hObject, eventdata, handles)
if handles.labeler.maIsMA
  TrkInfoUI(handles.labeler);
else
  handles.controller.raiseTargetsTableFigure();
end

function menu_go_nav_prefs_Callback(hObject, eventdata, handles)
handles.labeler.navPrefsUI();

function menu_go_gt_frames_Callback(hObject, eventdata, handles)
handles.labeler.gtShowGTManager();

function menu_evaluate_crossvalidate_Callback(hObject, eventdata, handles)

labeler = handles.labeler;

tbl = labeler.labelGetMFTableLabeled;  
if labeler.maIsMA
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
  handles.labeler.lerror('LabelerGUI:xvalid','Number of folds must be a positive integer greater than 1.');
end

tbl.split = ceil(nfold*rand(n,1));

t = labeler.tracker;
t.trainsplit(tbl);


function menu_track_clear_tracking_results_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
% legacy behavior not sure why; maybe b/c the user is prob wanting to increase avail mem
%labeler.preProcInitData(); 
res = questdlg('Are you sure you want to clear tracking results?');
if ~strcmpi(res,'yes'),
  return;
end
handles.labeler.setStatus('Clearing tracking results...');
tObj = labeler.tracker;
tObj.clearTrackingResults();
handles.labeler.clearStatus();
%msgbox('Tracking results cleared.','Done');


function menu_track_cpr_storefull_dont_store_Callback(hObject, eventdata, handles)
tObj = handles.labeler.tracker;
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
tObj = handles.labeler.tracker;
tObj.storeFullTracking = StoreFullTrackingType.FINALITER;


function menu_track_cpr_storefull_store_all_iterations_Callback(...
  hObject, eventdata, handles)
tObj = handles.labeler.tracker;
tObj.storeFullTracking = StoreFullTrackingType.ALLITERS;


function menu_track_cpr_view_diagnostics_Callback(...
  hObject, eventdata, handles)
labeler = handles.labeler;

% Look for existing/open CPRVizTrackDiagsGUI
h = handles.controller.findSatelliteByTag_('figCPRVizTrackDiagsGUI') ;
if ~isempty(h) && isvalid(h) ,
  figure(h) ;
  return
end

lc = labeler.lblCore;
if ~isempty(lc) && ~lc.hideLabels
  warningNoTrace('LabelerGUI:hideLabels','Hiding labels.');
  lc.labelsHide();
end
hVizGUI = CPRVizTrackDiagsGUI(handles.labeler);
handles.controller.addSatellite(hVizGUI) ;
guidata(handles.figure,handles);


function menu_track_track_and_export_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI();
if ~tfok
  return
end
labeler.trackAndExport([],'rawtrkname',rawtrkname);


function menu_track_batch_track_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
tbobj = TrackBatchGUI(labeler);
tbobj.run();


function menu_track_all_movies_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
mIdx = labeler.allMovIdx();
toTrackIn = labeler.mIdx2TrackList(mIdx);
tbobj = TrackBatchGUI(labeler,'toTrack',toTrackIn);
% [toTrackOut] = tbobj.run();
tbobj.run();
% todo: import predictions


function menu_track_current_movie_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
mIdx = labeler.currMovIdx;
toTrackIn = labeler.mIdx2TrackList(mIdx);
mdobj = SpecifyMovieToTrackGUI(labeler,labeler.gdata.figure,toTrackIn);
[toTrackOut,dostore] = mdobj.run();
if ~dostore,
  return;
end
trackBatch('labeler',labeler,'toTrack',toTrackOut);


function menu_track_id_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
labeler.track_id = ~labeler.track_id;
set(handles.menu_track_id,'checked',labeler.track_id);


function menu_file_clear_imported_Callback(hObject,evtdata,handles)
labeler = handles.labeler;
labeler.labels2Clear();

function menu_file_export_all_movies_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
nMov = labeler.nmoviesGTaware;
if nMov==0
  handles.labeler.lerror('LabelerGUI:noMov','No movies in project.');
end
iMov = 1:nMov;
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI();
if ~tfok
  return;
end
labeler.trackExportResults(iMov,'rawtrkname',rawtrkname);

function menu_track_set_labels_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
tObj = labeler.tracker;
if labeler.gtIsGTMode
  handles.labeler.lerror('LabelerGUI:gt','Unsupported in GT mode.');
end

if ~isempty(tObj) && tObj.hasBeenTrained() && (~labeler.maIsMA)
  % single animal. Use prediction if available else use imported below
  [tfhaspred,xy,tfocc] = tObj.getTrackingResultsCurrFrm(); %#ok<ASGLU>
  itgt = labeler.currTarget;

  if ~tfhaspred(itgt)
    if (labeler.nTrx>1)
      msgbox('No predictions for current frame.');
      return;    
    else % for single animal use imported predictions if available
      iMov = labeler.currMovie;
      frm = labeler.currFrame;  
      [tfhaspred,xy] = labeler.labels2{iMov}.getPTrkFrame(frm);
      if ~tfhaspred
        msgbox('No predictions for current frame.');
        return;    
      end
    end
  else
    xy = xy(:,:,itgt); % "targets" treatment differs from below
  end

  disp(xy);
  
  % AL20161219: possibly dangerous, assignLabelCoords prob was intended
  % only as a util method for subclasses rather than public API. This may
  % not do the right thing for some concrete LabelCores.
%   labeler.lblCore.assignLabelCoords(xy);

  lpos2xy = reshape(xy,labeler.nLabelPoints,2);
  %assert(size(lpos2,4)==1); % "targets" treatment differs from above
  %lpos2xy = lpos2(:,:,frm);
  labeler.labelPosSet(lpos2xy);

  labeler.lblCore.newFrame(frm,frm,1);

else
  iMov = labeler.currMovie;
  frm = labeler.currFrame;
  if iMov==0
    handles.labeler.lerror('LabelerGUI:setLabels','No movie open.');
  end
  
  if labeler.maIsMA
    % We need to be smart about which to use. 
    % If only one of imported or prediction exist for the current frame then use whichever exists
    % If both exist for current frame, then don't do anything and error.
    
    useImported = true;
    usePred = true;
    % getting imported info old sytle. Doesn't work anymore
    
%     s = labeler.labels2{iMov};
%     itgtsImported = Labels.isLabeledF(s,frm);
%     ntgtsImported = numel(itgtsImported);
 
    % check if we can use imported
    imp_trk = labeler.labeledpos2trkViz;
    if isempty(imp_trk)
      useImported=false;
    elseif isnan(imp_trk.currTrklet)
      useImported=false;
    else
      s = labeler.labels2{iMov};
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
      s = labeler.labels2{iMov};
      iTgt = imp_trk.currTrklet;
      [~,xy,tfocc] = s.getPTrkFrame(frm,'collapse',true);      
    else
      iTgt = tObj.trkVizer.currTrklet;
      [~,xy,tfocc] = tObj.getTrackingResultsCurrFrm();
    end
    xy = xy(:,:,iTgt); % "targets" treatment differs from below
    occ = tfocc(:,iTgt);
    ntgts = labeler.labelNumLabeledTgts();
    labeler.setTargetMA(ntgts+1);
    labeler.labelPosSet(xy,occ);
    labeler.updateTrxTable();
    iTgt = labeler.currTarget;
    labeler.lblCore.tv.updateTrackResI(xy,occ,iTgt);

  else
    if labeler.nTrx>1
      handles.labeler.lerror('LabelerGUI:setLabels','Unsupported for multiple targets.');
    end
    %lpos2 = labeler.labeledpos2{iMov};
    %MK 20230728, labels2 now should always be TrkFile, but keeping other
    %logic around just in case. Needs work for multiview though.
    if isa(labeler.labels2{iMov} ,'TrkFile')      
      [~,p] = labeler.labels2{iMov}.getPTrkFrame(frm);
    else
      p = Labels.getLabelsF(labeler.labels2{iMov},frm,1);
    end
    lpos2xy = reshape(p,labeler.nLabelPoints,2);
    %assert(size(lpos2,4)==1); % "targets" treatment differs from above
    %lpos2xy = lpos2(:,:,frm);
    labeler.labelPosSet(lpos2xy);
    
    labeler.lblCore.newFrame(frm,frm,1);
  end
end

function menu_track_background_predict_start_Callback(hObject,eventdata,handles)
tObj = handles.labeler.tracker;
if tObj.asyncIsPrepared
  tObj.asyncStartBgRunner();
else
  if ~tObj.hasTrained
    errordlg('A tracker has not been trained.','Background Tracking');
    return;
  end
  tObj.asyncPrepare();
  tObj.asyncStartBgRunner();
end
  
function menu_track_background_predict_end_Callback(hObject,eventdata,handles)
tObj = handles.labeler.tracker;
if tObj.asyncIsPrepared
  tObj.asyncStopBgRunner();
else
  warndlg('Background worker is not running.','Background tracking');
end

function menu_track_background_predict_stats_Callback(hObject,eventdata,handles)
tObj = handles.labeler.tracker;
if tObj.asyncIsPrepared
  tObj.asyncComputeStats();
else
  warningNoTrace('LabelerGUI:bgTrack',...
    'No background tracking information available.','Background tracking');
end

function menu_evaluate_gtmode_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
controller = handles.controller ;

handles.labeler.setStatus('Switching between Labeling and Ground Truth Mode...');

gt = labeler.gtIsGTMode;
gtNew = ~gt;
labeler.gtSetGTMode(gtNew);
% hGTMgr = labeler.gdata.GTMgr;
if gtNew
  mmc = controller.movieManagerController_ ;
  mmc.setVisible(true);
  figure(mmc.hFig);
end
handles.labeler.clearStatus();

function menu_evaluate_gtloadsuggestions_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
LabelerGT.loadSuggestionsUI(labeler);

function menu_evaluate_gtsetsuggestions_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
LabelerGT.setSuggestionsToLabeledUI(labeler);

function menu_evaluate_gtcomputeperf_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
assert(labeler.gtIsGTMode);
labeler.gtComputeGTPerformance();

function menu_evaluate_gtcomputeperfimported_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
assert(labeler.gtIsGTMode);
labeler.gtComputeGTPerformance('useLabels2',true);

function menu_evaluate_gtexportresults_Callback(hObject,eventdata,handles)
labeler = handles.labeler;

tblRes = labeler.gtTblRes;
if isempty(tblRes)
  errordlg('No GT results are currently available.','Export GT Results');
  return;
end

%assert(labeler.gtIsGTMode);
fname = labeler.getDefaultFilenameExportGTResults();
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
  
function figure_CloseRequestFcn(hObject, eventdata, handles)
if isfield(handles, 'controller') 
  controller = handles.controller ;
  controller.quitRequested() ;
else
  delete(hObject) ;
end

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
labeler = handles.labeler;
oc = onCleanup(@()playCleanup(hObject,handles,iconStrPlay));
if ~handles.isPlaying
  handles.isPlaying = true;
  guidata(hObject,handles);
  hObject.CData = Icons.ims.stop;
  labeler.(playMeth);
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
labeler = handles.labeler;
if ~labeler.hasProject
  return;
end
if ~labeler.hasMovie
  msgbox('There is no movie open.');
  return;
end
tfok = true;

%% Cropping
% function handles = cropInitImRects(handles)
% deleteValidGraphicsHandles(handles.cropHRect);
% handles.cropHRect = ...
%   arrayfun(@(x)imrect(x,[nan nan nan nan]),handles.axes_all,'uni',0); %#ok<IMRECT> 
% handles.cropHRect = cat(1,handles.cropHRect{:}); % ML 2016a ish can't concat imrects in arrayfun output
% arrayfun(@(x)set(x,'Visible','off','PickableParts','none','UserData',true),...
%   handles.cropHRect); % userdata: see cropImRectSetPosnNoPosnCallback
% for ivw=1:numel(handles.axes_all)
%   posnCallback = @(zpos)cbkCropPosn(zpos,ivw,handles.figure);
%   handles.cropHRect(ivw).addNewPositionCallback(posnCallback);
% end

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
% Update handles.cropHRect from labeler.cropIsCropMode, labeler.currMovie and
% labeler.movieFilesAll*cropInfo
%
% rect props set:
% - position
% - visibility, pickableparts
%
% rect props NOT set:
% - resizeability. 

labeler = handles.labeler;
tfCropMode = labeler.cropIsCropMode;
[tfHasCrop,roi] = labeler.cropGetCropCurrMovie();
if tfCropMode && tfHasCrop
  nview = labeler.nview;
  imnc = labeler.movierawnc;
  imnr = labeler.movierawnr;
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


function tbAdjustCropSize_Callback(hObject, eventdata, handles)
cropUpdateCropAdjustingCropSize(handles);
tb = handles.tbAdjustCropSize;
if tb.Value==tb.Min
  % user clicked "Done Adjusting"
  warningNoTrace('All movies in a given view must share the same crop size. The sizes of all crops have been updated as necessary.'); 
elseif tb.Value==tb.Max
  % user clicked "Adjust Crop Size"
  labeler = handles.labeler;
  if ~labeler.cropProjHasCrops
    labeler.cropInitCropsAllMovies;
    fprintf(1,'Default crop initialized for all movies.\n');
    cropUpdateCropHRects(handles);
  end
end
function pbClearAllCrops_Callback(hObject, eventdata, handles)
handles.labeler.cropClearAllCrops();


% --------------------------------------------------------------------
function menu_file_export_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_export_labels2_trk_curr_mov (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labeler = handles.labeler;
iMov = labeler.currMovie;
if iMov==0
  handles.labeler.lerror('LabelerGUI:noMov','No movie currently set.');
end
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI();
if ~tfok
  return;
end
labeler.trackExportResults(iMov,'rawtrkname',rawtrkname);


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
labeler = handles.labeler;
cbkApply = @(varargin)hlpApplyCosmetics(labeler,varargin{:});
LandmarkColors(labeler,cbkApply);
% AL 20220217: changes now applied immediately
% if ischange
%   cbkApply(savedres.colorSpecs,savedres.markerSpecs,savedres.skelSpecs);
% end

function hlpApplyCosmetics(labeler,colorSpecs,mrkrSpecs,skelSpecs)
labeler.updateLandmarkColors(colorSpecs);
labeler.updateLandmarkCosmetics(mrkrSpecs);
labeler.updateSkeletonCosmetics(skelSpecs);

function menu_track_edit_skeleton_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
landmark_specs('labeler',labeler);
%hasSkeleton = ~isempty(labeler.skeletonEdges) ;
%labeler.setShowSkeleton(hasSkeleton) ;

function menu_track_viz_dataaug_Callback(hObject,evtdata,handles)
labeler = handles.labeler;
labeler.retrainAugOnly() ;

function menu_view_showhide_skeleton_Callback(hObject, eventdata, handles)
if strcmpi(get(hObject,'Checked'),'off'),
  hObject.Checked = 'on';
  handles.labeler.setShowSkeleton(true);
else
  hObject.Checked = 'off';
  handles.labeler.setShowSkeleton(false);
end

function menu_view_showhide_maroi_Callback(hObject, eventdata, handles)
if strcmpi(get(hObject,'Checked'),'off'),
  handles.labeler.setShowMaRoi(true);
else
  handles.labeler.setShowMaRoi(false);
end

function menu_view_showhide_maroiaux_Callback(hObject, eventdata, handles)
tf = strcmpi(get(hObject,'Checked'),'off');
handles.labeler.setShowMaRoiAux(tf);

% --- Executes on selection change in popupmenu_prevmode.
function popupmenu_prevmode_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_prevmode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
mode = contents{get(hObject,'Value')};
if strcmpi(mode,'Reference'),
  handles.labeler.setPrevAxesMode(PrevAxesMode.FROZEN,handles.labeler.prevAxesModeInfo);
else
  handles.labeler.setPrevAxesMode(PrevAxesMode.LASTSEEN);
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

handles.labeler.setPrevAxesMode(PrevAxesMode.FROZEN);

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
  'Adam Taylor, Alice Robie, Felipe Rodriguez,'
  'Stephen Huston, Roian Egnor, Austin Edwards,'
  'Caroline Maloney, and Kristin Branson'};

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

labeler = handles.labeler;
labeler.cropSetCropMode(false);


% --------------------------------------------------------------------
function menu_view_occluded_points_box_Callback(hObject, eventdata, handles)
% hObject    handle to menu_view_occluded_points_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labeler = handles.labeler;
labeler.setShowOccludedBox(~labeler.showOccludedBox);
if labeler.showOccludedBox,
  labeler.lblCore.showOcc();
else
  labeler.lblCore.hideOcc();
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



function register_labeler(figure, labeler)

handles = guidata(figure) ;

handles.labeler = labeler ;

handles.labelTLInfo = InfoTimeline(labeler,handles.axes_timeline_manual,...
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
% % listeners{end+1,1} = addlistener(labeler,'didSetProjname',@cbkProjNameChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetCurrTarget',@cbkCurrTargetChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetLastLabelChangeTS',@cbkLastLabelChangeTS);
% % listeners{end+1,1} = addlistener(labeler,'didSetTrackParams',@cbkParameterChange);
% listeners{end+1,1} = addlistener(labeler,'didSetLabelMode',@cbkLabelModeChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLabels2Hide',@cbkLabels2HideChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLabels2ShowCurrTargetOnly',@cbkLabels2ShowCurrTargetOnlyChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetProjFSInfo',@cbkProjFSInfoChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowTrx',@cbkShowTrxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowOccludedBox',@cbkShowOccludedBoxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowTrxCurrTargetOnly',@cbkShowTrxCurrTargetOnlyChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetTrackersAll',@cbkTrackersAllChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetCurrTracker',@cbkCurrTrackerChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackModeIdx',@cbkTrackModeIdxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesSmall',@cbkTrackerNFramesChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesLarge',@cbkTrackerNFramesChanged);    
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesNear',@cbkTrackerNFramesChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieCenterOnTarget',@cbkMovieCenterOnTargetChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieRotateTargetUp',@cbkMovieRotateTargetUpChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieForceGrayscale',@cbkMovieForceGrayscaleChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetMovieInvert',@cbkMovieInvertChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieViewBGsubbed',@cbkMovieViewBGsubbedChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLblCore',@(src,evt)(handles.controller.didSetLblCore()));
% listeners{end+1,1} = addlistener(labeler,'gtIsGTModeChanged',@cbkGtIsGTModeChanged);
% listeners{end+1,1} = addlistener(labeler,'cropIsCropModeChanged',@cbkCropIsCropModeChanged);
% listeners{end+1,1} = addlistener(labeler,'cropUpdateCropGUITools',@cbkUpdateCropGUITools);
% listeners{end+1,1} = addlistener(labeler,'cropCropsChanged',@cbkCropCropsChanged);
% %listeners{end+1,1} = addlistener(labeler,'newProject',@cbkNewProject);
% listeners{end+1,1} = addlistener(labeler,'newMovie',@cbkNewMovie);
% %listeners{end+1,1} = addlistener(labeler,'projLoaded',@cbkProjLoaded);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'selectOn','PostSet',@cbklabelTLInfoSelectOn);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_tracker','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_allframes','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'proptypes','PostSet',@cbklabelTLInfoPropTypesUpdated);
% %listeners{end+1,1} = addlistener(labeler,'startAddMovie',@cbkAddMovie);
% %listeners{end+1,1} = addlistener(labeler,'finishAddMovie',@cbkAddMovie);
% %listeners{end+1,1} = addlistener(labeler,'startSetMovie',@cbkSetMovie);
% listeners{end+1,1} = addlistener(labeler,'dataImported',@cbkDataImported);
% listeners{end+1,1} = addlistener(labeler,'didSetShowSkeleton',@cbkShowSkeletonChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowMaRoi',@cbkShowMaRoiChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowMaRoiAux',@cbkShowMaRoiAuxChanged);

handles.listeners = listeners;

% Make the debug menu visible, if called for
handles.menu_debug.Visible = onIff(labeler.isInDebugMode) ;

% Stash the guidata
guidata(figure, handles) ;



function register_controller(figure, controller)
handles = guidata(figure) ;
handles.controller = controller ;
guidata(figure, handles) ;
