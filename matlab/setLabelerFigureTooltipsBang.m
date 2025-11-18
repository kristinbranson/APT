function setLabelerFigureTooltipsBang(mainFigure)
% LabelerTooltips: function to add tooltips to objects in the labeler
%
% form: handles=labeler_tooltips(handles)

handles = guihandles(mainFigure) ;

set(handles.pbClear,'TooltipString','Clear labels in current frame');
set(handles.pbTrain,'TooltipString','Train the part tracker');
set(handles.pbTrack,'TooltipString','Track current selection of frames and targets');
set(handles.pumTrack,'TooltipString','Which set of frames and targets to track when "Track" button pressed');
set(handles.tbAccept,'TooltipString','Accept and store labels for current frame (all parts must be labeled)');

set(handles.pbPlay,'TooltipString','Play movie');
set(handles.pbPlaySeg,'TooltipString','Jump back a bit, then play to current frame');
set(handles.pbPlaySegRev,'TooltipString','Jump forward a bit, then play in reverse to current frame');

set(handles.pbClearSelection,'TooltipString','Clear frames selected in timeline');
set(handles.tbTLSelectMode,'TooltipString','Select Frames in the timeline');

set(handles.pbResetZoom,'TooltipString','Zoom out to show whole video frame');
set(handles.pbSetZoom,'TooltipString','Store current zoom for recalling');
set(handles.pbRecallZoom,'TooltipString','Recall stored zoom level');

set(handles.tbAdjustCropSize,'TooltipString','Toggle on/off whether crop size(s) can be adjusted');
set(handles.pbClearAllCrops,'TooltipString','Clear cropping information for all videos');

% Below here, there's a bunch of Java stuff that seems to cause issues in
% Matlab 2024b.  So we'll skip that stuff for now, possibly return to it
% later.  -- ALT, 2025-01-22
return

dotooltips = lcl('jvm') && lcl('awt') && lcl('swing');

if ~dotooltips,
  return;
end

try
oldvisible = get(handles.figure,'Visible');
if strcmp(oldvisible,'off'),
  set(handles.figure,'Visible','on');
  pause(1);
end
jobjs = findjobj_modern(handles.figure,'class','MenuPeer');
jobjnames = cell(size(jobjs));
for i = 1:numel(jobjs),
  jobjnames{i} = get(jobjs(i),'Name');
end

% file menu
if isfield(handles,'menu_file_managemovies'),
  SetTooltip(handles.menu_file_managemovies,'Open movie manager dialog to switch to a different movie or add or remove movies from the project',jobjs,jobjnames);
end
% if isfield(handles,'menu_file_import_labels_trk_curr_mov'),
%   SetTooltip(handles.menu_file_import_labels_trk_curr_mov,'Import predictions from .trk file as LABELS for current movie',jobjs,jobjnames);
% end
% if isfield(handles,'menu_file_import_labels2_trk_curr_mov'),
%   SetTooltip(handles.menu_file_import_labels2_trk_curr_mov,'Import predictions from .trk file for current movie',jobjs,jobjnames);
% end
if isfield(handles,'menu_file_export_labels2_trk_curr_mov'),
  SetTooltip(handles.menu_file_export_labels2_trk_curr_mov,'Export predictions to .trk file for current movie',jobjs,jobjnames);
end
if isfield(handles,'menu_file_export_labels_trks'),
  SetTooltip(handles.menu_file_export_labels_trks,'Export LABELS to .trk files for all movies',jobjs,jobjnames);
end
if isfield(handles,'menu_file_export_labels_trks'),
  SetTooltip(handles.menu_file_export_labels_trks,'Export LABELS to .trk files for all movies',jobjs,jobjnames);
end
if isfield(handles,'menu_file_crop_mode'),
  SetTooltip(handles.menu_file_crop_mode,'Edit cropped regions of interest',jobjs,jobjnames);
end

% view menu
if isfield(handles,'menu_view_converttograyscale'),
  SetTooltip(handles.menu_view_converttograyscale,'Display color images in grayscale',jobjs,jobjnames);
end
if isfield(handles,'menu_view_adjustbrightness'),
  SetTooltip(handles.menu_view_adjustbrightness,'Change displayed image brightness and contrast',jobjs,jobjnames);
end
if isfield(handles,'menu_view_gammacorrect'),
  SetTooltip(handles.menu_view_gammacorrect,'Change displayed image gamma correction',jobjs,jobjnames);
end

if isfield(handles,'menu_view_flip_flipud'),
  SetTooltip(handles.menu_view_flip_flipud,'Vertically flip both the movie and labels in the display',jobjs,jobjnames);
end
if isfield(handles,'menu_view_flip_fliplr'),
  SetTooltip(handles.menu_view_flip_fliplr,'Horizontally flip both the movie and labels in the display',jobjs,jobjnames);
end
if isfield(handles,'menu_view_flip_flipud_movie_only'),
  SetTooltip(handles.menu_view_flip_flipud_movie_only,'Vertically flip only the movie in the display',jobjs,jobjnames);
end

if isfield(handles,'menu_view_reset_views'),
  SetTooltip(handles.menu_view_reset_views,'Reset zoom so that entire video frames are displayed',jobjs,jobjnames);
end

if isfield(handles,'menu_view_trajectories_centervideoontarget'),
  SetTooltip(handles.menu_view_trajectories_centervideoontarget,'When checked, axes will always be centered over current target',jobjs,jobjnames);
end
if isfield(handles,'menu_view_trajectories_centervideoontarget'),
  SetTooltip(handles.menu_view_trajectories_centervideoontarget,'When checked, axes will be rotated so that current target is facing up',jobjs,jobjnames);
end

if isfield(handles,'menu_setup_sequential_mode'),
  SetTooltip(handles.menu_setup_sequential_mode,'Sequential labeling: Click landmark locations in order',jobjs,jobjnames);
end
if isfield(handles,'menu_setup_template_mode'),
  SetTooltip(handles.menu_setup_template_mode,'Template labeling: Move around initial landmark locations',jobjs,jobjnames);
end
if isfield(handles,'menu_setup_highthroughput_mode'),
  SetTooltip(handles.menu_setup_highthroughput_mode,'High-throughput labeling: Label one landmark at a time in a series of frames',jobjs,jobjnames);
end
if isfield(handles,'menu_setup_multiview_calibrated_mode_2'),
  SetTooltip(handles.menu_setup_multiview_calibrated_mode_2,'Multi-view calibrated mode: Show epipolar line after labeling in one view',jobjs,jobjnames);
end

if isfield(handles,'menu_setup_label_overlay_montage'),
  SetTooltip(handles.menu_setup_label_overlay_montage,'Plot all labels on one frame to see label distribution',jobjs,jobjnames);
end
if isfield(handles,'menu_setup_label_overlay_montage_trx_centered'),
  SetTooltip(handles.menu_setup_label_overlay_montage_trx_centered,'Plot all trajectory-aligned labels on one frame to see label distribution',jobjs,jobjnames);
end

% go menu
if isfield(handles,'menu_go_targets_summary'),
  SetTooltip(handles.menu_go_targets_summary,'Switch to labeling a different target in a different video',jobjs,jobjnames);
end

% track menu
if isfield(handles,'menu_track_tracking_algorithm'),
  SetTooltip(handles.menu_track_tracking_algorithm,'Algorithm used to train tracker',jobjs,jobjnames);
end
% if isfield(handles,'menu_track_training_data_montage'),
%   SetTooltip(handles.menu_track_training_data_montage,'Plot sampled training examples',jobjs,jobjnames);
% end

% if isfield(handles,'menu_track_track_and_export'),
%   SetTooltip(handles.menu_track_track_and_export,'Track current selection of videos, targets, and frames, and export results to .trk files.',jobjs,jobjnames);
% end
if isfield(handles,'menu_track_clear_tracking_results'),
  SetTooltip(handles.menu_track_clear_tracking_results,'Remove all tracking results from the current project.',jobjs,jobjnames);
end

if isfield(handles,'menu_track_set_labels'),
  SetTooltip(handles.menu_track_set_labels,'Set labels to predictions for current frame',jobjs,jobjnames);
end

% h = findjobj_modern(handles.pbClear);
% h.doClick();
i = find(strcmp(jobjnames,'menu_file'),1);
jobjs(i).doClick();
drawnow;
jobjs(i).doClick();
drawnow;
if strcmp(oldvisible,'off'),
  set(handles.figure,'Visible','off');
end
catch ME,
  warning('Error setting menu tooltips: %s',getReport(ME));
end

function success = lcl(level)
success = false;
%fprintf('%s\n',level);
try
  error(javachk(level));
  success = true;
catch ME
  fprintf('err caught: %s\n',ME.message);
end
