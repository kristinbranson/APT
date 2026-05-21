function setLabelerFigureTooltipsBang(mainFigure)
% Add tooltips to widgets and menu items in the labeler main figure.

handles = guihandles(mainFigure) ;

set(handles.pbClear, 'TooltipString', 'Clear labels in current frame') ;
set(handles.pbTrain, 'TooltipString', 'Train the part tracker') ;
set(handles.pbTrack, 'TooltipString', 'Track current selection of frames and targets') ;
set(handles.pumTrack, 'TooltipString', 'Which set of frames and targets to track when "Track" button pressed') ;
set(handles.tbAccept, 'TooltipString', 'Accept and store labels for current frame (all parts must be labeled)') ;

set(handles.pbPlay, 'TooltipString', 'Play movie') ;
set(handles.pbPlaySeg, 'TooltipString', 'Jump back a bit, then play to current frame') ;
set(handles.pbPlaySegRev, 'TooltipString', 'Jump forward a bit, then play in reverse to current frame') ;

set(handles.pbClearSelection, 'TooltipString', 'Clear frames selected in timeline') ;
set(handles.tbTLSelectMode, 'TooltipString', 'Select Frames in the timeline') ;

set(handles.pbResetZoom, 'TooltipString', 'Zoom out to show whole video frame') ;
set(handles.pbSetZoom, 'TooltipString', 'Store current zoom for recalling') ;
set(handles.pbRecallZoom, 'TooltipString', 'Recall stored zoom level') ;

set(handles.tbAdjustCropSize, 'TooltipString', 'Toggle on/off whether crop size(s) can be adjusted') ;
set(handles.pbClearAllCrops, 'TooltipString', 'Clear cropping information for all videos') ;

% Menu tooltips.  The uimenu Tooltip property was added in R2020b, but on
% classic figures created with figure() (which is what APT uses) setting
% it errors with "Functionality not supported with figures created with
% the figure function" until R2025a (MATLAB 25.1), so skip this section
% on older releases.
if verLessThan('matlab', '25.1')  %#ok<VERLESSMATLAB>
  return
end

% file menu
setMenuTooltipIfPresent_(handles, 'menu_file_managemovies', ...
  'Open movie manager dialog to switch to a different movie or add or remove movies from the project') ;
setMenuTooltipIfPresent_(handles, 'menu_file_import_labels_trk_curr_mov', ...
  'Import predictions from .trk file as LABELS for current movie') ;
setMenuTooltipIfPresent_(handles, 'menu_file_import_tracking_results', ...
  'Import tracking results from .trk file for current movie') ;
setMenuTooltipIfPresent_(handles, 'menu_file_export_labels_trks', ...
  'Export LABELS to .trk files for all movies') ;
setMenuTooltipIfPresent_(handles, 'menu_file_crop_mode', ...
  'Edit cropped regions of interest') ;

% view menu
setMenuTooltipIfPresent_(handles, 'menu_view_converttograyscale', ...
  'Display color images in grayscale') ;
setMenuTooltipIfPresent_(handles, 'menu_view_adjustbrightness', ...
  'Change displayed image brightness and contrast') ;
setMenuTooltipIfPresent_(handles, 'menu_view_gammacorrect', ...
  'Change displayed image gamma correction') ;

setMenuTooltipIfPresent_(handles, 'menu_view_flip_flipud', ...
  'Vertically flip both the movie and labels in the display') ;
setMenuTooltipIfPresent_(handles, 'menu_view_flip_fliplr', ...
  'Horizontally flip both the movie and labels in the display') ;

setMenuTooltipIfPresent_(handles, 'menu_view_reset_views', ...
  'Reset zoom so that entire video frames are displayed') ;

setMenuTooltipIfPresent_(handles, 'menu_view_trajectories_centervideoontarget', ...
  'When checked, axes will always be centered over current target') ;

% label menu
setMenuTooltipIfPresent_(handles, 'menu_label_sequential_mode', ...
  'Sequential labeling: Click landmark locations in order') ;
setMenuTooltipIfPresent_(handles, 'menu_label_template_mode', ...
  'Template labeling: Move around initial landmark locations') ;
setMenuTooltipIfPresent_(handles, 'menu_label_multiview_mode', ...
  'Multi-view calibrated mode: Show epipolar line after labeling in one view') ;

setMenuTooltipIfPresent_(handles, 'menu_label_overlay_montage', ...
  'Plot all labels on one frame to see label distribution') ;
setMenuTooltipIfPresent_(handles, 'menu_label_overlay_montage_trx_centered', ...
  'Plot all trajectory-aligned labels on one frame to see label distribution') ;

% go menu
setMenuTooltipIfPresent_(handles, 'menu_go_targets_summary', ...
  'Switch to labeling a different target in a different video') ;

% track menu
setMenuTooltipIfPresent_(handles, 'menu_track_tracking_algorithm', ...
  'Algorithm used to train tracker') ;
setMenuTooltipIfPresent_(handles, 'menu_track_clear_tracking_results', ...
  'Remove tracking results for the current tracker, for all movies') ;

setMenuTooltipIfPresent_(handles, 'menu_label_set_labels', ...
  'Set labels to predictions for current frame') ;

end  % function

function setMenuTooltipIfPresent_(handles, fieldName, tooltip)
% Set the Tooltip on handles.(fieldName) if that field exists.
if isfield(handles, fieldName)
  set(handles.(fieldName), 'Tooltip', tooltip) ;
end
end  % function
