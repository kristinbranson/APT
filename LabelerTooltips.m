function handles=LabelerTooltips(handles)
% LabelerTooltips: function to add tooltips to objects in the labeler
%
% form: handles=labeler_tooltips(handles)



set(handles.pbClear,'TooltipString','Clear labeled points in current frame');
set(handles.pbTrain,'TooltipString','Train a part classifier');
set(handles.pbTrack,'TooltipString','Track points using current classifier');
set(handles.pumTrack,'TooltipString','Tracking options');
set(handles.tbAccept,'TooltipString','Label Accept button (all points must be labeled to accept)');

set(handles.pbPlay,'TooltipString','Play movie');
set(handles.pbPlaySeg,'TooltipString','Play frames around current frame');

set(handles.pbClearSelection,'TooltipString','Clear selection');
set(handles.tbTLSelectMode,'TooltipString','Select Frames');

set(handles.pbResetZoom,'TooltipString','Zoom out and center');
set(handles.pbSetZoom,'TooltipString','Set zoom to recall');
set(handles.pbRecallZoom,'TooltipString','Recall zoom level');

% set(handles.tbAdjustCropSize,'TooltipString','');
% set(handles.pbClearAllCrops,'TooltipString','');

















