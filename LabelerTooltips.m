function handles=LabelerTooltips(handles)
% LabelerTooltips: function to add tooltips to objects in the labeler
%
% form: handles=labeler_tooltips(handles)



set(handles.pbTrain,'TooltipString','Train a part classifier');
set(handles.pbTrack,'TooltipString','Track points using current classifier');
set(handles.pumTrack,'TooltipString','Tracking options');
set(handles.tbAccept,'TooltipString','Label Accept button (all points must be labeled to accept)');




















