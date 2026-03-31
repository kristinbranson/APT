function layoutMainFigureBang(hFig)
% Set the positions of all main-figure controls based on the current figure size.
% All positions are in logical pixels.
%
% When the figure is smaller than the minimum layout size, controls are laid
% out at the minimum size but shifted so they stay anchored to the top-left
% corner of the figure.

handles = guihandles(hFig) ;
figPos = hFig.Position ;
figWActual = figPos(3) ;
figHActual = figPos(4) ;
minLayoutWidth = 1534 ;
minLayoutHeight = 1062 ;
figW = max(figWActual, minLayoutWidth) ;
figH = max(figHActual, minLayoutHeight) ;

% Vertical offset so that controls stay anchored to the top-left corner
% when the figure is shorter than the minimum layout height.
% Zero when the figure is at or above minimum size.
dy = figHActual - figH ;

% -- Derived sizes for the main panel --
panelW = figW - 460 ;   % left edge at 450, right margin 10
panelH = figH - 204 ;   % bottom at 184, top margin 20

%% --- Direct children of main_figure ---

% Main panel (resizes freely)
handles.uipanel_curr.Position = [450, 184+dy, panelW, panelH] ;

% Top-left anchored (y shifts with figure height)
handles.uipanel_prev.Position = [19, figH-370+dy, 415, 350] ;
handles.uipanel_targets.Position = [16, figH-575+dy, 195, 192] ;
handles.uipanel_targetzoom.Position = [16, figH-635+dy, 200, 60] ;
% uipanel_frames x depends on whether targets panel is visible
if strcmp(handles.uipanel_targets.Visible, 'on')
  framesX = 16 + 195 ;
else
  framesX = 16 ;
end
handles.uipanel_frames.Position = [framesX, figH-635+dy, 220, 252] ;

% Bottom-right anchored (x shifts with figure width)
handles.pumTimelineProp.Position = [figW-171, 27+dy, 157, 30] ;
handles.pumTimelinePropType.Position = [figW-301, 27+dy, 124, 30] ;
handles.tx_timeline_islabeled.Position = [figW-84, 154+dy, 72, 21] ;

% Stretch width (width tracks main panel width)
handles.axes_timeline_manual.Position = [450, 62+dy, panelW, 87] ;
handles.axes_timeline_islabeled.Position = [450, 152+dy, panelW, 24] ;
handles.pnlStatus.Position = [1, -2+dy, figW, 28] ;

% Bottom-left anchored (fixed positions)
handles.tbAccept.Position = [228, 154+dy, 197, 49] ;
handles.pbClear.Position = [22, 154+dy, 197, 49] ;
handles.pbTrain.Position = [22, 98+dy, 197, 49] ;
handles.pbTrack.Position = [229, 98+dy, 197, 49] ;
handles.pumTrack.Position = [167, 52+dy, 258, 31] ;
handles.tbTLSelectMode.Position = [450, 35+dy, 48, 22] ;
handles.pbClearSelection.Position = [500, 35+dy, 48, 22] ;
handles.txLblCoreAux.Position = [20, figH-682+dy, 220, 42] ;
handles.txUnsavedChanges.Position = [260, 402+dy, 174, 20] ;
handles.txGTMode.Position = [303, 374+dy, 132, 23] ;
handles.txCropMode.Position = [304, 350+dy, 132, 23] ;
handles.uipanel_cropcontrols.Position = [34, 275+dy, 387, 73] ;
handles.text_framestotrack.Position = [19, 56+dy, 136, 23] ;
handles.text_framestotrackinfo.Position = [23, 34+dy, 406, 19] ;
handles.text_trackerinfo.Position = [19, figH-774+dy, 422, 88] ;

%% --- Children of uipanel_curr ---

% Resize freely
handles.axes_curr.Position = [27, 61, panelW-68, panelH-102] ;

% Top of panel + stretch width
handles.txMoviename.Position = [183, panelH-23, panelW-188, 19] ;

% Bottom of panel + stretch width
handles.slider_frame.Position = [164, 19, panelW-190, 24] ;

% Bottom-left of panel (fixed within panel)
handles.axes_occ.Position = [14, 62, 196, 54] ;
handles.text_occludedpoints.Position = [14, 93, 196, 23] ;
handles.edit_frame.Position = [91, 20, 68, 24] ;
handles.pbPlay.Position = [9, 17, 30, 30] ;
handles.pbPlaySeg.Position = [43, 17, 21, 30] ;
handles.pbPlaySegRev.Position = [66, 17, 21, 30] ;

%% --- Children of pnlStatus ---
handles.txStatus.Position = [5, 9, 1107, 19] ;
handles.txBGTrain.Position = [figW-135, 4, 132, 23] ;

end  % function
