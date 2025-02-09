function dispatchMainFigureCallback(callbackName, varargin)
feval(callbackName, varargin{:}) ;



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

handles.listeners = listeners;

% Make the debug menu visible, if called for
handles.menu_debug.Visible = onIff(labeler.isInDebugMode) ;

% Stash the guidata
guidata(figure, handles) ;



function register_controller(main_figure, controller)
handles = guidata(main_figure) ;
handles.controller = controller ;
set(handles.tblTrx, 'CellSelectionCallback', @(s,e)(controller.controlActuated('tblTrx', s, e))) ;
set(handles.tblFrames, 'CellSelectionCallback',@(s,e)(controller.controlActuated('tblFrames', s, e))) ;

hZ = zoom(main_figure);  % hZ is a "zoom object"
hZ.ActionPostCallback = @(s,e)(obj.cbkPostZoom(s,e)) ;
hP = pan(main_figure);  % hP is a "pan object"
hP.ActionPostCallback = @(s,e)(obj.cbkPostPan(s,e)) ;

set(main_figure, 'CloseRequestFcn', @(s,e)(controller.figure_CloseRequestFcn())) ;

guidata(main_figure, handles) ;
