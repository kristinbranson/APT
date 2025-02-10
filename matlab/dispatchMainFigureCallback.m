function dispatchMainFigureCallback(callbackName, varargin)
  feval(callbackName, varargin{:}) ;
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
  listeners = cell(0,1);
  listeners{end+1,1} = addlistener(handles.slider_frame,'ContinuousValueChange',@slider_frame_Callback);
  listeners{end+1,1} = addlistener(handles.sldZoom,'ContinuousValueChange',@sldZoom_Callback);
  handles.listeners = listeners;
  % Make the debug menu visible, if called for
  handles.menu_debug.Visible = onIff(labeler.isInDebugMode) ;
  % Stash the guidata
  guidata(figure, handles) ;
end


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

  handles.menu_track_reset_current_tracker.Callback = ...
    @(s,e)(controller.controlActuated('menu_track_reset_current_tracker', s, e)) ;
  handles.menu_track_delete_current_tracker.Callback = ...
    @(s,e)(controller.controlActuated('menu_track_delete_current_tracker', s, e)) ;
  handles.menu_track_delete_old_trackers.Callback = ...
    @(s,e)(controller.controlActuated('menu_track_delete_old_trackers', s, e)) ;
  
  guidata(main_figure, handles) ;
end
