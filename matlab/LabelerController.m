classdef LabelerController < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    mainFigure_  % the GH to the main figure
    listeners_
    satellites_ = gobjects(1,0)  % handles of dialogs, figures, etc that will get deleted when this object is deleted
    waitbarFigure_ = gobjects(1,0)  % a GH to a waitbar() figure, or empty
    %waitbarListeners_ = event.listener.empty(1,0)
    trackingMonitorVisualizer_
    trainingMonitorVisualizer_
    movieManagerController_
    % Things related to resizing
    pxTxUnsavedChangesWidth_  % We will record the width (in pixels) of txUnsavedChanges here, so we can keep it fixed
    %pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_
    %pumTrackInitFontSize_
    %pumTrackInitHeight_
    isPlaying_ = false  % whether a video is currently playing or not
    pumTrackFullStrings_ = []
  end

  properties  % private/protected by convention
    tvTrx_  % scalar TrackingVisualizerTrx
    isInYodaMode_ = false  
      % Set to true to allow control actuation to happen *ouside* or a try/catch
      % block.  Useful for debugging.  "Do, or do not.  There is no try." --Yoda
  end

  methods
    function obj = LabelerController(varargin)
      % Process args that have to be dealt with before creating the Labeler
      [isInDebugMode, isInAwsDebugMode, isInYodaMode] = ...
        myparse_nocheck(varargin, ...
                        'isInDebugMode',false, ...
                        'isInAwsDebugMode',false, ...
                        'isInYodaMode', false) ;

      % Create the labeler, tell it there will be a GUI attached
      labeler = Labeler('isgui', true, 'isInDebugMode', isInDebugMode,  'isInAwsDebugMode', isInAwsDebugMode) ;  

      % Set up the main instance variables
      obj.labeler_ = labeler ;
      % obj.mainFigure_ = LabelerGUI(labeler, obj) ;
      mainFigure = createLabelerMainFigure() ;
      obj.mainFigure_ = mainFigure ;
      obj.labeler_.registerController(obj) ;  % hack
      obj.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      obj.isInYodaMode_ = isInYodaMode ;  
        % If in yoda mode, we don't wrap GUI-event function calls in a try..catch.
        % Useful for debugging.
        
      % Set up this resize thing
      obj.initializeResizeInfo_() ;
      mainFigure.SizeChangedFcn = @(src,evt)(obj.resize()) ;
      obj.resize() ;

      % Update the controls enablement  
      obj.enableControls_('noproject') ;
      
      % Update the status
      obj.updateStatus([],[]) ;

      % % Populate the callbacks of the controls in the main figure---someday
      % apt.populate_callbacks_bang(mainFigure, obj) ;

      % Create the waitbar figure, which we re-use  
      obj.waitbarFigure_ = waitbar(0, '', ...
                                   'Visible', 'off', ...
                                   'CreateCancelBtn', @(source,event)(obj.didCancelWaitbar())) ;
      obj.waitbarFigure_.CloseRequestFcn = @(source,event)(nop()) ;
        
      % Add some controls to the UI that we can set up before there is a project
      obj.initialize_menu_track_backend_config_() ;
      
      % Get the handles out of the figure
      handles = guidata(mainFigure) ;

      % Set up some stuff between the labeler and the mainFigure.  Some of this
      % stuff should probably go elsewhere...
      handles.labeler = labeler ;
      handles.labelTLInfo = InfoTimeline(labeler,handles.axes_timeline_manual,...
                                         handles.axes_timeline_islabeled);
      set(handles.pumInfo,...
          'String',handles.labelTLInfo.getPropsDisp(),...
          'Value',handles.labelTLInfo.curprop);
      set(handles.pumInfo_labels,...
          'String',handles.labelTLInfo.getPropTypesDisp(),...
          'Value',handles.labelTLInfo.curproptype);

      % Make the debug menu visible, if called for
      handles.menu_debug.Visible = onIff(labeler.isInDebugMode) ;      

      % Set up some custom callbacks
      %handles.controller = obj ;
      set(handles.tblTrx, 'CellSelectionCallback', @(s,e)(obj.controlActuated('tblTrx', s, e))) ;
      set(handles.tblFrames, 'CellSelectionCallback',@(s,e)(obj.controlActuated('tblFrames', s, e))) ;
      hZ = zoom(mainFigure);  % hZ is a "zoom object"
      hZ.ActionPostCallback = @(s,e)(obj.cbkPostZoom(s,e)) ;
      hP = pan(mainFigure);  % hP is a "pan object"
      hP.ActionPostCallback = @(s,e)(obj.cbkPostPan(s,e)) ;
      set(mainFigure, 'CloseRequestFcn', @(s,e)(obj.figure_CloseRequestFcn())) ;    
      handles.menu_track_reset_current_tracker.Callback = ...
        @(s,e)(obj.controlActuated('menu_track_reset_current_tracker', s, e)) ;
      handles.menu_track_delete_current_tracker.Callback = ...
        @(s,e)(obj.controlActuated('menu_track_delete_current_tracker', s, e)) ;
      handles.menu_track_delete_old_trackers.Callback = ...
        @(s,e)(obj.controlActuated('menu_track_delete_old_trackers', s, e)) ;
      
      % Set up the figure callbacks to call obj, using the tag to determine the
      % method name.
      visit_children(main_figure, @set_standard_callback_if_none_bang, obj) ;

      % Add the listeners
      obj.listeners_ = event.listener.empty(1,0) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateDoesNeedSave', @(source,event)(obj.updateDoesNeedSave(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateStatus', @(source,event)(obj.updateStatus(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'didSetTrx', @(source,event)(obj.didSetTrx(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateTrxSetShowTrue', @(source,event)(obj.updateTrxSetShowTrue(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateTrxSetShowFalse', @(source,event)(obj.updateTrxSetShowFalse(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'didHopefullySpawnTrackingForGT', @(source,event)(obj.showDialogAfterHopefullySpawningTrackingForGT(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'didComputeGTResults', @(source,event)(obj.showGTResults(source, event))) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didLoadProject',@(source,event)(obj.didLoadProject()));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'update_text_trackerinfo',@(source,event)(obj.update_text_trackerinfo_()));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'refreshTrackMonitorViz',@(source,event)(obj.refreshTrackMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTrackMonitorViz',@(source,event)(obj.updateTrackMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'refreshTrainMonitorViz',@(source,event)(obj.refreshTrainMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTrainMonitorViz',@(source,event)(obj.updateTrainMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'raiseTrainingStoppedDialog',@(source,event)(obj.raiseTrainingStoppedDialog()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'newProject',@(source,event)(obj.didCreateNewProject()));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetProjectName',@(source,event)(obj.didChangeProjectName()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetProjFSInfo',@(source,event)(obj.didChangeProjFSInfo()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieInvert',@(source,event)(obj.didChangeMovieInvert()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'initialize_menu_track_tracking_algorithm',@(source,event)(obj.initialize_menu_track_tracking_algorithm_()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'update_menu_track_tracking_algorithm',@(source,event)(obj.update_menu_track_tracking_algorithm_()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'update_menu_track_tracker_history',@(source,event)(obj.update_menu_track_tracker_history_()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetCurrTracker',@(source,event)(obj.cbkCurrTrackerChanged()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLastLabelChangeTS',@(source,event)(obj.cbkLastLabelChangeTS()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackParams',@(source,event)(obj.cbkParameterChange()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackDLBackEnd', @(src,evt)(obj.update_menu_track_backend_config_()) ) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTargetCentrationAndZoom', @(src,evt)(obj.updateTargetCentrationAndZoom_()) ) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'trainStart', @(src,evt) (obj.cbkTrackerTrainStart())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'trainEnd', @(src,evt) (obj.cbkTrackerTrainEnd())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'trackStart', @(src,evt) (obj.cbkTrackerStart())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'trackEnd', @(src,evt) (obj.cbkTrackerEnd())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackerHideViz', @(src,evt) (obj.cbkTrackerHideVizChanged())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackerShowPredsCurrTargetOnly', @(src,evt) (obj.cbkTrackerShowPredsCurrTargetOnlyChanged())) ;
      
      obj.listeners_(end+1) = ...
        addlistener(labeler.progressMeter, 'didArm', @(source,event)(obj.armWaitbar())) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler.progressMeter, 'update', @(source,event)(obj.updateWaitbar())) ;      

      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetCurrTarget',@(s,e)(obj.cbkCurrTargetChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLabelMode',@(s,e)(obj.cbkLabelModeChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLabels2Hide',@(s,e)(obj.cbkLabels2HideChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLabels2ShowCurrTargetOnly',@(s,e)(obj.cbkLabels2ShowCurrTargetOnlyChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetShowTrx',@(s,e)(obj.cbkShowTrxChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetShowOccludedBox',@(s,e)(obj.cbkShowOccludedBoxChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetShowTrxCurrTargetOnly',@(s,e)(obj.cbkShowTrxCurrTargetOnlyChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackModeIdx',@(s,e)(obj.cbkTrackModeIdxChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackNFramesSmall',@(s,e)(obj.cbkTrackerNFramesChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackNFramesLarge',@(s,e)(obj.cbkTrackerNFramesChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackNFramesNear',@(s,e)(obj.cbkTrackerNFramesChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieCenterOnTarget',@(s,e)(obj.cbkMovieCenterOnTargetChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieRotateTargetUp',@(s,e)(obj.cbkMovieRotateTargetUpChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieForceGrayscale',@(s,e)(obj.cbkMovieForceGrayscaleChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieViewBGsubbed',@(s,e)(obj.cbkMovieViewBGsubbedChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLblCore',@(src,evt)(obj.didSetLblCore(src, evt)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'gtIsGTModeChanged',@(s,e)(obj.cbkGtIsGTModeChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'cropIsCropModeChanged',@(s,e)(obj.cbkCropIsCropModeChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'cropUpdateCropGUITools',@(s,e)(obj.cbkUpdateCropGUITools(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'cropCropsChanged',@(s,e)(obj.cbkCropCropsChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'newMovie',@(s,e)(obj.cbkNewMovie(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'dataImported',@(s,e)(obj.cbkDataImported(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetShowSkeleton',@(s,e)(obj.cbkShowSkeletonChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetShowMaRoi',@(s,e)(obj.cbkShowMaRoiChanged(s,e)));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetShowMaRoiAux',@(s,e)(obj.cbkShowMaRoiAuxChanged(s,e)));

      obj.listeners_(end+1) = ...
        addlistener(handles.axes_curr,'XLim','PostSet',@(s,e)(obj.axescurrXLimChanged(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.axes_curr,'XDir','PostSet',@(s,e)(obj.axescurrXDirChanged(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.axes_curr,'YDir','PostSet',@(s,e)(obj.axescurrYDirChanged(s,e))) ;

      obj.listeners_(end+1) = ...
        addlistener(handles.labelTLInfo,'selectOn','PostSet',@(s,e)(obj.cbklabelTLInfoSelectOn(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.labelTLInfo,'props','PostSet',@(s,e)(obj.cbklabelTLInfoPropsUpdated(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.labelTLInfo,'props_tracker','PostSet',@(s,e)(obj.cbklabelTLInfoPropsUpdated(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.labelTLInfo,'props_allframes','PostSet',@(s,e)(obj.cbklabelTLInfoPropsUpdated(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.labelTLInfo,'proptypes','PostSet',@(s,e)(obj.cbklabelTLInfoPropTypesUpdated(s,e))) ;

      obj.listeners_(end+1) = ...
        addlistener(handles.slider_frame,'ContinuousValueChange',@(s,e)(obj.controlActuated('slider_frame', s, e))) ;
      obj.listeners_(end+1) = ...
        addlistener(handles.sldZoom,'ContinuousValueChange',@(s,e)(obj.controlActuated('sldZoom', s, e))) ;
      
      % Stash the guidata
      guidata(mainFigure, handles) ;
      
      % Do this once listeners are set up
      obj.labeler_.handleCreationTimeAdditionalArguments_(varargin{:}) ;
    end

    function delete(obj)
      % Having the figure without a controller would be bad, so we make sure to
      % delete the figure (and subfigures) in our destructor.
      % We also delete the model.
      deleteValidGraphicsHandles(obj.satellites_) ;
      deleteValidGraphicsHandles(obj.waitbarFigure_) ;
      delete(obj.trackingMonitorVisualizer_) ;
      delete(obj.trainingMonitorVisualizer_) ;
      delete(obj.movieManagerController_) ;
      deleteValidGraphicsHandles(obj.mainFigure_) ;
      % In principle, a controller shouldn't delete its model---the model should be
      % allowed to persist until there are no more references to it.  
      % But it seems like this might surprise & annoy clients, b/c they expect that
      % when they quit APT via the GUI, the model should be deleted, even if (say)
      % there's still a reference to it in the top level scope.
      delete(obj.labeler_) ;
    end  % function
    
    function updateDoesNeedSave(obj, ~, ~)      
      labeler = obj.labeler_ ;
      doesNeedSave = labeler.doesNeedSave ;
      handles = guidata(obj.mainFigure_) ;
      hTx = handles.txUnsavedChanges ;
      if doesNeedSave
        set(hTx,'Visible','on');
      else
        set(hTx,'Visible','off');
      end
    end

    function updateStatus(obj, ~, ~)
      % Update the status text box to reflect the current model state.
      labeler = obj.labeler_ ;
      handles = guidata(obj.mainFigure_) ;
      is_busy = labeler.isStatusBusy ;
      if is_busy
        color = handles.busystatuscolor;
        if isfield(handles,'figs_all') && any(isgraphics(handles.figs_all)),
          set(handles.figs_all(isgraphics(handles.figs_all)),'Pointer','watch');
        else
          set(obj.mainFigure_,'Pointer','watch');
        end
      else
        color = handles.idlestatuscolor;
        if isfield(handles,'figs_all') && any(isgraphics(handles.figs_all)),
          set(handles.figs_all(isgraphics(handles.figs_all)),'Pointer','arrow');
        else
          set(obj.mainFigure_,'Pointer','arrow');
        end
      end
      set(handles.txStatus,'ForegroundColor',color);

      % Actually update the String in the status text box.  Use the shorter status
      % string from the labeler if the normal one is too long for the text box.
      raw_status_string = labeler.rawStatusString;
      has_project = labeler.hasProject ;
      project_file_path = labeler.projectfile ;
      status_string = ...
        interpolate_status_string(raw_status_string, has_project, project_file_path) ;
      set(handles.txStatus,'String',status_string) ;
      % If the textbox is overstuffed, change to the shorter status string
      drawnow('nocallbacks') ;  % Make sure extent is accurate
      extent = get(handles.txStatus,'Extent') ;  % reflects the size fo the String property
      position = get(handles.txStatus,'Position') ;  % reflects the size of the text box
      string_width = extent(3) ;
      box_width = position(3) ;
      if string_width > 0.95*box_width ,
        shorter_status_string = ...
          interpolate_shorter_status_string(raw_status_string, has_project, project_file_path) ;
        if isequal(shorter_status_string, status_string) ,
          % Sometimes the "shorter" status string is the same---don't change the
          % text box if that's the case
        else
          set(handles.txStatus,'String',shorter_status_string) ;
        end
      end

      % Make sure to update graphics now-ish
      drawnow('limitrate', 'nocallbacks');
    end

    function didSetTrx(obj, ~, ~)
      trx = obj.labeler_.trx ;
      obj.tvTrx_.init(true, numel(trx)) ;
    end

    function quitRequested(obj)
      is_ok_to_quit = obj.raiseUnsavedChangesDialogIfNeeded() ;
      if is_ok_to_quit ,
        delete(obj) ;
      end      
    end

    function is_ok_to_proceed = raiseUnsavedChangesDialogIfNeeded(obj)
      labeler = obj.labeler_ ;
      
      if ~verLessThan('matlab','9.6') && batchStartupOptionUsed
        return
      end

      OPTION_SAVE = 'Save first';
      OPTION_PROC = 'Proceed without saving';
      OPTION_CANC = 'Cancel';
      if labeler.doesNeedSave ,
        res = questdlg('You have unsaved changes to your project. If you proceed without saving, your changes will be lost.',...
          'Unsaved changes',OPTION_SAVE,OPTION_PROC,OPTION_CANC,OPTION_SAVE);
        switch res
          case OPTION_SAVE
            labeler.projSaveSmart();
            labeler.projAssignProjNameFromProjFileIfAppropriate();
            is_ok_to_proceed = true;
          case OPTION_CANC
            is_ok_to_proceed = false;
          case OPTION_PROC
            is_ok_to_proceed = true;
        end
      else
        % The model has no unsaved changes
        is_ok_to_proceed = true;        
      end
    end

    function updateTrxSetShowTrue(obj, ~, ~)
      % Update .hTrx, .hTraj based on .trx, .showTrx*, .currFrame
      labeler = obj.labeler_ ;
      if ~labeler.hasTrx,
        return
      end           
      tfShow = labeler.which_trx_are_showing() ;      
      tv = obj.tvTrx_ ;
      tv.setShow(tfShow);
      tv.updateTrx(tfShow);
    end
    
    function updateTrxSetShowFalse(obj, ~, ~)
      % Update .hTrx, .hTraj based on .trx, .showTrx*, .currFrame
      labeler = obj.labeler_ ;
      if ~labeler.hasTrx,
        return
      end            
      tfShow = labeler.which_trx_are_showing() ;      
      tv = obj.tvTrx_ ;
      tv.updateTrx(tfShow);
    end
    
    function didSetLblCore(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      lblCore = labeler.lblCore ;
      if ~isempty(lblCore) ,
        % Add listeners for setting lblCore props.  (At some point, these too will
        % feel the holy fire.)
        lblCore.addlistener('hideLabels', 'PostSet', @(src,evt)(obj.lblCoreHideLabelsChanged())) ;
        if isprop(lblCore,'streamlined')
          lblCore.addlistener('streamlined', 'PostSet', @(src,evt)(obj.lblCoreStreamlinedChanged())) ;
        end
        % Trigger the callbacks 'manually' to update UI elements right now
        obj.lblCoreHideLabelsChanged() ;
        if isprop(lblCore,'streamlined')
          obj.lblCoreStreamlinedChanged() ;
        end
      end      
    end

    function lblCoreHideLabelsChanged(obj)
      labeler = obj.labeler_ ;
      lblCore = labeler.lblCore ;
      handles = guidata(obj.mainFigure_) ;
      handles.menu_view_hide_labels.Checked = onIff(lblCore.hideLabels) ;
    end
    
    function lblCoreStreamlinedChanged(obj)
      labeler = obj.labeler_ ;
      lblCore = labeler.lblCore ;
      handles = guidata(obj.mainFigure_) ;
      handles.menu_setup_streamlined.Checked = onIff(lblCore.streamlined) ;
    end

    function pbTrack_actuated_(obj, source, event)
      obj.track_core_(source, event) ;
    end

    function menu_start_tracking_but_dont_call_python_actuated_(obj, source, event)
      obj.track_core_(source, event, 'do_call_apt_interface_dot_py', false) ;
    end
    
    function track_core_(obj, source, event, varargin)  %#ok<INUSD> 
      obj.labeler_.track(varargin{:}) ;
    end

%     function mftset = get_track_mode_(obj)
%       % This is designed to do the same thing as LabelerGUI::getTrackMode().
%       % The two methods should likely be consolidated at some point.  Private by
%       % convention
%       pumTrack = findobj(obj.mainFigure_, 'Tag', 'pumTrack') ;
%       idx = pumTrack.Value ;
%       % Note, .TrackingMenuNoTrx==.TrackingMenuTrx(1:K), so we can just index
%       % .TrackingMenuTrx.
%       mfts = MFTSetEnum.TrackingMenuTrx;
%       mftset = mfts(idx);      
%     end

    function menu_debug_generate_db_actuated_(obj, source, event)
      obj.train_core_(source, event, 'do_just_generate_db', true) ;
    end

    function pbTrain_actuated_(obj, source, event)
      obj.train_core_(source, event) ;
    end

    function menu_start_training_but_dont_call_python_actuated_(obj, source, event)
      obj.train_core_(source, event, 'do_call_apt_interface_dot_py', false) ;
    end

    function train_core_(obj, source, event, varargin)
      % This is like pbTrain_Callback() in LabelerGUI.m, but set up to stop just
      % after DB creation.

      % Process keyword args
      [do_just_generate_db, ...
       do_call_apt_interface_dot_py] = ...
        myparse(varargin, ...
                'do_just_generate_db', false, ...
                'do_call_apt_interface_dot_py', true) ;
      
      labeler = obj.labeler_ ;
      [doTheyExist, message] = labeler.doProjectAndMovieExist() ;
      if ~doTheyExist ,
        error(message) ;
      end
      if labeler.doesNeedSave ,
        res = questdlg('Project has unsaved changes. Save before training?','Save Project','Save As','No','Cancel','Save As') ;
        if strcmp(res,'Cancel')
          return
        elseif strcmp(res,'Save As')
          menu_file_saveas_Callback(source, event, guidata(source)) ;
        end    
      end

      % See if the tracker is in a fit state to be trained
      [tfCanTrain, reason] = labeler.trackCanTrain() ;
      if ~tfCanTrain,
        error('Tracker not fit to be trained: %s', reason) ;
      end
      
      % Call on the labeler to do the real training
      labeler.train(...
        'trainArgs',{}, ...
        'do_just_generate_db', do_just_generate_db, ...
        'do_call_apt_interface_dot_py', do_call_apt_interface_dot_py) ;
    end  % method

    function menu_quit_but_dont_delete_temp_folder_actuated_(obj, source, event)  %#ok<INUSD> 
      obj.labeler_.projTempDirDontClearOnDestructor = true ;
      obj.quitRequested() ;
    end  % method    

    function menu_track_backend_config_aws_configure_actuated_(obj, source, event)  %#ok<INUSD> 
      obj.selectAwsInstance_('canlaunch',1,...
                             'canconfigure',2, ...
                             'forceSelect',1) ;
    end

    function menu_track_backend_config_aws_setinstance_actuated_(obj, source, event)  %#ok<INUSD> 
      obj.selectAwsInstance_() ;
    end

    function menu_track_tracking_algorithm_item_actuated_(obj, source, event)  %#ok<INUSD> 
      % Get the tracker index
      tracker_index = source.UserData;
      labeler = obj.labeler_ ;

      % Validation happens inside Labeler now
      % % Validate it
      % trackers = labeler.trackersAll;
      % tracker_count = numel(trackers) ;
      % if ~is_index_in_range(tracker_index, tracker_count)
      %   error('APT:invalidPropertyValue', 'Invalid tracker index') ;
      % end
      
      % % If a custom top-down tracker, ask if we want to keep it or make a new one.
      % previousTracker = trackers{tracker_index};
      % if isa(previousTracker,'DeepTrackerTopDownCustom')
      %   do_use_previous = ask_if_should_use_previous_custom_top_down_tracker(previousTracker) ;
      % else
      %   do_use_previous = [] ;  % value will be ignored
      % end  % if isa(tAll{iTrk},'DeepTrackerTopDownCustom')
      
      % Finally, call the model method to set the tracker
      labeler.trackMakeNewTrackerCurrent(tracker_index) ;      
    end

    function menu_track_tracker_history_item_actuated_(obj, source, event)  %#ok<INUSD> 
      % Get the index of the tracker in the tracker history
      trackerHistoryIndex = source.UserData ;

      % Call the labeler method
      labeler = obj.labeler_ ;
      labeler.trackMakeOldTrackerCurrent(trackerHistoryIndex) ;      
    end

    function showDialogAfterHopefullySpawningTrackingForGT(obj, source, event)  %#ok<INUSD> 
      % Event handler that gets called after the labeler tries to spawn jobs for GT.
      % Raises a dialog, and registers it as a 'satellite' window so we can delete
      % it when the main window closes.
      labeler = obj.labeler_ ;
      tfsucc = labeler.didSpawnTrackingForGT ;
      DIALOGTTL = 'GT Tracking';
      if isscalar(tfsucc) && tfsucc ,
        msg = 'Tracking of GT frames spawned. GT results will be shown when tracking is complete.';
        h = msgbox(msg,DIALOGTTL);
      else
        msg = sprintf('GT tracking failed');
        h = warndlg(msg,DIALOGTTL);
      end
      obj.addSatellite(h) ;  % register dialog to we can delete when main window closes
      %obj.satellites_(1,end+1) = h ;  % register dialog to we can delete when main window closes
    end

    function showGTResults(obj, source, event)  %#ok<INUSD> 
      % Event handler that gets called after the labeler finishes computing GT results.
      % Raises a dialog, and registers it as a 'satellite' window so we can delete
      % it when the main window closes.
      obj.createGTResultFigures_() ;
      h = msgbox('GT results available in Labeler property ''gtTblRes''.');
      % obj.satellites_(1,end+1) = h ;  % register dialog to we can delete when main window closes
      obj.addSatellite(h) ;  % register dialog to we can delete when main window closes
    end

    function createGTResultFigures_(obj, varargin)
      labeler = obj.labeler_ ;      
      t = labeler.gtTblRes;

      [fcnAggOverPts,aggLabel] = ...
        myparse(varargin,...
                'fcnAggOverPts',@(x)max(x,[],ndims(x)), ... % or eg @mean
                'aggLabel','Max' ...
                );
      
      l2err = t.L2err;  % For single-view MA, nframes x nanimals x npts.  For single-view SA, nframes x npts
      aggOverPtsL2err = fcnAggOverPts(l2err);  
        % t.L2err, for a single-view MA project, seems to be 
        % ground-truth-frame-count x animal-count x keypoint-count, and
        % aggOverPtsL2err is ground-truth-frame-count x animal-count.
        %   -- ALT, 2024-11-19
      % KB 20181022: Changed colors to match sets instead of points
      clrs =  labeler.LabelPointColors;
      nclrs = size(clrs,1);
      lsz = size(l2err);
      npts = lsz(end);
      assert(npts==labeler.nLabelPoints);
      if nclrs~=npts
        warningNoTrace('Labeler:gt',...
          'Number of colors do not match number of points.');
      end

      if ndims(l2err) == 3
        l2err_reshaped = reshape(l2err,[],npts);  % For single-view MA, (nframes*nanimals) x npts
        valid = ~all(isnan(l2err_reshaped),2);
        l2err_filtered = l2err_reshaped(valid,:);  % For single-view MA, nvalidanimalframes x npts
      else        
        % Why don't we need to filter for e.g. single-view SA?  -- ALT, 2024-11-21
        l2err_filtered = l2err ;
      end
      nviews = labeler.nview;
      nphyspt = npts/nviews;
      prc_vals = [50,75,90,95,98];
      prcs = prctile(l2err_filtered,prc_vals,1);
      prcs = reshape(prcs,[],nphyspt,nviews);
      lpos = t(1,:).pLbl;
      if ndims(lpos)==3
        lpos = squeeze(lpos(1,1,:));
      else
        lpos = squeeze(lpos(1,:));
      end
      lpos = reshape(lpos,npts,2);
      allims = cell(1,nviews);
      allpos = zeros([nphyspt,2,nviews]);
      txtOffset = labeler.labelPointsPlotInfo.TextOffset;
      for view = 1:nviews
        curl = lpos( ((view-1)*nphyspt+1):view*nphyspt,:);
        [im,isrotated,~,~,A] = labeler.readTargetImageFromMovie(t.mov(1),t.frm(1),t.iTgt(1),view);
        if isrotated
          curl = [curl,ones(nphyspt,1)]*A;
          curl = curl(:,1:2);
        end
        minpos = min(curl,[],1);
        maxpos = max(curl,[],1);
        centerpos = (minpos+maxpos)/2;
        % border defined by borderfrac
        r = max(1,(maxpos-minpos));
        xlim = round(centerpos(1)+[-1,1]*r(1));
        ylim = round(centerpos(2)+[-1,1]*r(2));
        xlim = min(size(im,2),max(1,xlim));
        ylim = min(size(im,1),max(1,ylim));
        im = im(ylim(1):ylim(2),xlim(1):xlim(2),:);
        curl(:,1) = curl(:,1)-xlim(1);
        curl(:,2) = curl(:,2)-ylim(1);
        allpos(:,:,view) = curl;
        allims{view} = im;
      end  % for
      
      fig_1 = figure('Name','GT err percentiles');
      %obj.satellites_(1,end+1) = fig_1 ;
      obj.addSatellite(fig_1) ;
      plotPercentileHist(allims,prcs,allpos,prc_vals,fig_1,txtOffset)

      % Err by landmark
      fig_2 = figure('Name','GT err by landmark');
      %obj.satellites_(1,end+1) = fig_2 ;
      obj.addSatellite(fig_2) ;
      ax = axes(fig_2) ;
      boxplot(ax,l2err_filtered,'colors',clrs,'boxstyle','filled');
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Landmark/point',args{:});
      if nviews>1
        xtick_str = {};
        for view = 1:nviews
          for n = 1:nphyspt
            if n==1
              xtick_str{end+1} = sprintf('View %d -- %d',view,n); %#ok<AGROW> 
            else
              xtick_str{end+1} = sprintf('%d',n); %#ok<AGROW> 
            end
          end
        end
        xticklabels(xtick_str)
      end
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,'GT err by landmark',args{:});
      ax.YGrid = 'on';
      
      % AvErrAcrossPts by movie
      tstr = sprintf('%s (over landmarks) GT err by movie',aggLabel);
      fig_3 = figurecascaded(fig_2,'Name',tstr);
      % obj.satellites_(1,end+1) = fig_3 ;
      obj.addSatellite(fig_3) ;      
      ax = axes(fig_3);
      [iMovAbs,gt] = t.mov.get();  % both outputs are nframes x 1
      assert(all(gt));
      grp = categorical(iMovAbs);
      if ndims(aggOverPtsL2err)==3
        taggerr = permute(aggOverPtsL2err,[1,3,2]);
      else
        taggerr = aggOverPtsL2err ;
      end
      % expand out grp to match elements of taggerr 1-to-1
      assert(isequal(size(taggerr,1), size(grp,1))) ;
      taggerr_shape = size(taggerr) ;
      biggrp = repmat(grp, [1 taggerr_shape(2:end)]) ;
      assert(isequal(size(taggerr), size(biggrp))) ;
      % columnize
      taggerr_column = taggerr(:) ;
      grp_column = biggrp(:) ;
      grplbls = arrayfun(@(z1,z2)sprintf('mov%s (n=%d)',z1{1},z2),...
                         categories(grp_column),countcats(grp_column),...
                         'UniformOutput',false);
      boxplot(ax, ...
              taggerr_column,...
              grp_column,...
              'colors',clrs,...
              'boxstyle','filled',...
              'labels',grplbls);
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Movie',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,tstr,args{:});
      ax.YGrid = 'on';
%      
      % Mean err by movie, pt
%       fig_4 = figurecascaded(fig_3,'Name','Mean GT err by movie, landmark');
%       obj.satellites_(1,end+1) = fig_4 ;
%       ax = axes(fig_4);
%       tblStats = grpstats(t(:,{'mov' 'L2err'}),{'mov'});
%       tblStats.mov = tblStats.mov.get;
%       tblStats = sortrows(tblStats,{'mov'});
%       movUnCnt = tblStats.GroupCount; % [nmovx1]
%       meanL2Err = tblStats.mean_L2err; % [nmovxnpt]
%       nmovUn = size(movUnCnt,1);
%       szassert(meanL2Err,[nmovUn npts]);
%       meanL2Err(:,end+1) = nan; % pad for pcolor
%       meanL2Err(end+1,:) = nan;       
%       hPC = pcolor(meanL2Err);
%       hPC.LineStyle = 'none';
%       colorbar;
%       xlabel(ax,'Landmark/point',args{:});
%       ylabel(ax,'Movie',args{:});
%       xticklbl = arrayfun(@num2str,1:npts,'uni',0);
%       yticklbl = arrayfun(@(x)sprintf('mov%d (n=%d)',x,movUnCnt(x)),1:nmovUn,'uni',0);
%       set(ax,'YTick',0.5+(1:nmovUn),'YTickLabel',yticklbl);
%       set(ax,'XTick',0.5+(1:npts),'XTickLabel',xticklbl);
%       axis(ax,'ij');
%       title(ax,'Mean GT err (px) by movie, landmark',args{:});
%       
%       nmontage = min(nmontage,height(t));
%       obj.trackLabelMontage(t,'aggOverPtsL2err','hPlot',fig_4,'nplot',nmontage);
    end  % function
    
    function selectAwsInstance_(obj, varargin)
      [canLaunch,canConfigure,forceSelect] = ...
        myparse(varargin, ...
                'canlaunch',true,...
                'canconfigure',1,...
                'forceSelect',true);
            
      labeler = obj.labeler_ ;
      backend = labeler.trackDLBackEnd ;
      if isempty(backend) ,
        error('Backend not configured') ;
      end
      if backend.type ~= DLBackEnd.AWS ,
        error('Backend is not of type AWS') ;
      end        
      awsec2 = backend.awsec2 ;
      if ~awsec2.areCredentialsSet || canConfigure >= 2,
        if canConfigure,
          [tfsucc,keyName,pemFile] = ...
            promptUserToSpecifyPEMFileName(awsec2.keyName,awsec2.pem);
          if tfsucc ,
            % For changing things in the model, we go through the top-level model object
            %labeler.setAwsPemFileAndKeyName(pemFile, keyName) ;
            labeler.set_backend_property('awsPEM', pemFile) ;
            labeler.set_backend_property('awsKeyName', keyName) ;            
          end
          if ~tfsucc && ~awsec2.areCredentialsSet,
            reason = 'AWS EC2 instance is not configured.';
            error(reason) ;
          end
        else
          reason = 'AWS EC2 instance is not configured.';
          error(reason) ;
        end
      end
      if forceSelect || ~awsec2.isInstanceIDSet,
        if awsec2.isInstanceIDSet ,
          instanceID = awsec2.instanceID;
        else
          instanceID = '';
        end
        if canLaunch,
          qstr = 'Launch a new instance or attach to an existing instance?';
          if ~awsec2.isInstanceIDSet,
            qstr = ['APT is not attached to an AWS EC2 instance. ',qstr];
          else
            qstr = sprintf('APT currently attached to AWS EC2 instance %s. %s',instanceID,qstr);
          end
          tstr = 'Specify AWS EC2 instance';
          btn = questdlg(qstr,tstr,'Launch New','Attach to Existing','Cancel','Cancel');
          if isempty(btn)
            btn = 'Cancel';
          end
        else
          btn = 'Attach to Existing';
        end
        while true,
          switch btn
            case 'Launch New'
              tf = awsec2.launchInstance();
              if ~tf
                reason = 'Could not launch AWS EC2 instance.';
                error(reason) ;
              end
              break
            case 'Attach to Existing',
              [tfsucc,instanceIDs,instanceTypes] = awsec2.listInstances();
              if ~tfsucc,
                reason = 'Error listing instances.';
                error(reason) ;
              end
              if isempty(instanceIDs),
                if canLaunch,
                  btn = questdlg('No instances found. Launch a new instance?',tstr,'Launch New','Cancel','Cancel');
                  continue
                else
                  reason = 'No instances found.';
                  error(reason) ;
                end
              end
              
              PROMPT = {
                'Instance'
                };
              NAME = 'AWS EC2 Select Instance';
              INPUTBOXWIDTH = 100;
              BROWSEINFO = struct('type',{'popupmenu'});
              s = cellfun(@(x,y) sprintf('%s (%s)',x,y),instanceIDs,instanceTypes,'Uni',false);
              v = 1;
              if ~isempty(awsec2.instanceID),
                v = find(strcmp(instanceIDs,awsec2.instanceID),1);
                if isempty(v),
                  v = 1;
                end
              end
              DEFVAL = {{s,v}};
              resp = inputdlgWithBrowse(PROMPT,NAME,repmat([1 INPUTBOXWIDTH],1,1),...
                                        DEFVAL,'on',BROWSEINFO);
              tfsucc = ~isempty(resp);
              if tfsucc
                instanceID = instanceIDs{resp{1}};
                instanceType = instanceTypes{resp{1}};
              else
                return
              end
              break
            otherwise
              % This is a cancel
              return
          end
        end
        % For changing things in the model, we go through the top-level model object
        %labeler.setAWSInstanceIDAndType(instanceID, instanceType) ;
        labeler.set_backend_property('awsInstanceID', instanceID) ;
        labeler.set_backend_property('awsInstanceType', instanceType) ;
      end
    end  % function selectAwsInstance_()

    function exceptionMaybe = controlActuated(obj, controlName, source, event, varargin)  % public so that control actuation can be easily faked
      % The advantage of passing in the controlName, rather than,
      % say always storing it in the tag of the graphics object, and
      % then reading it out of the source arg, is that doing it this
      % way makes it easier to fake control actuations by calling
      % this function with the desired controlName and an empty
      % source and event.
      if obj.isInYodaMode_ ,
        % "Do, or do not.  There is no try." --Yoda
        obj.controlActuatedCore_(controlName, source, event, varargin{:}) ;
        exceptionMaybe = {} ;
      else        
        try
          obj.controlActuatedCore_(controlName, source, event, varargin{:}) ;
          exceptionMaybe = {} ;
        catch exception
          obj.labeler_.clearStatus() ;
          if isequal(exception.identifier,'APT:invalidPropertyValue') || isequal(exception.identifier,'APT:cancelled'),
            % ignore completely, don't even pass on to output
            exceptionMaybe = {} ;
          else
            raiseDialogOnException(exception) ;
            exceptionMaybe = { exception } ;
          end
        end
      end
    end  % function

    function controlActuatedCore_(obj, controlName, source, event, varargin)
      if isempty(source) ,
        % This means the control actuated was a 'faux' control, or in some cases 
        methodName=[controlName '_actuated_'] ;
        if ismethod(obj,methodName) ,
          obj.(methodName)(source, event, varargin{:});
        end
      else
        type=get(source,'Type');
        if isequal(type,'uitable') ,
          if isfield(event,'EditData') || isprop(event,'EditData') ,  % in older Matlabs, event is a struct, in later, an object
            methodName=[controlName '_cell_edited_'];
          else
            methodName=[controlName '_cell_selected_'];
          end
          if ismethod(obj,methodName) ,
            obj.(methodName)(source, event, varargin{:});
          end
        elseif isequal(type,'uicontrol') || isequal(type,'uimenu') ,
          methodName=[controlName '_actuated_'] ;
          if ismethod(obj,methodName) ,
            obj.(methodName)(source, event, varargin{:});
          end
        else
          % odd --- just ignore
        end
      end
    end  % function

    function armWaitbar(obj)
      % When we arm, want to re-center figure on main window, then do a normal
      % update.
      centerOnParentFigure(obj.waitbarFigure_, obj.mainFigure_) ;
      obj.updateWaitbar() ;
    end

    function updateWaitbar(obj)
      % Update the waitbar to reflect the current state of
      % obj.labeler_.progressMeter.  In addition to other things, makes figure
      % visible or not depending on that state.
      labeler = obj.labeler_ ;
      progressMeter = labeler.progressMeter ;
      visibility = onIff(progressMeter.isActive) ;
      fractionDone = progressMeter.fraction ;
      message = progressMeter.message ;
      obj.waitbarFigure_.Name = progressMeter.title ;
      obj.waitbarFigure_.Visible = visibility ;      
      waitbar(fractionDone, obj.waitbarFigure_, message) ;
    end

    function didCancelWaitbar(obj)
      labeler = obj.labeler_ ;
      progressMeter = labeler.progressMeter ;
      progressMeter.cancel() ;
    end

    function didLoadProject(obj)
      obj.updateTarget_();
      obj.enableControls_('projectloaded') ;
    end
    
    function updateTarget_(obj)
      % Get the handles out of the main figure
      main_figure = obj.mainFigure_ ;
      if isempty(main_figure) || ~isvalid(main_figure)
        return
      end      
      handles = guidata(main_figure) ;
      
      lObj = obj.labeler_ ;
      if (lObj.hasTrx || lObj.maIsMA) && ~lObj.isinit ,
        iTgt = lObj.currTarget;
        lObj.currImHud.updateTarget(iTgt);
          % lObj.currImHud is really a view object, but is stored in the Labeler for
          % historical reasons.  It should probably be stored in obj (the
          % LabelerController).  Someday we will move it, but right now it's referred to
          % by so many places in Labeler, and LabelCore, etc that I don't want to start
          % shaving that yak right now.  -- ALT, 2025-01-30
        handles.labelTLInfo.newTarget();
        if lObj.gtIsGTMode
          tfHilite = lObj.gtCurrMovFrmTgtIsInGTSuggestions();
        else
          tfHilite = false;
        end
        handles.allAxHiliteMgr.setHighlight(tfHilite);
      end
    end  % function

    function enableControls_(obj, state)
      % Enable/disable controls, as appropriate.

      % Get the handles out of the main figure
      main_figure = obj.mainFigure_ ;
      if isempty(main_figure) || ~isvalid(main_figure)
        return
      end      
      handles = guidata(main_figure) ;

      % Update the enablement of the handles, depending on the state
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
          set(handles.FigureToolBar,'Visible','off') ;
          if isfield(handles, 'menu_debug') && isgraphics(handles.menu_debug)
            set(handles.menu_debug,'Enable','off') ;
          end
            
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
          if isfield(handles, 'menu_debug') && isgraphics(handles.menu_debug)
            set(handles.menu_debug,'Enable','off') ;
          end
          
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
          if isfield(handles, 'menu_debug') && isgraphics(handles.menu_debug)
            set(handles.menu_debug,'Enable','off') ;
          end
      
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
          if isfield(handles, 'menu_debug') && isgraphics(handles.menu_debug)
            set(handles.menu_debug,'Enable','on') ;
          end
          
          lObj = obj.labeler_ ;
          tObj = lObj.tracker;    
          tfTracker = ~isempty(tObj);
          onOff = onIff(tfTracker);
          handles.menu_track.Enable = onOff;
          handles.pbTrain.Enable = onOff;
          handles.pbTrack.Enable = onOff;
          handles.menu_view_hide_predictions.Enable = onOff;    
          set(handles.menu_track_auto_params_update, 'Checked', lObj.trackAutoSetParams) ;
      
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
          else
            set(handles.h_ma_only,'Enable','off');
          end
          if lObj.nLabelPointsAdd == 0,
            set(handles.h_addpoints_only,'Visible','off');
          else
            set(handles.h_addpoints_only,'Visible','on');
          end

        otherwise
          error('Not implemented') ;
      end
    end  % function

    function update_text_trackerinfo_(obj)
      % Updates the tracker info string to match what's in 
      % obj.labeler_.tracker.trackerInfo.
      % Called via notify() when labeler.tracker.trackerInfo is changed.
      
      % Get the handles out of the main figure
      main_figure = obj.mainFigure_ ;
      if isempty(main_figure) || ~isvalid(main_figure)
        return
      end      
      handles = guidata(main_figure) ;
      
      % Update the relevant text object
      tracker = obj.labeler_.tracker ;
      if ~isempty(tracker) ,
        handles.text_trackerinfo.String = tracker.getTrackerInfoString();
      end
    end  % function

    function raiseTargetsTableFigure(obj)
      labeler = obj.labeler_ ;
      main_figure = obj.mainFigure_ ;
      [tfok,tblBig] = labeler.hlpTargetsTableUIgetBigTable();
      if ~tfok
        return
      end
      
      tblSumm = labeler.trackGetSummaryTable(tblBig) ;
      hF = figure('Name','Target Summary (click row to navigate)',...
                  'MenuBar','none','Visible','off', ...
                  'Tag', 'target_table_figure');
      hF.Position(3:4) = [1280 500];
      centerfig(hF, main_figure);
      hPnl = uipanel('Parent',hF,'Position',[0 .08 1 .92],'Tag','uipanel_TargetsTable');
      BTNWIDTH = 100;
      DXY = 4;
      btnHeight = hPnl.Position(2)*hF.Position(4)-2*DXY;
      btnPos = [hF.Position(3)-BTNWIDTH-DXY DXY BTNWIDTH btnHeight];      
      hBtn = uicontrol('Style','pushbutton','Parent',hF,...
                       'Position',btnPos,'String','Update',...
                       'fontsize',12, ...
                       'Tag', 'target_table_update_button');
      FLDINFO = {
        'mov' 'Movie' 'integer' 30
        'iTgt' 'Target' 'integer' 30
        'trajlen' 'Traj. Length' 'integer' 45
        'frm1' 'Start Frm' 'integer' 30
        'nFrmLbl' '# Frms Lbled' 'integer' 60
        'nFrmTrk' '# Frms Trked' 'integer' 60
        'nFrmImported' '# Frms Imported' 'integer' 90
        'nFrmLblTrk' '# Frms Lbled&Trked' 'integer' 120
        'lblTrkMeanErr' 'Track Err' 'float' 60
        'nFrmLblImported' '# Frms Lbled&Imported' 'integer'  120
        'lblImportedMeanErr' 'Imported Err' 'float' 60
        'nFrmXV' '# Frms XV' 'integer' 40
        'xvMeanErr' 'XV Err' 'float' 40 };
      tblfldsassert(tblSumm,FLDINFO(:,1));
      nt = NavigationTable(hPnl,...
                           [0 0 1 1],...
                           @(row,rowdata)(obj.controlActuated('target_table_row', [], [], row, rowdata)),...
                           'ColumnName',FLDINFO(:,2)',...
                           'ColumnFormat',FLDINFO(:,3)',...
                           'ColumnPreferredWidth',cell2mat(FLDINFO(:,4)'));
      %                    @(row,rowdata)(labeler.setMFT(rowdata.mov,rowdata.frm1,rowdata.iTgt)),...
      nt.setData(tblSumm);
      
      hF.UserData = nt;
      %hBtn.Callback = @(s,e)labeler.hlpTargetsTableUIupdate(nt);
      hBtn.Callback = @(source,event)(obj.controlActuated(hBtn.Tag, source, event)) ;
      hF.Units = 'normalized';
      hBtn.Units = 'normalized';
      hF.Visible = 'on';

      obj.addSatellite(hF) ;
    end  % function
    
    function target_table_row_actuated_(obj, source, event, row, rowdata)  %#ok<INUSD>
      % Does what needs doing when the target table row is selected.
      labeler = obj.labeler_ ;
      labeler.setMFT(rowdata.mov,rowdata.frm1,rowdata.iTgt) ;
    end  % function

    function target_table_update_button_actuated_(obj, source, event)  %#ok<INUSD>
      % Does what needs doing when the target table update button is actuated.
      labeler = obj.labeler_ ;      
      [tfok, tblBig] = labeler.hlpTargetsTableUIgetBigTable() ;
      if tfok
        fig = obj.findSatelliteByTag_('target_table_figure') ;
        if ~isempty(fig) ,
          navTbl = fig.UserData ;
          navTbl.setData(labeler.trackGetSummaryTable(tblBig)) ;
        end
      end
    end  % function
    
    % function addDepHandle_(obj, h)
    %   % Add the figure handle h to the list of dependent figures
    %   main_figure = obj.mainFigure_ ;
    %   handles = guidata(main_figure) ;
    %   handles.depHandles(end+1,1) = h ;
    %   guidata(main_figure,handles) ;
    % end  % function
    
    function result = isSatellite(obj, h)
      result = any(obj.satellites_ == h) ;
    end

    function h = findSatelliteByTag_(obj, query_tag)
      % Find the handle with Tag query_tag in handles.depHandles.
      % If no matching tag, returns [].
      tags = arrayfun(@(h)(h.Tag), obj.satellites_, 'UniformOutput', false) ;
      is_match = strcmp(query_tag, tags) ;
      index = find(is_match,1) ;
      if isempty(index) ,
        h = [] ;
      else
        h = obj.satellites_(index) ;
      end
    end  % function

    function suspComputeUI(obj)
      labeler = obj.labeler_ ;      
      tfsucc = labeler.suspCompute();
      if ~tfsucc
        return
      end
      title = sprintf('Suspicious frames: %s',labeler.suspDiag) ;
      hF = figure('Name',title);
      tbl = labeler.suspSelectedMFT ;
      tblFlds = tbl.Properties.VariableNames;
      nt = NavigationTable(hF, ...
                           [0 0 1 1], ...
                           @(row,rowdata)(obj.controlActuated('susp_frame_table_row', [], [], row, rowdata)),...
                           'ColumnName',tblFlds);
                           % @(i,rowdata)(obj.suspCbkTblNaved(i)),...
      nt.setData(tbl);
      hF.UserData = nt;
      obj.addSatellite(hF);
    end  % function

    function susp_frame_table_row_actuated_(obj, source, event, row, rowdata)  %#ok<INUSD>
      % Does what needs doing when the suspicious frame table row is selected.
      labeler = obj.labeler_ ;
      labeler.suspCbkTblNaved(row) ;
    end  % function
    
    function refreshTrackMonitorViz(obj)
      % Create a TrackMonitorViz (very similar to a controller) if one doesn't
      % exist.  If one *does* exist, delete that one first.
      labeler = obj.labeler_ ;
      if ~isempty(obj.trackingMonitorVisualizer_) 
        if isvalid(obj.trackingMonitorVisualizer_) ,
          delete(obj.trackingMonitorVisualizer_) ;
        end
        obj.trackingMonitorVisualizer_ = [] ;
      end
      obj.trackingMonitorVisualizer_ = TrackMonitorViz(obj, labeler) ;
    end  % function

    function updateTrackMonitorViz(obj)
      labeler = obj.labeler_ ;
      sRes = labeler.tracker.bgTrkMonitor.sRes ;
      if ~isempty(obj.trackingMonitorVisualizer_) && isvalid(obj.trackingMonitorVisualizer_) ,
        obj.trackingMonitorVisualizer_.resultsReceived(sRes) ;
      end
    end  % function

    function refreshTrainMonitorViz(obj)
      % Create a TrainMonitorViz (very similar to a controller) if one doesn't
      % exist.  If one *does* exist, delete that one first.
      labeler = obj.labeler_ ;
      if ~isempty(obj.trainingMonitorVisualizer_) 
        if isvalid(obj.trainingMonitorVisualizer_) ,
          delete(obj.trainingMonitorVisualizer_) ;
        end
        obj.trainingMonitorVisualizer_ = [] ;
      end
      obj.trainingMonitorVisualizer_ = TrainMonitorViz(obj, labeler) ;
    end  % function

    function updateTrainMonitorViz(obj)
      labeler = obj.labeler_ ;
      sRes = labeler.tracker.bgTrnMonitor.sRes ;
      obj.trainingMonitorVisualizer_.resultsReceived(sRes) ;
    end  % function

    function addSatellite(obj, h)
      % Add a 'satellite' figure, so we don't lose track of them

      % 'GC' dead handles
      isValid = arrayfun(@isvalid, obj.satellites_) ;
      obj.satellites_ = obj.satellites_(isValid) ;

      % Make sure it's really a new one, then add it
      isSameAsNewGuy = arrayfun(@(sat)(sat==h), obj.satellites_);
      if ~any(isSameAsNewGuy)
        obj.satellites_(1, end+1) = h ;
      end
    end  % function

    function clearSatellites(obj)
      deleteValidGraphicsHandles(obj.satellites_);
      obj.satellites_ = gobjects(1,0);
    end  % function

    function raiseTrainingStoppedDialog(obj)
      % Raise a dialog that reports how many training iterations have completed, and
      % ask if the user wants to save the project.  Normally called via event
      % notification after training is stopped early via user pressing the "Stop
      % training" button in the "Training Monitor" window.
      labeler = obj.labeler_ ;
      tracker = labeler.tracker ;
      iterCurr = tracker.trackerInfo.iterCurr ;
      iterFinal = tracker.trackerInfo.iterFinal ;
      n_out_of_d_string = DeepTracker.printIter(iterCurr, iterFinal) ;
      question_string = sprintf('Training stopped after %s iterations. Save project now?',...
                                n_out_of_d_string) ;
      res = questdlg(question_string,'Save?','Save','Save as...','No','Save');
      if strcmpi(res,'Save'),
        labeler.projSaveSmart();
        labeler.projAssignProjNameFromProjFileIfAppropriate();
      elseif strcmpi(res,'Save as...'),
        labeler.projSaveAs();
        labeler.projAssignProjNameFromProjFileIfAppropriate();
      end  % if      
    end

    function didCreateNewProject(obj)
      labeler =  obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      
      %handles = clearDepHandles(handles);
      obj.clearSatellites() ;
      
      % Initialize the uitable of labeled frames
      obj.initTblFrames_() ;
      
      %curr_status_string=handles.txStatus.String;
      %SetStatus(handles,curr_status_string,true);
      
      % figs, axes, images
      deleteValidGraphicsHandles(handles.figs_all(2:end));
      handles.figs_all = handles.figs_all(1);
      handles.axes_all = handles.axes_all(1);
      handles.images_all = handles.images_all(1);
      handles.axes_occ = handles.axes_occ(1);
      
      nview = labeler.nview;
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
      %controller = handles.controller ;
      for iView=2:nview
        figs(iView) = ...
          figure('CloseRequestFcn',@(s,e)(obj.cbkAuxFigCloseReq(s,e)),...
                 'Color',figs(1).Color, ...
                 'Menubar','none', ...
                 'Toolbar','figure', ...
                 'UserData',struct('view',iView), ...
                 'Tag', sprintf('figs_all(%d)', iView) ...
                 );
        axs(iView) = axes('Parent', figs(iView));
        obj.addSatellite(figs(iView)) ;
        
        ims(iView) = imagesc(0,'Parent',axs(iView));
        set(ims(iView),'PickableParts','none');
        %axisoff(axs(iView));
        hold(axs(iView),'on');
        set(axs(iView),'Color',[0 0 0]);
        
        axparent = axs(iView).Parent;
        axpos = axs(iView).Position;
        axunits = axs(iView).Units;
        axpos(3:4) = axpos(3:4).*axOccSzRatios;
        axsOcc(iView) = ...
          axes('Parent',axparent,'Position',axpos,'Units',axunits,...
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
        handles = obj.cropInitImRects_(handles) ;
      catch ME
        fprintf(2,'Crop Mode initialization error: %s\n',ME.message);
      end
      
      if isfield(handles,'allAxHiliteMgr') && ~isempty(handles.allAxHiliteMgr)
        % Explicit deletion not supposed to be nec
        delete(handles.allAxHiliteMgr);
      end
      handles.allAxHiliteMgr = AxesHighlightManager(axs);
      
      axis(handles.axes_occ,[0 labeler.nLabelPoints+1 0 2]);
      
      % Delete handles.hLinkPrevCurr
      % The link destruction/recreation may not be necessary
      if isfield(handles,'hLinkPrevCurr') && isvalid(handles.hLinkPrevCurr)
        delete(handles.hLinkPrevCurr);
      end

      % Copy the handles back to the figure guidata, b/c obj.hlpSetConfigOnViews()
      % needs them to be up-to-date
      guidata(mainFigure, handles) ;
      
      % Configure the non-primary view windows
      viewCfg = labeler.projPrefs.View;
      handles.newProjAxLimsSetInConfig = ...
        obj.hlpSetConfigOnViews_(viewCfg, ...
                                 viewCfg(1).CenterOnTarget) ;  % lObj.CenterOnTarget is not set yet
        % This use of apt.hlpSetConfigOnViews() is a bit sketchy, but if you change
        % it, note that handles at this point is out-of-sync with guidata(mainFigure)
      AX_LINKPROPS = {'XLim' 'YLim' 'XDir' 'YDir'};
      handles.hLinkPrevCurr = ...
        linkprop([handles.axes_curr,handles.axes_prev],AX_LINKPROPS);
      
      arrayfun(@(x)(colormap(x,gray())),figs);
      obj.updateGUIFigureNames_() ;
      obj.updateMainAxesName_();
      
      arrayfun(@(fig)zoom(fig,'off'),handles.figs_all); % Cannot set KPF if zoom or pan is on
      arrayfun(@(fig)pan(fig,'off'),handles.figs_all);
      hTmp = findall(handles.figs_all,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',@(src,evt)(obj.cbkKPF(src,evt))) ;
      handles.h_ignore_arrows = [handles.slider_frame];
      %set(handles.figs_all,'WindowButtonMotionFcn',@(src,evt)cbkWBMF(src,evt,lObj));
      %set(handles.figs_all,'WindowButtonUpFcn',@(src,evt)cbkWBUF(src,evt,lObj));
      % if ispc
      %   set(handles.figs_all,'WindowScrollWheelFcn',@(src,evt)cbkWSWF(src,evt,lObj));
      % end
      
      % eg when going from proj-with-trx to proj-no-trx, targets table needs to
      % be cleared
      set(handles.tblTrx,'Data',cell(0,size(handles.tblTrx.ColumnName,2)));
      
      handles = obj.setShortcuts_(handles);
      
      handles.labelTLInfo.initNewProject();
      
      delete(obj.movieManagerController_) ;
      obj.movieManagerController_ = MovieManagerController(labeler) ;
      drawnow(); 
        % 20171002 Without this drawnow(), new tabbed MovieManager shows up with 
        % buttons clipped at bottom edge of UI (manually resizing UI then "snaps"
        % buttons/figure back into a good state)   
      obj.movieManagerController_.setVisible(false);
      
      handles.GTMgr = GTManager(labeler);
      handles.GTMgr.Visible = 'off';
      obj.addSatellite(handles.GTMgr) ;
      
      % Re-store the modified guidata in the figure
      guidata(mainFigure, handles) ;
    end  % function

    function menu_file_new_actuated_(obj, ~, ~)
      % Create a new project
      labeler = obj.labeler_ ;
      labeler.setStatus('Starting New Project');
      if obj.raiseUnsavedChangesDialogIfNeeded() ,
        cfg = ProjectSetup(obj.mainFigure_);  % launches the project setup window
        if ~isempty(cfg)    
          labeler.projNew(cfg);
          if ~isempty(obj.movieManagerController_) && isvalid(obj.movieManagerController_) ,
            obj.movieManagerController_.setVisible(true);
          else
            error('LabelerController:menu_file_new_actuated_', 'Please create or load a project.') ;
          end
        end  
      end
    end  % function

    function updateMainFigureName(obj)    
      labeler = obj.labeler_ ;
      maxlength = 80;
      if isempty(labeler.projectfile),
        projname = [labeler.projname,' (unsaved)'];
      elseif numel(labeler.projectfile) <= maxlength,
        projname = labeler.projectfile;
      else
        [~,projname] = fileparts2(labeler.projectfile);
      end
      obj.mainFigure_.Name = sprintf('APT - Project %s',projname) ;
    end  % function

    function didChangeProjectName(obj)
      obj.updateMainFigureName() ;
      obj.updateStatus() ;      
    end  % function

    function didChangeProjFSInfo(obj)
      obj.updateMainFigureName() ;
      obj.updateStatus() ;      
    end  % function

    function didChangeMovieInvert(obj)
      obj.updateGUIFigureNames_() ;
      obj.updateMainAxesName_() ;
    end  % function

    function updateGUIFigureNames_(obj)
      labeler = obj.labeler_ ;
      handles = guidata(obj.mainFigure_) ;
      figs = handles.figs_all ;

      obj.updateMainFigureName() ;
      viewNames = labeler.viewNames ;
      for i=2:labeler.nview ,
        vname = viewNames{i} ;
        if isempty(vname)
          str = sprintf('View %d',i) ;
        else
          str = sprintf('View: %s',vname) ;
        end
        if numel(labeler.movieInvert) >= i && labeler.movieInvert(i) ,
          str = [str,' (inverted)'] ;  %#ok<AGROW>
        end
        figs(i).Name = str ;
        figs(i).NumberTitle = 'off' ;
      end
    end  % function

    function updateMainAxesName_(obj)
      labeler = obj.labeler_ ;
      viewNames = labeler.viewNames ;
      if labeler.nview > 1 ,
        if isempty(viewNames{1}) ,
          str = 'View 1, ' ;
        else
          str = sprintf('View: %s, ',viewNames{1}) ;
        end
      else
        str = '' ;
      end
      mname = labeler.moviename ;
      if labeler.nview>1
        str = [str,sprintf('Movieset %d',labeler.currMovie)] ;
      else
        str = [str,sprintf('Movie %d',labeler.currMovie)] ;
      end
      if labeler.gtIsGTMode
        str = [str,' (GT)'] ;
      end
      str = [str,': ',mname] ;
      if ~isempty(labeler.movieInvert) && labeler.movieInvert(1) ,
        str = [str,' (inverted)'] ;
      end
      handles = guidata(obj.mainFigure_) ;
      set(handles.txMoviename,'String',str) ;
    end  % function
    
    function handles = setShortcuts_(obj, handles)
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      prefs = labeler.projPrefs;
      if ~isfield(prefs,'Shortcuts')
        return;
      end
      prefs = prefs.Shortcuts;
      fns = fieldnames(prefs);
      ismenu = false(1,numel(fns));
      for i = 1:numel(fns)
        h = findobj(mainFigure,'Tag',fns{i},'-property','Accelerator');
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
    end  % function

    function menu_file_shortcuts_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      while true,
        [~,newShortcuts] = propertiesGUI([],labeler.projPrefs.Shortcuts);
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
      labeler.projPrefs.Shortcuts = newShortcuts ;
      handles = guidata(obj.mainFigure_) ;
      handles = obj.setShortcuts_(handles);
      guidata(obj.mainFigure_, handles) ;
    end  % function

    function handles = cropInitImRects_(obj, handles)
      deleteValidGraphicsHandles(handles.cropHRect);
      handles.cropHRect = ...
        arrayfun(@(x)imrect(x,[nan nan nan nan]),handles.axes_all,'uni',0); %#ok<IMRECT>
      handles.cropHRect = cat(1,handles.cropHRect{:}); % ML 2016a ish can't concat imrects in arrayfun output
      arrayfun(@(x)set(x,'Visible','off','PickableParts','none','UserData',true),...
        handles.cropHRect); % userdata: see cropImRectSetPosnNoPosnCallback
      for ivw=1:numel(handles.axes_all)
        posnCallback = @(zpos)cbkCropPosn(obj,zpos,ivw);
        handles.cropHRect(ivw).addNewPositionCallback(posnCallback);
      end
    end  % function

    function cbkCropPosn(obj,posn,iview)
      labeler = obj.labeler_ ;
      hFig = obj.mainFigure_ ;
      handles = guidata(hFig) ;
      tfSetPosnLabeler = get(handles.cropHRect(iview),'UserData');
      if tfSetPosnLabeler
        [roi,roiw,roih] = CropInfo.rectPos2roi(posn);
        tb = handles.tbAdjustCropSize;
        if tb.Value==tb.Max  % tbAdjustCropSizes depressed; using as proxy for, imrect is resizable
          fprintf('roi (width,height): (%d,%d)\n',roiw,roih);
        end
        labeler.cropSetNewRoiCurrMov(iview,roi);
      end
    end  % function

    function menu_view_reset_views_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      %mainFigure = obj.mainFigure_ ;
      %handles = guidata(mainFigure) ;
      viewCfg = labeler.projPrefs.View;
      obj.hlpSetConfigOnViews_(viewCfg, labeler.movieCenterOnTarget) ;
      movInvert = ViewConfig.getMovieInvert(viewCfg);
      labeler.movieInvert = movInvert;
      labeler.movieCenterOnTarget = viewCfg(1).CenterOnTarget;
      labeler.movieRotateTargetUp = viewCfg(1).RotateTargetUp;
    end  % function
    
    function tfKPused = cbkKPF(obj, source, event)
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.isReady ,
        return
      end
      
      tfKPused = false;
      isarrow = ismember(event.Key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'});
      if isarrow && ismember(source,handles.h_ignore_arrows),
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
      
      tfShift = any(strcmp('shift',event.Modifier));
      tfCtrl = any(strcmp('control',event.Modifier));
      
      isMA = labeler.maIsMA;
      % KB20160724: shortcuts from preferences
      % skip this for MA projs where we need separate hotkey mappings
      if ~isMA && all(isfield(handles,{'shortcutkeys','shortcutfns'}))
        % control key pressed?
        if tfCtrl && numel(event.Modifier)==1 && any(strcmpi(event.Key,handles.shortcutkeys))
          i = find(strcmpi(event.Key,handles.shortcutkeys),1);
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
      
      lcore = labeler.lblCore;
      if ~isempty(lcore)
        tfKPused = lcore.kpf(source,event);
        if tfKPused
          return
        end
      end
      
      %disp(evt);
      if any(strcmp(event.Key,{'leftarrow' 'rightarrow'}))
        switch event.Key
          case 'leftarrow'
            if tfShift
              sam = labeler.movieShiftArrowNavMode;
              samth = labeler.movieShiftArrowNavModeThresh;
              samcmp = labeler.movieShiftArrowNavModeThreshCmp;
              [tffound,f] = sam.seekFrame(labeler,-1,samth,samcmp);
              if tffound
                labeler.setFrameProtected(f);
              end
            else
              labeler.frameDown(tfCtrl);
            end
            tfKPused = true;
          case 'rightarrow'
            if tfShift
              sam = labeler.movieShiftArrowNavMode;
              samth = labeler.movieShiftArrowNavModeThresh;
              samcmp = labeler.movieShiftArrowNavModeThreshCmp;
              [tffound,f] = sam.seekFrame(labeler,1,samth,samcmp);
              if tffound
                labeler.setFrameProtected(f);
              end
            else
              labeler.frameUp(tfCtrl);
            end
            tfKPused = true;
        end
        return
      end
      
      if labeler.gtIsGTMode && strcmp(event.Key,{'r'})
        labeler.gtNextUnlabeledUI();
        return
      end
    end  % function
          
    function menu_file_quick_open_actuated_(obj, source, event)  %#ok<INUSD>
      lObj = obj.labeler_ ;
      if obj.raiseUnsavedChangesDialogIfNeeded() ,
        [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(false);
        if ~tfsucc
          return;
        end
        
        movfile = movfile{1};
        trxfile = trxfile{1};
        
        cfg = Labeler.cfgGetLastProjectConfigNoView() ;
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
        
        [~,projName,~] = fileparts(movfile);
        cfg.ProjectName = projName ;
        lObj.projNew(cfg);
        lObj.movieAdd(movfile,trxfile);
        lObj.movieSet(1,'isFirstMovie',true);      
      end
    end  % function
    
    function projAddLandmarks(obj, nadd)
      % Function to add new kinds of landmarks to an existing project.  E.g. If you
      % had a fly .lbl file where you weren't tracking the wing tips, but then you
      % wanted to start tracking the wingtips, you would call this function.
      % Currently not exposed in the GUI, probably should be eventually.

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure);

      if labeler.nLabelPointsAdd > 0,
        warndlg('Cannot add more landmarks twice in a row. If there are no more partially labeled frames, run projAddLandmarksFinished() to finish.', ...
                'Cannot add landmarks twice');
        return;
      end
            
%       % if labeling mode is sequential, set to template
%       if strcmpi(obj.labelMode,'SEQUENTIAL'),
%         obj.labelingInit('labelMode',LabelMode.TEMPLATE,'dosettemplate',false);
%       end
      
      if labeler.nview>1,
        warning('Adding landmarks for multiview projects not yet tested. Not sure if this will work!!');
      end

      isinit0 = labeler.isinit;
      labeler.isinit = true;
      %delete(obj.lblCore);
      %obj.lblCore = [];
      labeler.preProcData = [];
      labeler.ppdb = [];

      
      oldnphyspts = labeler.nPhysPoints;
      oldnpts = labeler.nLabelPoints;
      nptsperset = size(labeler.labeledposIPtSetMap,2);

      newnphyspts = oldnphyspts+nadd;
      newnpts = oldnpts + nadd*nptsperset;
           
      % update landmark info
      
      % landmark names - one per set
      newnames = Labeler.defaultLandmarkNames(oldnphyspts+1:oldnphyspts+nadd);
      labeler.skelNames = cat(1,labeler.skelNames,newnames);
      
      % pt2set
      oldipt2set = reshape(labeler.labeledposIPt2Set,[oldnphyspts,nptsperset]);
      newipt2set = repmat(oldnphyspts+(1:nadd)',[1,nptsperset]);
      labeler.labeledposIPt2Set = reshape(cat(1,oldipt2set,newipt2set),[newnpts,1]);
      
      % pt2view
      oldipt2view = reshape(labeler.labeledposIPt2View,[oldnphyspts,nptsperset]);
      newipt2view = repmat(1:nptsperset,[nadd,1]);
      labeler.labeledposIPt2View = reshape(cat(1,oldipt2view,newipt2view),[newnpts,1]);
      
      % this is changing for existing points if nview > 1
      labeler.labeledposIPtSetMap = reshape(1:newnpts,[newnphyspts,nptsperset]);
      old2newpt = reshape(labeler.labeledposIPtSetMap(1:oldnphyspts,:),[oldnpts,1]);
      [~,new2oldpt] = ismember((1:newnpts)',old2newpt);
      
      % update labels
      labeler.labelPosAddLandmarks(new2oldpt);

      % skeletonEdges and flipLandmarkMatches should not change
      
      labeler.nLabelPoints = newnpts;
      labeler.nLabelPointsAdd = nadd*nptsperset;
      
      % reset colors to defaults
      labeler.labelPointsPlotInfo.Colors = feval(labeler.labelPointsPlotInfo.ColorMapName,newnphyspts);
      labeler.predPointsPlotInfo.Colors = feval(labeler.predPointsPlotInfo.ColorMapName,newnphyspts);
      labeler.impPointsPlotInfo.Colors = feval(labeler.impPointsPlotInfo.ColorMapName,newnphyspts);

      % reset reference frame plotting
      labeler.genericInitLabelPointViz('lblPrev_ptsH','lblPrev_ptsTxtH',...
                                       handles.axes_prev,labeler.labelPointsPlotInfo);
      if ~isempty(labeler.prevAxesModeInfo)
        labeler.prevAxesLabelsRedraw();
      end
      
      % remake info timeline
%      handles.labelTLInfo.delete();
%       handles.labelTLInfo = InfoTimeline(obj,handles.axes_timeline_manual,...
%         handles.axes_timeline_islabeled);
      handles.labelTLInfo.initNewProject();
      handles.labelTLInfo.setLabelsFull(true);
      guidata(mainFigure,handles);
      
      % clear tracking data
      cellfun(@(x)x.clearTracklet(),labeler.labels2);
      cellfun(@(x)x.clearTracklet(),labeler.labels2GT);
            
      % Clear all the trained trackers
      labeler.clearAllTrackers();
      % Trackers created/initted in projLoad and projNew; eg when loading,
      % the loaded .lbl knows what trackers to create.
      
      labeler.trackDLBackEnd = DLBackEndClass() ;  % This seems...  odd?  -- ALT, 2025-02-03
      labeler.trackDLBackEnd.isInAwsDebugMode = labeler.isInAwsDebugMode ;
      % not resetting trackParams, hopefully nothing in here that depends
      % on number of landmarks
      %obj.trackParams = [];
      
      labeler.labeledposNeedsSave = true;
      labeler.doesNeedSave_ = true;     
      
      labeler.lblCore.init(newnphyspts,labeler.labelPointsPlotInfo);
      labeler.preProcInit();
      labeler.isinit = isinit0;
      labeler.labelsUpdateNewFrame(true);
      set(handles.menu_setup_sequential_add_mode,'Visible','on');
    end  % function
    
    function projAddLandmarksFinished(obj)
      % Used in conjunction with projAddLandmarks().  Function to finish a session
      % of adding new kinds of landmarks to an existing project. Currently not
      % exposed in the GUI, probably should be eventually.
      
      labeler = obj.labeler_ ;
      if labeler.nLabelPointsAdd == 0,
        return;
      end
      
      % check if there are no partially labeled frames
      [~,~,frms] = labeler.findPartiallyLabeledFrames();
      if numel(frms) > 0,
        warndlg('There are still some partially labeled frames. You must label all partially labeled frames before finishing.', ...
                'Not all frames completely labeled');
        return;
      end
      
      labeler.nLabelPointsAdd = 0;
      
      % set label mode to sequential if sequential add
      if labeler.labelMode == LabelMode.SEQUENTIALADD,
        labeler.labelingInit('labelMode',LabelMode.SEQUENTIAL);
      end
      % hide sequential add mode
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      set(handles.menu_setup_sequential_add_mode,'Visible','off');
    end  % function
    
    function [tfok,rawtrkname] = getExportTrkRawNameUI(obj, varargin)
      % Prompt the user to get a raw/base trkfilename.
      %
      % varargin: see defaultExportTrkRawname
      % 
      % tfok: user canceled or similar
      % rawtrkname: use only if tfok==true
      
      labeler = obj.labeler_ ;
      rawtrkname = inputdlg(strcatg('Enter name/pattern for trkfile(s) to be exported. Available macros: ', ...
                                    '$movdir, $movfile, $projdir, $projfile, $projname, $trackertype.'),...
                            'Export Trk File',1,{labeler.defaultExportTrkRawname(varargin{:})});
      tfok = ~isempty(rawtrkname);
      if tfok
        rawtrkname = rawtrkname{1};
      end
    end  % function
    
    function initialize_menu_track_tracking_algorithm_(obj)
      % Populate the Track > 'Tracking algorithm' submenu.
      % This only needs to be done when starting a new project or loading a project.

      % Get out the main objects
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % Delete the old submenu items
      old_menu_track_trackers = handles.menu_track_tracking_algorithm.Children ;
      deleteValidGraphicsHandles(old_menu_track_trackers) ;
      
      % Remake the submenu items
      trackers = labeler.trackersAll ;
      trackerCount = numel(trackers) ;
      for i=1:trackerCount  
        algName = trackers{i}.algorithmName;
        algLabel = trackers{i}.algorithmNamePretty;
        enable = onIff(~strcmp(algName,'dpk'));
        uimenu('Parent',handles.menu_track_tracking_algorithm,...
               'Label',algLabel,...
               'Callback',@LabelerGUIControlActuated,...
               'Tag','menu_track_tracking_algorithm_item',...
               'UserData',i,...
               'Enable',enable,...
               'Position',i) ;
      end

      % Update the checkboxes, etc
      obj.update_menu_track_tracking_algorithm_() ;
    end  % function

    function update_menu_track_tracking_algorithm_(obj)
      % Update the Track > 'Tracking algorithm' submenu.
      % This essentially means updating what elements are checked or not.

      % Get out the main objects
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      
      % Remake the submenu items
      menus = handles.menu_track_tracking_algorithm.Children ;
      trackers = labeler.trackersAll ;
      trackerCount = numel(trackers) ;
      isMatch = labeler.doesCurrentTrackerMatchFromTrackersAllIndex() ;
      for i=1:trackerCount
        menu = menus(i) ;
        menuTrackersAllIndex = menu.UserData ;
        menu.Checked = onIff(isMatch(menuTrackersAllIndex)) ;
      end
    end  % function

    function update_menu_track_tracker_history_(obj)
      % Populate the Track > 'Tracking algorithm' submenu.
      % This deletes all the menu items and then remakes them.

      % Get out the main objects
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end      
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % Delete the old submenu items
      menu_track_tracker_history = handles.menu_track_tracker_history ;
      old_submenu_items = menu_track_tracker_history.Children ;
      deleteValidGraphicsHandles(old_submenu_items) ;
      
      % Remake the submenu items
      trackers = labeler.trackerHistory ;
      trackerCount = numel(trackers) ;
      tag = 'menu_track_tracker_history_item' ;
      for i = 1:trackerCount  
        tracker = trackers{i} ;
        algNamePretty = tracker.algorithmNamePretty ;
        rawTrnNameLbl = tracker.trnNameLbl ;
        trnNameLbl = fif(isempty(rawTrnNameLbl), 'untrained', rawTrnNameLbl) ;
        menuItemLabel = sprintf('%s (%s)', algNamePretty, trnNameLbl) ;
        uimenu('Parent',menu_track_tracker_history, ...
               'Label',menuItemLabel, ...
               'Callback',@(s,e)(obj.controlActuated(tag, s, e)), ...
               'Tag',tag, ...
               'UserData',i, ...
               'Position',i, ...
               'Checked', onIff(i==1)) ;  
          % The first element of labeler.trackerHistory is always the current one
      end
    end  % function

    % function update_menu_track_tracker_history_(obj)
    %   % Populate the Track > 'Tracking algorithm' submenu.
    % 
    %   % Get out the main objects
    %   labeler = obj.labeler_ ;
    %   if labeler.isinit ,
    %     return
    %   end      
    %   mainFigure = obj.mainFigure_ ;
    %   handles = guidata(mainFigure) ;
    % 
    %   % Delete the old submenu items
    %   menu_track_tracker_history = handles.menu_track_tracker_history ;
    % 
    %   % Remake the submenu items
    %   menu_items = menu_track_tracker_history.Children ;
    %   menu_item_count = numel(menu_items) ;
    %   index_from_index = (1:menu_item_count) ;
    %   arrayfun(@(i,item)(item.set('Checked', onIff(i==1))), ...
    %            index_from_index, ...
    %            menu_items) ;
    %     % The first element of labeler.trackerHistory is always the current one
    % end  % function

    function initialize_menu_track_backend_config_(obj)
      % Populate the Track > Backend menu

      % Get out the main objects
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      
      if ~isfield(handles,'menu_track_backend_config')
        % set up first time only, should not change
        handles.menu_track_backend_config = uimenu( ...
          'Parent',handles.menu_track,...
          'Label','Backend configuration',...
          'Visible','on',...
          'Tag','menu_track_backend_config');
        moveMenuItemAfter(handles.menu_track_backend_config, handles.menu_track_tracker_history) ;
        handles.menu_track_backend_config_jrc = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','JRC Cluster',...
          'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
          'Tag','menu_track_backend_config_jrc',...
          'userdata',DLBackEnd.Bsub);
        handles.menu_track_backend_config_aws = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','AWS Cloud',...
          'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
          'Tag','menu_track_backend_config_aws',...
          'userdata',DLBackEnd.AWS);
        handles.menu_track_backend_config_docker = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','Docker',...
          'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
          'Tag','menu_track_backend_config_docker',...
          'userdata',DLBackEnd.Docker);  
        handles.menu_track_backend_config_conda = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','Conda',...
          'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
          'Tag','menu_track_backend_config_conda',...
          'userdata',DLBackEnd.Conda,...
          'Visible',true);
        handles.menu_track_backend_config_conda.Enable = 'on';
        % KB added menu item to get more info about how to set up
        handles.menu_track_backend_config_moreinfo = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','on',...
          'Label','More information...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendMenuMoreInfo()),...
          'Tag','menu_track_backend_config_moreinfo');  
        handles.menu_track_backend_config_test = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','Test backend configuration',...
          'Callback',@(s,e)(obj.cbkTrackerBackendTest()),...
          'Tag','menu_track_backend_config_test');
        
        % JRC Cluster 'submenu'
        handles.menu_track_backend_config_jrc_setconfig = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','on',...
          'Label','(JRC) Set number of slots for training...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendSetJRCNSlots()),...
          'Tag','menu_track_backend_config_jrc_setconfig');  
      
        handles.menu_track_backend_config_jrc_setconfig_track = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','off',...
          'Label','(JRC) Set number of slots for tracking...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendSetJRCNSlotsTrack()),...
          'Tag','menu_track_backend_config_jrc_setconfig_track');  
      
        handles.menu_track_backend_config_jrc_additional_bsub_args = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','(JRC) Additional bsub arguments...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendAdditionalBsubArgs()),...
          'Tag','menu_track_backend_config_jrc_additional_bsub_args');  
      
        handles.menu_track_backend_config_jrc_set_singularity_image = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','off',...
          'Label','(JRC) Set Singularity image...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendSetSingularityImage()),...
          'Tag','menu_track_backend_config_jrc_set_singularity_image');  
      
        % AWS submenu (enabled when backend==AWS)
        handles.menu_track_backend_config_aws_configure = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','on',...
          'Label','(AWS) Configure...',...
          'Callback',@LabelerGUIControlActuated,...
          'Tag','menu_track_backend_config_aws_configure');  
      
        handles.menu_track_backend_config_aws_setinstance = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','(AWS) Set EC2 instance',...
          'Callback',@LabelerGUIControlActuated,...
          'Tag','menu_track_backend_config_aws_setinstance');  
        
        % Docker 'submenu' (added by KB)
        handles.menu_track_backend_config_setdockerssh = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','on',...
          'Label','(Docker) Set remote host...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendSetDockerSSH()),...
          'Tag','menu_track_backend_config_setdockerssh');  
      
        handles.menu_track_backend_config_docker_image_spec = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Label','(Docker) Set image spec...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendSetDockerImageSpec()),...
          'Tag','menu_track_backend_config_docker_image_spec');  
      
        % Conda submenu
        handles.menu_track_backend_set_conda_env = uimenu( ...
          'Parent',handles.menu_track_backend_config,...
          'Separator','on', ...
          'Label','(Conda) Set environment...',...
          'Callback',@(s,e)(obj.cbkTrackerBackendSetCondaEnv()),...
          'Tag','menu_track_backend_set_conda_env');        
      end

      % Store the modified handles struct
      guidata(mainFigure, handles) ;      
    end  % function

    function cbkTrackerBackendMenu(obj, source, event)  %#ok<INUSD>
      lObj = obj.labeler_ ;
      %mainFigure = obj.mainFigure_ ;
      %handles = guidata(mainFigure) ;
      beType = source.UserData;
      lObj.set_backend_property('type', beType) ;
    end  % function

    function cbkTrackerBackendMenuMoreInfo(obj)
      lObj = obj.labeler_ ;
      res = web(lObj.DLCONFIGINFOURL,'-new');
      if res ~= 0,
        msgbox({'Information on configuring Deep Learning GPU/Backends can be found at'
                'https://github.com/kristinbranson/APT/wiki/Deep-Neural-Network-Tracking.'},...
                'Deep Learning GPU/Backend Information','replace');
      end
    end  % function

    function cbkTrackerBackendTest(obj)
      lObj = obj.labeler_ ;
      cacheDir = lObj.DLCacheDir;
      assert(exist(cacheDir,'dir'),...
             'Deep Learning cache directory ''%s'' does not exist.',cacheDir);
      backend = lObj.trackDLBackEnd;
      testBackendConfigUI(backend, cacheDir);
    end  % function
      
    function cbkTrackerBackendSetJRCNSlots(obj)
      lObj = obj.labeler_ ;
      n = inputdlg('Number of cluster slots for training','a',1,{num2str(lObj.trackDLBackEnd.jrcnslots)});
      if isempty(n)
        return
      end
      n = str2double(n{1});
      if isnan(n)
        return
      end
      lObj.trackDLBackEnd.jrcnslots = n ;
    end  % function

    function cbkTrackerBackendSetJRCNSlotsTrack(obj)
      lObj = obj.labeler_ ;
      n = inputdlg('Number of cluster slots for tracking','a',1,{num2str(lObj.trackDLBackEnd.jrcnslotstrack)});
      if isempty(n)
        return
      end
      n = str2double(n{1});
      if isnan(n)
        return
      end
      lObj.trackDLBackEnd.jrcnslotstrack = n ;
    end  % function
    
    function cbkTrackerBackendAdditionalBsubArgs(obj)
      lObj = obj.labeler_ ;
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
    end  % function

    function cbkTrackerBackendSetSingularityImage(obj)
      lObj = obj.labeler_ ;
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
    end  % function

    function cbkCurrTrackerChanged(obj)
      % Update the controls that need to be updated after the current tracker
      % changes

      % Get the objects we need to mess with
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end 
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      tracker = labeler.tracker ;

      % Enable/disable controls that depend on whether a tracker is available
      tfTracker = ~isempty(tracker) ;
      onOrOff = onIff(tfTracker && labeler.isReady) ;
      handles.menu_track.Enable = onOrOff;
      handles.pbTrain.Enable = onOrOff;
      handles.pbTrack.Enable = onOrOff;
      handles.menu_view_hide_predictions.Enable = onOrOff;
      handles.menu_view_show_preds_curr_target_only.Enable = onOrOff;

      % % Remake the tracker history submenu
      % obj.update_menu_track_tracker_history_() ;

      % Update the check marks in menu_track_backend_condfig menu
      obj.update_menu_track_backend_config_();

      % Update the InfoTimeline
      handles.labelTLInfo.didChangeCurrentTracker();

      % Update the guidata
      guidata(mainFigure, handles) ;
    end  % function
    
    % function cbkTrackerShowVizReplicatesChanged(obj)
    %   lObj = obj.labeler_ ;
    %   mainFigure = obj.mainFigure_ ;
    %   handles = guidata(mainFigure) ;
    %   handles.menu_track_cpr_show_replicates.Checked = onIff(lObj.tracker.showVizReplicates) ;
    % end  % function
    
    % function cbkTrackerStoreFullTrackingChanged(obj)
    %   labeler = obj.labeler_ ;
    %   mainFigure = obj.mainFigure_ ;
    %   handles = guidata(mainFigure) ;
    %   sft = labeler.tracker.storeFullTracking;
    %   switch sft
    %     case StoreFullTrackingType.NONE
    %       handles.menu_track_cpr_storefull_dont_store.Checked = 'on';
    %       handles.menu_track_cpr_storefull_store_final_iteration.Checked = 'off';
    %       handles.menu_track_cpr_storefull_store_all_iterations.Checked = 'off';
    %       handles.menu_track_cpr_view_diagnostics.Enable = 'off';
    %     case StoreFullTrackingType.FINALITER
    %       handles.menu_track_cpr_storefull_dont_store.Checked = 'off';
    %       handles.menu_track_cpr_storefull_store_final_iteration.Checked = 'on';
    %       handles.menu_track_cpr_storefull_store_all_iterations.Checked = 'off';
    %       handles.menu_track_cpr_view_diagnostics.Enable = 'on';
    %     case StoreFullTrackingType.ALLITERS
    %       handles.menu_track_cpr_storefull_dont_store.Checked = 'off';
    %       handles.menu_track_cpr_storefull_store_final_iteration.Checked = 'off';
    %       handles.menu_track_cpr_storefull_store_all_iterations.Checked = 'on';
    %       handles.menu_track_cpr_view_diagnostics.Enable = 'on';
    %     otherwise
    %       assert(false);
    %   end
    % end  % function
    
    function cbkTrackerTrainStart(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      algName = lObj.tracker.algorithmName;
      %algLabel = lObj.tracker.algorithmNamePretty;
      backend_type_string = lObj.trackDLBackEnd.prettyName();
      handles.txBGTrain.String = sprintf('%s training on %s (started %s)',algName,backend_type_string,datestr(now(),'HH:MM'));
      handles.txBGTrain.ForegroundColor = handles.busystatuscolor;
      handles.txBGTrain.FontWeight = 'normal';
      handles.txBGTrain.Visible = 'on';
    end  % function

    function cbkTrackerTrainEnd(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      handles.txBGTrain.Visible = 'off';
      handles.txBGTrain.String = 'Idle';
      handles.txBGTrain.ForegroundColor = handles.idlestatuscolor;
      val = true;
      str = 'Tracker trained';
      lObj.setDoesNeedSave(val, str) ;
    end  % function

    function cbkTrackerStart(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      algName = lObj.tracker.algorithmName;
      %algLabel = lObj.tracker.algorithmNamePretty;
      backend_type_string = lObj.trackDLBackEnd.prettyName() ;
      handles.txBGTrain.String = sprintf('%s tracking on %s (started %s)',algName,backend_type_string,datestr(now,'HH:MM'));
      handles.txBGTrain.ForegroundColor = handles.busystatuscolor;
      handles.txBGTrain.FontWeight = 'normal';
      handles.txBGTrain.Visible = 'on';
    end  % function

    function cbkTrackerEnd(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      handles.txBGTrain.Visible = 'off';
      handles.txBGTrain.String = 'Idle';
      handles.txBGTrain.ForegroundColor = handles.idlestatuscolor;
      val = true;
      str = 'New frames tracked';
      lObj.setDoesNeedSave(val, str) ;
    end  % function

    function cbkTrackerHideVizChanged(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;      
      tracker = lObj.tracker ;
      handles.menu_view_hide_predictions.Checked = onIff(tracker.hideViz) ;
    end  % function

    function cbkTrackerShowPredsCurrTargetOnlyChanged(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      tracker = lObj.tracker ;
      handles.menu_view_show_preds_curr_target_only.Checked = onIff(tracker.showPredsCurrTargetOnly) ;
    end  % function

    function update_menu_track_backend_config_(obj)
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      if ~isfield(handles, 'menu_track_backend_config_jrc') 
        % Early return if the menus have not been set up yet
        return
      end      
      beType = labeler.trackDLBackEnd.type;
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
    end  % function
    
    function cbkTrackerBackendSetDockerSSH(obj)
      lObj = obj.labeler_ ;
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
    end  % function
    
    function cbkTrackerBackendSetDockerImageSpec(obj)
      lObj = obj.labeler_ ;      
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
    end  % function
    
    function cbkTrackerBackendSetCondaEnv(obj)
      lObj = obj.labeler_ ;      
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
    end  % function

    function cbkLastLabelChangeTS(obj)
      % when lastLabelChangeTS is updated, update the tracker info text in the main APT window
      lObj = obj.labeler_ ;      
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      if ~isempty(lObj.tracker) ,
        handles.text_trackerinfo.String = lObj.tracker.getTrackerInfoString() ;
      end
    end  % function
    
    function cbkParameterChange(obj)
      lObj = obj.labeler_ ;      
      if isempty(lObj.tracker) ,
        return
      end
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      handles.text_trackerinfo.String = lObj.tracker.getTrackerInfoString() ;
    end  % function
    
    function initTblFrames_(obj)
      % Initialize the uitable of labeled frames in the 'Labeled Frames' window.

      labeler = obj.labeler_ ;      
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tbl0 = handles.tblFrames ;
      isMA = labeler.maIsMA ;

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

      set(tbl0,...
        'ColumnWidth',COLWIDTH,...
        'ColumnName',COLNAMES,...
        'Data',cell(0,numel(COLNAMES)),...
        'FontUnits','points',...
        'FontSize',9.75,... % matches .tblTrx
        'BackgroundColor',[.3 .3 .3; .45 .45 .45]);
    end  % function
    
    function tfAxLimsSpecifiedInCfg = hlpSetConfigOnViews_(obj, viewCfg, centerOnTarget)
      % Configure the figures and axes showing the different views of the animal(s)
      % according to the specification in viewCfg.

      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      axes_all = handles.axes_all;
      tfAxLimsSpecifiedInCfg = ...
        ViewConfig.setCfgOnViews(viewCfg, ...
                                 handles.figs_all, ...
                                 axes_all, ...
                                 handles.images_all, ...
                                 handles.axes_prev) ;
      if ~centerOnTarget
        [axes_all.CameraUpVectorMode] = deal('auto');
        [axes_all.CameraViewAngleMode] = deal('auto');
        [axes_all.CameraTargetMode] = deal('auto');
        [axes_all.CameraPositionMode] = deal('auto');
      end
      [axes_all.DataAspectRatio] = deal([1 1 1]);
      handles.menu_view_show_tick_labels.Checked = onIff(~isempty(axes_all(1).XTickLabel));
      handles.menu_view_show_grid.Checked = axes_all(1).XGrid;
    end  % function
    
    function cbkAuxFigCloseReq(controller, src, evt)  %#ok<INUSD>
      if ~controller.isSatellite(src) 
        delete(src);
        return  
      end
      
      CLOSESTR = 'Close anyway';
      DONTCLOSESTR = 'Cancel, don''t close';
      tfbatch = batchStartupOptionUsed() ; % ci
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
          delete(src)
      end    
    end   % function

    function menu_track_reset_current_tracker_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      response = ...
        questdlg(strcatg('Reset current tracker? This will clear your trained tracker, along with all tracking results. ', ...
                         'Hit cancel if you do not want to do this.'), ...
                 'Reset current tracker?', ...
                 'Reset', 'Cancel', ...
                 'Cancel') ;
      if strcmpi(response, 'Reset') ,
        labeler.resetCurrentTracker() ;
      else
        return
      end
    end  % function

    function menu_track_delete_current_tracker_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      response = ...
        questdlg(strcatg('Delete current tracker? This will clear your trained tracker, along with all tracking results. ', ...
                         'Hit cancel if you do not want to do this.'), ...
                 'Delete current tracker?', ...
                 'Delete', 'Cancel', ...
                 'Cancel') ;
      if strcmpi(response, 'Delete') ,
        labeler.deleteCurrentTracker() ;
      else
        return
      end
    end  % function

    function menu_track_delete_old_trackers_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      response = ...
        questdlg(strcatg('Delete old trackers? This will clear all trained trackers except the current one, along with all tracking results. ', ...
                         'Hit cancel if you do not want to do this.'), ...
                 'Delete old trackers?', ...
                 'Delete', 'Cancel', ...
                 'Cancel') ;
      if strcmpi(response, 'Delete') ,
        labeler.deleteOldTrackers() ;
      else
        return
      end
    end  % function

    function cbkCurrTargetChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      if (labeler.hasTrx || labeler.maIsMA) && ~labeler.isinit ,
        iTgt = labeler.currTarget;
        labeler.currImHud.updateTarget(iTgt);
        handles.labelTLInfo.newTarget();
        labeler.hlpGTUpdateAxHilite();
      end
    end  % function

    function cbkLabelModeChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      lblMode = labeler.labelMode;
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
          handles.menu_setup_ma_twoclick_align.Checked = labeler.isTwoClickAlign;
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
    end  % function

    function cbkLabels2HideChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      handles.menu_view_hide_imported_predictions.Checked = onIff(labeler.labels2Hide);
    end  % function

    function cbkLabels2ShowCurrTargetOnlyChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      handles.menu_view_show_imported_preds_curr_target_only.Checked = ...
        onIff(labeler.labels2ShowCurrTargetOnly);
    end  % function

    function cbkShowTrxChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      onOff = onIff(~labeler.showTrx);
      handles.menu_view_hide_trajectories.Checked = onOff;
    end  % function

    function cbkShowOccludedBoxChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      onOff = onIff(labeler.showOccludedBox);
      handles.menu_view_occluded_points_box.Checked = onOff;
      set([handles.text_occludedpoints,handles.axes_occ],'Visible',onOff);
    end  % function

    function cbkShowTrxCurrTargetOnlyChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      onOff = onIff(labeler.showTrxCurrTargetOnly);
      handles.menu_view_plot_trajectories_current_target_only.Checked = onOff;
    end  % function

    function cbkTrackModeIdxChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      if labeler.isinit ,
        return
      end
      pumTrack = handles.pumTrack;
      pumTrack.Value = labeler.trackModeIdx;
      try %#ok<TRYNC>
        %fullstrings = getappdata(pumTrack,'FullStrings');
        fullstrings = obj.pumTrackFullStrings_ ;
        set(handles.text_framestotrackinfo,'String',fullstrings{pumTrack.Value});
      end
      % Edge case: conceivably, pumTrack.Strings may not be updated (eg for a
      % noTrx->hasTrx transition before this callback fires). In this case,
      % hPUM.Value (trackModeIdx) will be out of bounds and a warning till be
      % thrown, PUM will not be displayed etc. However when hPUM.value is
      % updated, this should resolve.
    end  % function

    function cbkTrackerNFramesChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      if labeler.isinit ,
        return
      end
      obj.setPUMTrackStrs_() ;
    end  % function

    function setPUMTrackStrs_(obj)
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      if labeler.hasTrx
        mfts = MFTSetEnum.TrackingMenuTrx;
      else
        mfts = MFTSetEnum.TrackingMenuNoTrx;
      end
      menustrs = arrayfun(@(x)x.getPrettyStr(labeler.getMftInfoStruct()),mfts,'uni',0);
      if ispc() || ismac()
        menustrs_compact = arrayfun(@(x)x.getPrettyStrCompact(labeler.getMftInfoStruct()),mfts,'uni',0);
      else
        % iss #161
        menustrs_compact = arrayfun(@(x)x.getPrettyStrMoreCompact(labeler.getMftInfoStruct()),mfts,'uni',0);
      end
      pumTrack = handles.pumTrack;
      pumTrack.String = menustrs_compact;
      %setappdata(pumTrack,'FullStrings',menustrs);
      obj.pumTrackFullStrings_ = menustrs ;
      if labeler.trackModeIdx>numel(menustrs)
        labeler.trackModeIdx = 1;
      end
      obj.resize() ;
    end  % function

    function cbkMovieCenterOnTargetChanged(obj, src, evt)   %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      tf = labeler.movieCenterOnTarget;
      mnu = handles.menu_view_trajectories_centervideoontarget;
      mnu.Checked = onIff(tf);
      if tf,
        obj.videoZoom(labeler.targetZoomRadiusDefault);
      end
    end  % function

    function cbkMovieRotateTargetUpChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      tf = labeler.movieRotateTargetUp;
      if tf
        ax = handles.axes_curr;
        warnst = warning('off','LabelerGUI:axDir');
        % When axis is in image mode, ydir should be reversed!
        ax.XDir = 'normal';
        ax.YDir = 'reverse';
        warning(warnst);
      end
      mnu = handles.menu_view_rotate_video_target_up;
      mnu.Checked = onIff(tf);
      labeler.UpdatePrevAxesDirections();
    end  % function
    
    function cbkMovieForceGrayscaleChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;       
      handles = guidata(mainFigure) ;
      tf = labeler.movieForceGrayscale;
      mnu = handles.menu_view_converttograyscale;
      mnu.Checked = onIff(tf);
    end  % function

    function cbkMovieViewBGsubbedChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      tf = labeler.movieViewBGsubbed;
      mnu = handles.menu_view_show_bgsubbed_frames;
      mnu.Checked = onIff(tf);
    end  % function

    function cbkGtIsGTModeChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      gt = labeler.gtIsGTMode;
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
      hlpGTUpdateAxHilite(labeler);
    end

    function cbkCropIsCropModeChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      %mainFigure = obj.mainFigure_ ;  
      %handles = guidata(mainFigure) ;
      labeler.setStatus('Switching crop mode...');
      obj.cropReactNewCropMode_(labeler.cropIsCropMode);
      if labeler.hasMovie
        labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
      end
      labeler.clearStatus();
    end  % function

    function cbkUpdateCropGUITools(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      obj.cropReactNewCropMode_(labeler.cropIsCropMode) ;
    end  % function
    
    function cbkCropCropsChanged(obj, src, evt)  %#ok<INUSD>
      % labeler = obj.labeler_ ;       
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
      obj.cropUpdateCropHRects_();
    end  % function

    function cbkNewMovie(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      %gt = labeler.gtIsGTMode;

      %movRdrs = labeler.movieReader;
      %ims = arrayfun(@(x)x.readframe(1),movRdrs,'uni',0);
      hAxs = handles.axes_all;
      hIms = handles.images_all; % Labeler has already loaded with first frame
      assert(isequal(labeler.nview,numel(hAxs),numel(hIms)));

      tfResetAxLims = evt.isFirstMovieOfProject || labeler.movieRotateTargetUp;
      tfResetAxLims = repmat(tfResetAxLims,labeler.nview,1);
      if isfield(handles,'newProjAxLimsSetInConfig')
        % AL20170520 Legacy projects did not save their axis lims in the .lbl
        % file.
        tfResetAxLims = tfResetAxLims | ~handles.newProjAxLimsSetInConfig;
        handles = rmfield(handles,'newProjAxLimsSetInConfig');
      end

      if labeler.hasMovie && evt.isFirstMovieOfProject,
        handles.controller.enableControls_('projectloaded');
      end

      if ~labeler.gtIsGTMode,
        set(handles.menu_go_targets_summary,'Enable','on');
      else
        set(handles.menu_go_targets_summary,'Enable','off');
      end

      wbmf = @(src,evt)(obj.cbkWBMF(src,evt));
      wbuf = @(src,evt)(obj.cbkWBUF(src,evt));
      movnr = labeler.movienr;
      movnc = labeler.movienc;
      figs = handles.figs_all;
      if labeler.hasMovie
        % guard against callback during new proj creation etc; labeler.movienc/nr
        % are NaN which creates a badly-inited imgzoompan. Theoretically seems
        % this wouldn't matter as the next imgzoompan created (when movie
        % actually added) should be properly initted...
        for ivw=1:labeler.nview
          set(figs(ivw),'WindowScrollWheelFcn',@(src,evt)(obj.scroll_callback(src,evt)));
          set(figs(ivw),'WindowButtonMotionFcn',wbmf,'WindowButtonUpFcn',wbuf);

          [hascrop,cropInfo] = labeler.cropGetCropCurrMovie();
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
      for iView = 1:labeler.nview
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

      nframes = labeler.nframes;
      sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
      set(handles.slider_frame,'Value',0,'SliderStep',sliderstep);

      tfHasMovie = labeler.currMovie>0;
      if tfHasMovie
        minzoomrad = 10;
        maxzoomrad = (labeler.movienc(1)+labeler.movienr(1))/4;
        handles.sldZoom.UserData = log([minzoomrad maxzoomrad]);
      end

      TRX_MENUS = {...
        'menu_view_trajectories_centervideoontarget'
        'menu_view_rotate_video_target_up'
        'menu_view_hide_trajectories'
        'menu_view_plot_trajectories_current_target_only'};
      %  'menu_setup_label_overlay_montage_trx_centered'};
      tftblon = labeler.hasTrx || labeler.maIsMA;
      onOff = onIff(tftblon);
      cellfun(@(x)set(handles.(x),'Enable',onOff),TRX_MENUS);
      hTbl = handles.tblTrx;
      set(hTbl,'Enable',onOff);
      guidata(handles.figure, handles);

      obj.setPUMTrackStrs_() ;

      % See note in AxesHighlightManager: Trx vs noTrx, Axes vs Panels
      handles.allAxHiliteMgr.setHilitePnl(labeler.hasTrx);

      hlpGTUpdateAxHilite(labeler);

      if labeler.cropIsCropMode
        cropUpdateCropHRects(handles);
      end
      handles.menu_file_crop_mode.Enable = onIff(~labeler.hasTrx);

      % update HUD, statusbar
      mname = labeler.moviename;
      if labeler.nview>1
        movstr = 'Movieset';
      else
        movstr = 'Movie';
      end
      if labeler.gtIsGTMode
        str = sprintf('%s %d (GT): %s',movstr,labeler.currMovie,mname);
      else
        str = sprintf('%s %d: %s',movstr,labeler.currMovie,mname);
      end
      set(handles.txMoviename,'String',str);

      % by default, use calibration if there is calibration for this movie
      lc = labeler.lblCore;
      if ~isempty(lc) && lc.supportsCalibration,
        handles.menu_setup_use_calibration.Checked = onIff(lc.isCalRig && lc.showCalibration);
      end
    end  % function

    function cbkDataImported(obj, src, evt)  %#ok<INUSD>
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      handles.labelTLInfo.newTarget(); % Using this as a "refresh" for now
    end  % function

    function cbkShowSkeletonChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      hasSkeleton = ~isempty(labeler.skeletonEdges) ;
      isChecked = onIff(hasSkeleton && labeler.showSkeleton) ;
      set(handles.menu_view_showhide_skeleton, 'Enable', hasSkeleton, 'Checked', isChecked) ;
    end  % function

    function cbkShowMaRoiChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      onOff = onIff(labeler.showMaRoi);
      handles.menu_view_showhide_maroi.Checked = onOff;
    end  % function

    function cbkShowMaRoiAuxChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      onOff = onIff(labeler.showMaRoiAux);
      handles.menu_view_showhide_maroiaux.Checked = onOff;
    end  % function
    
    function initializeResizeInfo_(obj)
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;

      % Record the width of txUnsavedChanges, so we can keep it fixed
      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      pxTxUnsavedChangesWidth = hTx.Position(3);
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;

      obj.pxTxUnsavedChangesWidth_ = pxTxUnsavedChangesWidth ;
    end
    
    function resize(obj)
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;

      % Take steps to keep right edge of unsaved changes text box aligned with right
      % edge of the previous/reference frame panel
      pxTxUnsavedChangesWidth = obj.pxTxUnsavedChangesWidth_ ;
      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1) + hPnlPrev.Position(3) ;
      hTx.Position(1) = uiPnlPrevRightEdge - pxTxUnsavedChangesWidth ;
      hTx.Position(3) = pxTxUnsavedChangesWidth ;
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      %obj.updateStatus() ;  % do we need this here?
    end
    
    function cropReactNewCropMode_(obj, tf)
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;

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

      obj.cropUpdateCropHRects_() ;
      obj.cropUpdateCropAdjustingCropSize_(false) ;
    end
    
    function cropUpdateCropHRects_(obj)
      % Update handles.cropHRect from lObj.cropIsCropMode, lObj.currMovie and
      % lObj.movieFilesAll*cropInfo
      %
      % rect props set:
      % - position
      % - visibility, pickableparts
      %
      % rect props NOT set:
      % - resizeability.

      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;

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
          cropImRectSetPosnNoPosnBang(h,CropInfo.roi2RectPos(roi(ivw,:)));
          set(h,'Visible','on','PickableParts','all');
          fcn = makeConstrainToRectFcn('imrect',[1 imnc(ivw)],[1 imnr(ivw)]);
          h.setPositionConstraintFcn(fcn);
        end
      else
        arrayfun(@(rect)cropImRectSetPosnNoPosnBang(rect,[nan nan nan nan]),...
                 handles.cropHRect);
        arrayfun(@(x)set(x,'Visible','off','PickableParts','none'),handles.cropHRect);
      end
    end

    function cropUpdateCropAdjustingCropSize_(obj, tfAdjust)
      % cropUpdateCropAdjustingCropSize(handles) --
      %   update .cropHRects.resizeable based on tbAdjustCropSize
      % cropUpdateCropAdjustingCropSize(handles,tfCropMode) --
      %   update .cropHRects.resizeable and tbAdjustCropSize based on tfAdjust

      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;

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
    end
    
    function cbkWBMF(obj, src, evt)
      labeler = obj.labeler_ ;      
      lcore = labeler.lblCore;
      if ~isempty(lcore)
        lcore.wbmf(src,evt);
      end
    end
    
    function cbkWBUF(obj, src, evt)
      labeler = obj.labeler_ ;      
      if ~isempty(labeler.lblCore)
        labeler.lblCore.wbuf(src,evt);
      end
    end
    
    function scroll_callback(obj, hObject, eventdata)
      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      ivw = find(hObject==handles.figs_all);
      ax = handles.axes_all(ivw);
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
      him = handles.images_all(ivw);
      imglimx = get(him,'XData');
      imglimy = get(him,'YData');
      xlim(1) = max(imglimx(1),curp(1,1)-(curp(1,1)-xlim(1))/scrl);
      xlim(2) = min(imglimx(2),curp(1,1)+(xlim(2)-curp(1,1))/scrl);
      ylim(1) = max(imglimy(1),curp(1,2)-(curp(1,2)-ylim(1))/scrl);
      ylim(2) = min(imglimy(2),curp(1,2)+(ylim(2)-curp(1,2))/scrl);
      axis(ax,[xlim(1),xlim(2),ylim(1),ylim(2)]);
      % fprintf('Scrolling %d!!\n',eventdata.VerticalScrollAmount)
    end

    function closeImContrast(obj, iAxRead, iAxApply)
      % ReadClim from axRead and apply to axApply

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      axAll = handles.axes_all;
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
      		set(handles.axes_prev,'CLim',clim);
      	end
      end
    end

    function [tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(obj)

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
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
    end
    
    function hlpRemoveFocus(obj)
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

      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      uicontrol(handles.txStatus);
    end

    function tblFrames_cell_selected_(obj, src, evt)
      labeler = obj.labeler_ ;
      %mainFigure = obj.mainFigure_ ;  
      %handles = guidata(mainFigure) ;
      row = evt.Indices;
      if ~isempty(row)
        row = row(1);
        dat = get(src,'Data');
        labeler.setFrame(dat{row,1},'changeTgtsIfNec',true);
      end
      obj.hlpRemoveFocus() ;
    end

    function axescurrXLimChanged(obj, hObject, eventdata)  %#ok<INUSD>
      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
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
    end

    function axescurrXDirChanged(obj, hObject, eventdata)  %#ok<INUSD>
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      videoRotateTargetUpAxisDirCheckWarn(handles);
    end

    function axescurrYDirChanged(obj, hObject, eventdata)  %#ok<INUSD>
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      videoRotateTargetUpAxisDirCheckWarn(handles);
    end
    
    function cbkPostZoom(obj,src,evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      if evt.Axes == handles.axes_prev,
        labeler.UpdatePrevAxesLimits();
      end
    end

    function cbkPostPan(obj,src,evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      if evt.Axes == handles.axes_prev,
        labeler.UpdatePrevAxesLimits();
      end
    end

    function cbklabelTLInfoSelectOn(obj, src, evt)  %#ok<INUSD>
      % labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      lblTLObj = evt.AffectedObject;
      tb = handles.tbTLSelectMode;
      tb.Value = lblTLObj.selectOn;
    end

    function cbklabelTLInfoPropsUpdated(obj, src, evt)  %#ok<INUSD>
      % Update the props dropdown menu and timeline.
      % labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      labelTLInfo = evt.AffectedObject;
      props = labelTLInfo.getPropsDisp();
      set(handles.pumInfo,'String',props);
    end

    function cbklabelTLInfoPropTypesUpdated(obj, src, evt)  %#ok<INUSD>
      % Update the props dropdown menu and timeline.
      % labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      labelTLInfo = evt.AffectedObject;
      proptypes = labelTLInfo.getPropTypesDisp();
      set(handles.pumInfo_labels,'String',proptypes);
    end
    
    function menuSetupLabelModeCbkGeneric(obj, src, evt)  %#ok<INUSD>
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      lblMode = handles.setupMenu2LabelMode.(src.Tag);
      handles.labeler.labelingInit('labelMode',lblMode);
    end
    
    function figure_CloseRequestFcn(obj, src, evt)  %#ok<INUSD>
      obj.quitRequested() ;
    end

    function videoZoom(obj,zoomRadius)
      % Zoom to square window over current frame center with given radius.
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      [x0,y0] = labeler.videoCurrentCenter();
      lims = [x0-zoomRadius,x0+zoomRadius,y0-zoomRadius,y0+zoomRadius];
      axis(handles.axes_curr,lims);
    end    

    function [xsz,ysz] = videoCurrentSize(obj)
      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      v = axis(handles.axes_curr);
      xsz = v(2)-v(1);
      ysz = v(4)-v(3);
    end

    function [x0,y0] = videoCurrentCenter(obj)
      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      %v = axis(handles.axes_curr);
      x0 = mean(get(handles.axes_curr,'XLim'));
      y0 = mean(get(handles.axes_curr,'YLim'));
    end

    function v = videoCurrentAxis(obj)
      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      v = axis(handles.axes_curr);
    end

    function videoSetAxis(obj,lims,resetcamera)
      if nargin<3
        resetcamera = true;
      end
      %labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      % resets camera view too
      ax = handles.axes_curr;
      if resetcamera
        ax.CameraUpVector = [0, -1,0];
        ax.CameraUpVectorMode = 'auto';
        ax.CameraViewAngleMode = 'auto';
        ax.CameraPositionMode = 'auto';
        ax.CameraTargetMode = 'auto';
      end
      axis(ax,lims);
    end

    function videoCenterOn(obj,x,y)
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      [xsz,ysz] = labeler.videoCurrentSize();
      lims = [x-xsz/2,x+xsz/2,y-ysz/2,y+ysz/2];
      axis(handles.axes_curr,lims);      
    end
    
    function xy = videoClipToVideo(obj,xy)
      % Clip coords to video size.
      %
      % xy (in): [nx2] xy-coords
      %
      % xy (out): [nx2] xy-coords, clipped so that x in [1,nc] and y in [1,nr]
      
      labeler = obj.labeler_ ;
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
      
      xy = CropInfo.roiClipXY(labeler.movieroi,xy);
    end

    function dxdy = videoCurrentUpVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "up" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 
      
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      ax = handles.axes_curr;
      if labeler.hasTrx && labeler.movieRotateTargetUp
        v = ax.CameraUpVector; % should be norm 1
        dxdy = v(1:2);
      else
        dxdy = [0 -1];
      end
    end

    function dxdy = videoCurrentRightVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "right" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      ax = handles.axes_curr;
      if labeler.hasTrx && labeler.movieRotateTargetUp
        v = ax.CameraUpVector; % should be norm 1
        parity = mod(strcmp(ax.XDir,'normal') + strcmp(ax.YDir,'normal'),2);
        if parity
          dxdy = [-v(2) v(1)]; % etc
        else
          dxdy = [v(2) -v(1)]; % rotate v by -pi/2.
        end
      else
        dxdy = [1 0];
      end      
    end
    
    function videoPlay(obj)
      labeler = obj.labeler_ ;
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
            
      labeler.videoPlaySegmentCore(labeler.currFrame,labeler.nframes,...
        'setFrameArgs',{'updateTables',false});
    end
    
    function videoPlaySegment(obj)
      % Play segment centererd at .currFrame
      
      labeler = obj.labeler_ ;
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
      
      f = labeler.currFrame;
      df = labeler.moviePlaySegRadius;
      fstart = max(1,f-df);
      fend = min(labeler.nframes,f+df);
      labeler.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end

    function videoPlaySegFwdEnding(obj)
      % Play segment ending at .currFrame
      labeler = obj.labeler_ ;
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
            
      f = labeler.currFrame;
      df = labeler.moviePlaySegRadius;
      fstart = max(1,f-df);
      fend = f;
      labeler.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end
    
    function videoPlaySegRevEnding(obj)
      % Play segment (reversed) ending at .currFrame
      labeler = obj.labeler_ ;
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
      
      f = labeler.currFrame;
      df = labeler.moviePlaySegRadius;
      fstart = min(f+df,labeler.nframes);
      fend = f;
      labeler.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end


    function videoPlaySegmentCore(obj,fstart,fend,varargin)
      
      labeler = obj.labeler_ ;
      % mainFigure = obj.mainFigure_ ;  
      % handles = guidata(mainFigure) ;
      
      [setFrameArgs,freset] = myparse(varargin,...
        'setFrameArgs',{},...
        'freset',nan);
      tfreset = ~isnan(freset);
            
      tffwd = fend>fstart;

      ticker = tic();
      while true
        % Ways to exit loop:
        % 1. user cancels playback through GUI mutation of obj.isPlaying_
        % 2. fend reached
        % 3. ctrl-c
        
        if ~obj.isPlaying_
          break;
        end
                  
        dtsec = toc(ticker);
        df = dtsec*labeler.moviePlayFPS;
        if tffwd
          f = ceil(df)+fstart;
          if f > fend
            break;
          end
        else
          f = fstart-ceil(df);
          if f < fend
            break;
          end
        end

        labeler.setFrame(f,setFrameArgs{:});
        drawnow('limitrate');
      end
      
      if tfreset
        % AL20170619 passing setFrameArgs a bit fragile; needed for current
        % callers (don't update labels in videoPlaySegment)
        labeler.setFrame(freset,setFrameArgs{:}); 
      end
      
      % - icon managed by caller      
    end

    function videoCenterOnCurrTargetPoint(obj)
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      [tfsucc,xy] = labeler.videoCenterOnCurrTargetPointHelp();
      if tfsucc
        [x0,y0] = labeler.videoCurrentCenter();
        dx = xy(1)-x0;
        dy = xy(2)-y0;
        ax = handles.axes_curr;
        axisshift(ax,dx,dy);
        ax.CameraPositionMode = 'auto'; % issue #86, behavior differs between 16b and 15b. Use of manual zoom toggles .CPM into manual mode
        ax.CameraTargetMode = 'auto'; % issue #86, etc Use of manual zoom toggles .CTM into manual mode
        %ax.CameraViewAngleMode = 'auto';
      end
    end  
    
    function videoCenterOnCurrTarget(obj, x, y, th)
      % Shift axis center/target and CameraUpVector without touching zoom.
      % 
      % Potential TODO: CamViewAngle treatment looks a little bizzare but
      % seems to work ok. Theoretically better (?), at movieSet time, cache
      % a default CameraViewAngle, and at movieRotateTargetUp set time, set
      % the CamViewAngle to either the default or the default/2 etc.

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      [x0,y0] = obj.videoCurrentCenter();
      tfexternal = nargin>1;
      if ~tfexternal
        [x,y,th] = labeler.currentTargetLoc();
      end
      if isnan(x)
        warningNoTrace('No target selected');
        return;
      end

      dx = x-x0;
      dy = y-y0;
      ax = handles.axes_curr;
      axisshift(ax,dx,dy);
      ax.CameraPositionMode = 'auto'; % issue #86, behavior differs between 16b and 15b. Use of manual zoom toggles .CPM into manual mode
      ax.CameraTargetMode = 'auto'; % issue #86, etc Use of manual zoom toggles .CTM into manual mode
      %ax.CameraViewAngleMode = 'auto';
      if labeler.movieRotateTargetUp || tfexternal
        ax.CameraUpVector = [cos(th) sin(th) 0];
        % if verLessThan('matlab','R2016a')
        %   % See iss#86. In R2016a, the zoom/pan behavior of axes in 3D mode
        %   % (currently, any axis with CameraViewAngleMode manually set)
        %   % changed. Prior to R2016a, zoom on such an axis altered camera
        %   % position via .CameraViewAngle, etc, with the axis limits
        %   % unchanged. Starting in R2016a, zoom on 3D axes changes the axis
        %   % limits while the camera position is unchanged.
        %   %
        %   % Currently we prefer the modern treatment and the
        %   % center-on-target, rotate-target, zoom slider, etc treatments
        %   % are built around that treatment. For prior MATLABs, we work
        %   % around -- it is a little awkward as the fundamental strategy
        %   % behind zoom is different. For prior MATLABs users should prefer
        %   % the Zoom slider in the Targets panel as opposed to using the
        %   % zoom tools in the toolbar.
        %   tf = getappdata(mainFigure,'manualZoomOccured');
        %   if tf
        %     ax.CameraViewAngleMode = 'auto';
        %     setappdata(mainFigure,'manualZoomOccured',false);
        %   end
        % end
        if strcmp(ax.CameraViewAngleMode,'auto')
          cva = ax.CameraViewAngle;
          ax.CameraViewAngle = cva/2;
        end
      else
        ax.CameraUpVectorMode = 'auto';
      end
    end
    
    function updateTargetCentrationAndZoom_(obj)
      labeler = obj.labeler_ ;      
      if labeler.isinit ,
        return
      end
      if (labeler.hasTrx || labeler.maIsMA) && labeler.movieCenterOnTarget && ~labeler.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTarget();
        obj.videoZoom(labeler.targetZoomRadiusDefault);
      elseif labeler.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTargetPoint();
      end
    end  % function

    function play(obj, iconStrPlay, playMethodName)
      %labeler = obj.labeler_ ;      
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;
      
      pbPlay = handles.pbPlay ;
      oc = onCleanup(@()(obj.playCleanup(pbPlay, iconStrPlay))) ;
      if ~obj.isPlaying_
        obj.isPlaying_ = true ;
        pbPlay.CData = Icons.ims.stop ;
        obj.(playMethodName) ;
      end
    end

    function playCleanup(obj, pbPlay, iconStrPlay)
      pbPlay.CData = Icons.ims.(iconStrPlay) ;
      obj.isPlaying_ = false ;
    end

    function tblTrx_cell_selected_(obj, src, evt) %#ok<*DEFNU>
      % Current/last row selection is maintained in hObject.UserData

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~(labeler.hasTrx || labeler.maIsMA)
        return
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
    end

    %
    % This is where the insertion of the dispatchMainFigureCallback.m methods
    % starts.
    %



    function pumTrack_actuated_(obj, src,evt)  %#ok<INUSD>

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      labeler.trackModeIdx = src.Value;
    end



    function slider_frame_actuated_(obj, src,evt,varargin)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % Hints: get(src,'Value') returns position of slider
      %        get(src,'Min') and get(src,'Max') to determine range of slider

      debugtiming = false;
      if debugtiming,
        starttime = tic() ;
      end



      if ~labeler.hasProject
        set(src,'Value',0);
        return;
      end
      if ~labeler.hasMovie
        set(src,'Value',0);
        msgbox('There is no movie open.');
        return;
      end

      v = get(src,'Value');
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
        set(src,'Value',sldval);
      end

      if debugtiming,
        fprintf('Slider callback setting to frame %d took %f seconds\n',f,toc(starttime));
      end


    end



    function edit_frame_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return;
      end



      f = str2double(get(src,'String'));
      if isnan(f)
        set(src,'String',num2str(labeler.currFrame));
        return;
      end
      f = min(max(1,round(f)),labeler.nframes);
      if ~labeler.trxCheckFramesLive(f)
        set(src,'String',num2str(labeler.currFrame));
        warnstr = sprintf('Frame %d is out-of-range for current target.',f);
        warndlg(warnstr,'Out of range');
        return;
      end
      set(src,'String',num2str(f));
      if f ~= labeler.currFrame
        labeler.setFrame(f)
      end



    end



    function pbClear_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return;
      end
      labeler.lblCore.clearLabels();
      labeler.CheckPrevAxesTemplate();


    end



    function tbAccept_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % debugtiming = false;
      % if debugtiming,
      %   starttime = tic;
      % end

      if ~labeler.doProjectAndMovieExist()
        return;
      end
      lc = labeler.lblCore;
      switch lc.state
        case LabelState.ADJUST
          lc.acceptLabels();
          %labeler.InitializePrevAxesTemplate();
        case LabelState.ACCEPTED
          lc.unAcceptLabels();
          %labeler.CheckPrevAxesTemplate();
        otherwise
          assert(false);
      end

      % if debugtiming,
      %   fprintf('toggleAccept took %f seconds\n',toc(starttime));
      % end

    end

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




    function sldZoom_actuated_(obj, src, evt, ~)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      if ~labeler.doProjectAndMovieExist()
        return;
      end


      v = src.Value;
      userdata = src.UserData;
      logzoomrad = userdata(2)+v*(userdata(1)-userdata(2));
      zoomRad = exp(logzoomrad);
      obj.videoZoom(zoomRad);
      hlpRemoveFocus(src,handles);

    end



    function pbResetZoom_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      hAxs = handles.axes_all;
      hIms = handles.images_all;
      assert(numel(hAxs)==numel(hIms));
      arrayfun(@zoomOutFullView,hAxs,hIms,false(1,numel(hIms)));

    end



    function pbSetZoom_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.targetZoomRadiusDefault = diff(handles.axes_curr.XLim)/2;

    end



    function pbRecallZoom_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      % TODO this is broken!!
      obj.videoCenterOnCurrTarget();
      obj.videoZoom(labeler.targetZoomRadiusDefault);
    end



    function tbTLSelectMode_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return;
      end
      tl = handles.labelTLInfo;
      tl.selectOn = src.Value;

    end



    function pbClearSelection_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return;
      end
      tl = handles.labelTLInfo;
      tl.selectClearSelection();

      % function cbkFreezePrevAxesToMainWindow(src,evt)
      % handles = guidata(src);
      % labeler.setPrevAxesMode(PrevAxesMode.FROZEN);

      % function cbkUnfreezePrevAxes(src,evt)
      % handles = guidata(src);
      % labeler.setPrevAxesMode(PrevAxesMode.LASTSEEN);

      %% menu
    end



    function menu_file_save_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.setStatus('Saving project...');
      labeler.projSaveSmart();
      labeler.projAssignProjNameFromProjFileIfAppropriate();
      labeler.clearStatus()

    end



    function menu_file_saveas_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.setStatus('Saving project...');
      labeler.projSaveAs();
      labeler.projAssignProjNameFromProjFileIfAppropriate();
      labeler.clearStatus()

    end



    function menu_file_load_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;



      labeler.setStatus('Loading Project...') ;
      if obj.raiseUnsavedChangesDialogIfNeeded() ,
        currMovInfo = labeler.projLoad();
        if ~isempty(currMovInfo)
          obj.movieManagerController_.setVisible(true);
          wstr = ...
            sprintf(strcatg('Could not find file for movie(set) %d: %s.\n\nProject opened with no movie selected. ', ...
            'Double-click a row in the MovieManager or use the ''Switch to Movie'' button to start working on a movie.'), ...
            currMovInfo.iMov, ...
            currMovInfo.badfile);
          warndlg(wstr,'Movie not found','modal');
        end
      end
      labeler.clearStatus()

      % function tfcontinue = hlpSave(labelerObj)
      % tfcontinue = true;
      %
      % if ~verLessThan('matlab','9.6') && batchStartupOptionUsed
      %   return;
      % end
      %
      % OPTION_SAVE = 'Save first';
      % OPTION_PROC = 'Proceed without saving';
      % OPTION_CANC = 'Cancel';
      % if labelerObj.doesNeedSave ,
      %   res = questdlg('You have unsaved changes to your project. If you proceed without saving, your changes will be lost.',...
      %     'Unsaved changes',OPTION_SAVE,OPTION_PROC,OPTION_CANC,OPTION_SAVE);
      %   switch res
      %     case OPTION_SAVE
      %       labelerObj.projSaveSmart();
      %       labelerObj.projAssignProjNameFromProjFileIfAppropriate();
      %     case OPTION_CANC
      %       tfcontinue = false;
      %     case OPTION_PROC
      %       % none
      %   end
      % end



    end

    function menu_file_managemovies_actuated_(src, evt)  %#ok<INUSD>

      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~isempty(obj.movieManagerController_) && isvalid(obj.movieManagerController_) ,
        obj.movieManagerController_.setVisible(true);
      else
        labeler.lerror('LabelerGUI:movieManagerController','Please create or load a project.');
      end



    end



    function menu_file_import_labels_trk_curr_mov_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      if ~labeler.hasMovie
        labeler.lerror('LabelerGUI:noMovie','No movie is loaded.');
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
      labeler.labelImportTrkPromptGenericSimple(iMov,...
        'labelImportTrk','gtok',false);

    end



    function menu_file_import_labels2_trk_curr_mov_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      if ~labeler.hasMovie
        labeler.lerror('LabelerGUI:noMovie','No movie is loaded.');
      end
      iMov = labeler.currMovie; % gt-aware
      labeler.setStatus('Importing tracking results...');
      labeler.labelImportTrkPromptGenericSimple(iMov,'labels2ImportTrk','gtok',true);
      labeler.clearStatus();

    end



    function menu_file_export_labels_trks_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      [tfok,rawtrkname] = obj.getExportTrkRawNameUI('labels',true);
      if ~tfok
        return;
      end
      labeler.setStatus('Exporting tracking results...');
      labeler.labelExportTrk(1:labeler.nmoviesGTaware,'rawtrkname',rawtrkname);
      labeler.clearStatus();

    end



    function menu_file_export_labels_table_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


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

    end



    function menu_file_import_labels_table_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


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

    end



    function menu_file_export_stripped_lbl_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      fname = labeler.getDefaultFilenameExportStrippedLbl();
      [f,p] = uiputfile(fname,'Export File');
      if isequal(f,0)
        return
      end
      fname = fullfile(p,f);
      labeler.setStatus(sprintf('Exporting training data to %s',fname));
      labeler.projExportTrainData(fname)
      fprintf('Saved training data to file ''%s''.\n',fname);
      labeler.clearStatus();

    end



    function menu_file_crop_mode_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;




      if ~isempty(labeler.tracker) && ~labeler.gtIsGTMode && labeler.labelPosMovieHasLabels(labeler.currMovie),
        res = questdlg('Frames of the current movie are labeled. Editing the crop region for this movie will cause trackers to be reset. Continue?');
        if ~strcmpi(res,'Yes'),
          return;
        end
      end

      labeler.setStatus('Switching crop mode...');
      labeler.cropSetCropMode(~labeler.cropIsCropMode);
      labeler.clearStatus();

    end



    function menu_file_clean_tempdir_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.setStatus('Deleting temp directories...');
      labeler.projRemoveOtherTempDirs();
      labeler.clearStatus();

    end



    function menu_file_bundle_tempdir_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.setStatus('Bundling the temp directory...');
      labeler.projBundleTempDir();
      labeler.clearStatus();


    end



    function menu_help_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


    end



    function menu_help_labeling_actions_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      lblCore = labeler.lblCore;
      if isempty(lblCore)
        h = 'Please open a movie first.';
      else
        h = lblCore.getLabelingHelp();
      end
      msgbox(h,'Labeling Actions','help',struct('Interpreter','tex','WindowStyle','replace'));

    end



    function menu_help_about_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      about(labeler);

    end



    function menu_setup_sequential_mode_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.menuSetupLabelModeCbkGeneric(src);

    end



    function menu_setup_sequential_add_mode_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.menuSetupLabelModeCbkGeneric(src);

    end



    function menu_setup_template_mode_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.menuSetupLabelModeCbkGeneric(src);

    end



    function menu_setup_highthroughput_mode_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.menuSetupLabelModeCbkGeneric(src);

    end



    function menu_setup_multiview_calibrated_mode_2_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.menuSetupLabelModeCbkGeneric(src);

    end



    function menu_setup_multianimal_mode_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.menuSetupLabelModeCbkGeneric(src);

    end



    function menu_setup_label_overlay_montage_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.setStatus('Plotting all labels on one axes to visualize label distribution...');

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
      labeler.clearStatus();
    end



    function menu_setup_label_outliers_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.setStatus('Finding outliers in labels...');

      label_outlier_gui(labeler);
      labeler.clearStatus();

    end



    function menu_setup_set_nframe_skip_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


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

    end



    function menu_setup_streamlined_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      lc = labeler.lblCore;
      assert(isa(lc,'LabelCoreMultiViewCalibrated2'));
      lc.streamlined = ~lc.streamlined;

    end



    function menu_setup_ma_twoclick_align_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      lc = labeler.lblCore;
      tftc = ~lc.tcOn;
      labeler.isTwoClickAlign = tftc; % store the state
      lc.setTwoClickOn(tftc);
      src.Checked = onIff(tftc); % skip listener business for now

    end



    function menu_setup_set_labeling_point_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      ipt = labeler.lblCore.iPoint;
      ret = inputdlg('Select labeling point','Point number',1,{num2str(ipt)});
      if isempty(ret)
        return;
      end
      ret = str2double(ret{1});
      labeler.lblCore.setIPoint(ret);


    end



    function menu_setup_use_calibration_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;



      lc = labeler.lblCore;
      if lc.supportsCalibration,
        lc.toggleShowCalibration();
        src.Checked = onIff(lc.showCalibration);
      else
        src.Checked = 'off';
      end

    end



    function menu_setup_load_calibration_file_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


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


      %set_use_calibration(handles,true);
      lc = labeler.lblCore;
      if lc.supportsCalibration,
        lc.setShowCalibration(true);
      end
      handles.menu_setup_use_calibration.Checked = onIff(lc.showCalibration);
      RC.saveprop('lastCalibrationFile',fname);
    end



    function menu_view_show_bgsubbed_frames_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tf = ~strcmp(src.Checked,'on');

      labeler.movieViewBGsubbed = tf;

    end



    function menu_view_adjustbrightness_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      [tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(obj);
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
      		@(s,e) closeImContrast(obj,iAxRead,iAxApply));
      end

    end



    function menu_view_converttograyscale_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tf = ~strcmp(src.Checked,'on');

      labeler.movieForceGrayscale = tf;
      if labeler.hasMovie
        % Pure convenience: update image for user rather than wait for next
        % frame-switch. Could also put this in Labeler.set.movieForceGrayscale.
        labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
      end
    end



    function menu_view_gammacorrect_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      [tfok,~,iAxApply] = hlpAxesAdjustPrompt(obj);
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

    end



    function menu_file_quit_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      obj.quitRequested() ;
    end



    function menu_view_hide_trajectories_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.setShowTrx(~labeler.showTrx);

    end



    function menu_view_plot_trajectories_current_target_only_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.setShowTrxCurrTargetOnly(~labeler.showTrxCurrTargetOnly);

    end



    function menu_view_trajectories_centervideoontarget_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.movieCenterOnTarget = ~labeler.movieCenterOnTarget;
    end



    function menu_view_rotate_video_target_up_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.movieRotateTargetUp = ~labeler.movieRotateTargetUp;
    end



    function menu_view_flip_flipud_movie_only_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      [tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        labeler.movieInvert(iAxApply) = ~labeler.movieInvert(iAxApply);
        if labeler.hasMovie
          labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
        end
      end
    end



    function menu_view_flip_flipud_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      [tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        for iAx = iAxApply(:)'
          ax = handles.axes_all(iAx);
          ax.YDir = toggleAxisDir(ax.YDir);
        end
        labeler.UpdatePrevAxesDirections();
      end
    end



    function menu_view_flip_fliplr_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      [tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        for iAx = iAxApply(:)'
          ax = handles.axes_all(iAx);
          ax.XDir = toggleAxisDir(ax.XDir);
          %     if ax==handles.axes_curr
          %       ax2 = handles.axes_prev;
          %       ax2.XDir = toggleAxisDir(ax2.XDir);
          %     end
          labeler.UpdatePrevAxesDirections();
        end
      end
    end



    function menu_view_show_axes_toolbar_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      ax = handles.axes_curr;
      if strcmp(src.Checked,'on')
        onoff = 'off';
      else
        onoff = 'on';
      end
      ax.Toolbar.Visible = onoff;
      src.Checked = onoff;
      % For now not listening to ax.Toolbar.Visible for cmdline changes


    end



    function menu_view_fit_entire_image_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      hAxs = handles.axes_all;
      hIms = handles.images_all;
      assert(numel(hAxs)==numel(hIms));
      arrayfun(@zoomOutFullView,hAxs,hIms,true(1,numel(hAxs)));
      labeler.movieCenterOnTarget = false;


    end



    function menu_view_hide_labels_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      lblCore = labeler.lblCore;
      if ~isempty(lblCore)
        lblCore.labelsHideToggle();
      end

    end



    function menu_view_hide_predictions_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      tracker = labeler.tracker;
      if ~isempty(tracker)
        tracker.hideVizToggle();
      end

    end



    function menu_view_show_preds_curr_target_only_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      tracker = labeler.tracker;
      if ~isempty(tracker)
        tracker.showPredsCurrTargetOnlyToggle();
      end

    end



    function menu_view_hide_imported_predictions_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.labels2VizToggle();



    end



    function menu_view_show_imported_preds_curr_target_only_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.labels2VizSetShowCurrTargetOnly(~labeler.labels2ShowCurrTargetOnly);
    end



    function menu_view_show_tick_labels_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % just use checked state of menu for now, no other state
      toggleOnOff(src,'Checked');
      hlpTickGridBang(handles.axes_all, handles.menu_view_show_tick_labels, handles.menu_view_show_grid) ;



    end



    function menu_view_show_grid_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % just use checked state of menu for now, no other state
      toggleOnOff(src,'Checked');
      hlpTickGridBang(handles.axes_all, handles.menu_view_show_tick_labels, handles.menu_view_show_grid) ;



    end



    function menu_track_setparametersfile_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % Really, "configure parameters"


      if any(labeler.bgTrnIsRunningFromTrackerIndex()),
        warndlg('Cannot change training parameters while trackers are training.','Training in progress','modal');
        return;
      end
      labeler.setStatus('Setting training parameters...');

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

      labeler.clearStatus();


    end



    function menu_track_settrackparams_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;



      labeler.setStatus('Setting tracking parameters...');

      [tPrm] = labeler.trackGetTrackParams();

      sPrmTrack = ParameterSetup(handles.figure,tPrm,'labelerObj',labeler); % modal

      if ~isempty(sPrmTrack),
        sPrmNew = labeler.trackSetTrackParams(sPrmTrack);
        RC.saveprop('lastCPRAPTParams',sPrmNew);
        %cbkSaveNeeded(labeler,true,'Parameters changed');
        labeler.setDoesNeedSave(true, 'Parameters changed') ;
      end

      labeler.clearStatus();


    end



    function menu_track_auto_params_update_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      checked = get(src,'Checked');
      set(src,'Checked',~checked);
      labeler.trackAutoSetParams = ~checked;

      labeler.setDoesNeedSave(true, 'Auto compute training parameters changed') ;


    end



    function menu_track_use_all_labels_to_train_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      tObj = labeler.tracker;
      if isempty(tObj)
        labeler.lerror('LabelerGUI:tracker','No tracker for this project.');
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
    end



    function menu_track_trainincremental_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.trainIncremental();

    end



    function menu_go_targets_summary_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if labeler.maIsMA
        TrkInfoUI(labeler);
      else
        obj.raiseTargetsTableFigure();
      end

    end



    function menu_go_nav_prefs_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.navPrefsUI();

    end



    function menu_go_gt_frames_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.gtShowGTManager();

    end



    function menu_evaluate_crossvalidate_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;




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
        labeler.lerror('LabelerGUI:xvalid','Number of folds must be a positive integer greater than 1.');
      end

      tbl.split = ceil(nfold*rand(n,1));

      t = labeler.tracker;
      t.trainsplit(tbl);


    end



    function menu_track_clear_tracking_results_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      % legacy behavior not sure why; maybe b/c the user is prob wanting to increase avail mem
      %labeler.preProcInitData();
      res = questdlg('Are you sure you want to clear tracking results?');
      if ~strcmpi(res,'yes'),
        return;
      end
      labeler.setStatus('Clearing tracking results...');
      tObj = labeler.tracker;
      tObj.clearTrackingResults();
      labeler.clearStatus();
      %msgbox('Tracking results cleared.','Done');
    end



    function menu_track_batch_track_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      tbobj = TrackBatchGUI(labeler);
      tbobj.run();


    end



    function menu_track_all_movies_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      mIdx = labeler.allMovIdx();
      toTrackIn = labeler.mIdx2TrackList(mIdx);
      tbobj = TrackBatchGUI(labeler,'toTrack',toTrackIn);
      % [toTrackOut] = tbobj.run();
      tbobj.run();
      % todo: import predictions


    end



    function menu_track_current_movie_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      mIdx = labeler.currMovIdx;
      toTrackIn = labeler.mIdx2TrackList(mIdx);
      mdobj = SpecifyMovieToTrackGUI(labeler,mainFigure,toTrackIn);
      [toTrackOut,dostore] = mdobj.run();
      if ~dostore,
        return;
      end
      trackBatch('labeler',labeler,'toTrack',toTrackOut);


    end



    function menu_track_id_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.track_id = ~labeler.track_id;
      set(handles.menu_track_id,'checked',labeler.track_id);


    end



    function menu_file_clear_imported_actuated_(obj, src,evtdata)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.labels2Clear();

    end



    function menu_file_export_all_movies_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      nMov = labeler.nmoviesGTaware;
      if nMov==0
        labeler.lerror('LabelerGUI:noMov','No movies in project.');
      end
      iMov = 1:nMov;
      [tfok,rawtrkname] = obj.getExportTrkRawNameUI();
      if ~tfok
        return;
      end
      labeler.trackExportResults(iMov,'rawtrkname',rawtrkname);

    end



    function menu_track_set_labels_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      tObj = labeler.tracker;
      if labeler.gtIsGTMode
        labeler.lerror('LabelerGUI:gt','Unsupported in GT mode.');
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
          labeler.lerror('LabelerGUI:setLabels','No movie open.');
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
            labeler.lerror('LabelerGUI:setLabels','Unsupported for multiple targets.');
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

    end



    function menu_track_background_predict_start_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tObj = labeler.tracker;
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

    end



    function menu_track_background_predict_end_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tObj = labeler.tracker;
      if tObj.asyncIsPrepared
        tObj.asyncStopBgRunner();
      else
        warndlg('Background worker is not running.','Background tracking');
      end

    end



    function menu_track_background_predict_stats_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tObj = labeler.tracker;
      if tObj.asyncIsPrepared
        tObj.asyncComputeStats();
      else
        warningNoTrace('LabelerGUI:bgTrack',...
          'No background tracking information available.','Background tracking');
      end

    end



    function menu_evaluate_gtmode_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.setStatus('Switching between Labeling and Ground Truth Mode...');

      gt = labeler.gtIsGTMode;
      gtNew = ~gt;
      labeler.gtSetGTMode(gtNew);
      if gtNew
        mmc = obj.movieManagerController_ ;
        mmc.setVisible(true);
        figure(mmc.hFig);
      end
      labeler.clearStatus();

    end



    function menu_evaluate_gtloadsuggestions_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      LabelerGT.loadSuggestionsUI(labeler);

    end



    function menu_evaluate_gtsetsuggestions_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      LabelerGT.setSuggestionsToLabeledUI(labeler);

    end



    function menu_evaluate_gtcomputeperf_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      assert(labeler.gtIsGTMode);
      labeler.gtComputeGTPerformance();

    end



    function menu_evaluate_gtcomputeperfimported_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      assert(labeler.gtIsGTMode);
      labeler.gtComputeGTPerformance('useLabels2',true);

    end



    function menu_evaluate_gtexportresults_actuated_(obj, src,evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;



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

    end



    function pumInfo_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      cprop = get(src,'Value');
      handles.labelTLInfo.setCurProp(cprop);
      cpropNew = handles.labelTLInfo.getCurProp();
      if cpropNew ~= cprop,
        set(src,'Value',cpropNew);
      end
      hlpRemoveFocus(src,handles);

    end



    function pbPlaySeg_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return
      end
      obj.play('playsegment', 'videoPlaySegFwdEnding') ;

    end



    function pbPlaySegRev_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return
      end
      obj.play('playsegmentrev', 'videoPlaySegRevEnding') ;

    end



    function pbPlay_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if ~labeler.doProjectAndMovieExist()
        return
      end
      obj.play('play', 'videoPlay') ;
    end



    function tbAdjustCropSize_actuated_(obj, src, evt)  %#ok<INUSD>


      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      cropUpdateCropAdjustingCropSize(handles);
      tb = handles.tbAdjustCropSize;
      if tb.Value==tb.Min
        % user clicked "Done Adjusting"
        warningNoTrace('All movies in a given view must share the same crop size. The sizes of all crops have been updated as necessary.');
      elseif tb.Value==tb.Max
        % user clicked "Adjust Crop Size"
        if ~labeler.cropProjHasCrops
          labeler.cropInitCropsAllMovies;
          fprintf(1,'Default crop initialized for all movies.\n');
          obj.cropUpdateCropHRects_();
        end
      end
    end



    function pbClearAllCrops_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.cropClearAllCrops();


    end



    function menu_file_export_labels2_trk_curr_mov_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to menu_file_export_labels2_trk_curr_mov (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)


      iMov = labeler.currMovie;
      if iMov==0
        labeler.lerror('LabelerGUI:noMov','No movie currently set.');
      end
      [tfok,rawtrkname] = obj.getExportTrkRawNameUI();
      if ~tfok
        return;
      end
      labeler.trackExportResults(iMov,'rawtrkname',rawtrkname);


    end



    function menu_file_import_export_advanced_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to menu_file_import_export_advanced (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)


    end



    function menu_track_tracking_algorithm_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to menu_track_tracking_algorithm (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)

    end



    function menu_view_landmark_colors_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      cbkApply = @(varargin)(labeler.hlpApplyCosmetics(varargin{:})) ;
      LandmarkColors(labeler,cbkApply);
      % AL 20220217: changes now applied immediately
      % if ischange
      %   cbkApply(savedres.colorSpecs,savedres.markerSpecs,savedres.skelSpecs);
      % end

    end



    function menu_track_edit_skeleton_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      landmark_specs('labeler',labeler);
      %hasSkeleton = ~isempty(labeler.skeletonEdges) ;
      %labeler.setShowSkeleton(hasSkeleton) ;

    end



    function menu_track_viz_dataaug_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;


      labeler.retrainAugOnly() ;

    end



    function menu_view_showhide_skeleton_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if strcmpi(get(src,'Checked'),'off'),
        src.Checked = 'on';
        labeler.setShowSkeleton(true);
      else
        src.Checked = 'off';
        labeler.setShowSkeleton(false);
      end

    end



    function menu_view_showhide_maroi_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      if strcmpi(get(src,'Checked'),'off'),
        labeler.setShowMaRoi(true);
      else
        labeler.setShowMaRoi(false);
      end

    end



    function menu_view_showhide_maroiaux_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      tf = strcmpi(get(src,'Checked'),'off');
      labeler.setShowMaRoiAux(tf);

    end



    function popupmenu_prevmode_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to popupmenu_prevmode (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)

      contents = cellstr(get(src,'String'));
      mode = contents{get(src,'Value')};
      if strcmpi(mode,'Reference'),
        labeler.setPrevAxesMode(PrevAxesMode.FROZEN,labeler.prevAxesModeInfo);
      else
        labeler.setPrevAxesMode(PrevAxesMode.LASTSEEN);
      end



    end



    function pushbutton_freezetemplate_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      labeler.setPrevAxesMode(PrevAxesMode.FROZEN);



    end



    function pushbutton_exitcropmode_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to pushbutton_exitcropmode (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)


      labeler.cropSetCropMode(false);


    end



    function menu_view_occluded_points_box_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to menu_view_occluded_points_box (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)


      labeler.setShowOccludedBox(~labeler.showOccludedBox);
      if labeler.showOccludedBox,
        labeler.lblCore.showOcc();
      else
        labeler.lblCore.hideOcc();
      end

    end



    function pumInfo_labels_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;

      % src    handle to pumInfo_labels (see GCBO)
      % evt  reserved - to be defined in a future version of MATLAB
      % handles    structure with handles and user data (see GUIDATA)

      % Hints: contents = cellstr(get(src,'String')) returns pumInfo_labels contents as cell array
      %        contents{get(src,'Value')} returns selected item from pumInfo_labels

      ipropType = get(src,'Value');
      % see also InfoTimeline/enforcePropConsistencyWithUI
      iprop = get(handles.pumInfo,'Value');
      props = handles.labelTLInfo.getPropsDisp(ipropType);
      if iprop > numel(props),
        iprop = 1;
      end
      set(handles.pumInfo,'String',props,'Value',iprop);
      handles.labelTLInfo.setCurPropType(ipropType,iprop);
    end

  end  % methods  
end  % classdef
