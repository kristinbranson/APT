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
    pxTxUnsavedChangesWidth_
    pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_
    pumTrackInitFontSize_
    pumTrackInitHeight_
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
      obj.mainFigure_ = createLabelerMainFigure() ;
      dispatchMainFigureCallback('register_labeler', obj.mainFigure_, labeler) ;
      dispatchMainFigureCallback('register_controller', obj.mainFigure_, obj) ;
      obj.labeler_.registerController(obj) ;  % hack
      obj.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      obj.isInYodaMode_ = isInYodaMode ;  
        % If in yoda mode, we don't wrap GUI-event function calls in a try..catch.
        % Useful for debugging.
        
      % Set up this resize thing
      obj.initializeResizeInfo_() ;
      obj.mainFigure_.SizeChangedFcn = @(src,evt)(obj.resize()) ;
      obj.resize() ;

      % Update the controls enablement  
      obj.enableControls_('noproject') ;
      
      % Update the status
      obj.updateStatus([],[]) ;

      % % Populate the callbacks of the controls in the main figure---someday
      % apt.populate_callbacks_bang(obj.mainFigure_, obj) ;

      % Create the waitbar figure, which we re-use  
      obj.waitbarFigure_ = waitbar(0, '', ...
                                   'Visible', 'off', ...
                                   'CreateCancelBtn', @(source,event)(obj.didCancelWaitbar())) ;
      obj.waitbarFigure_.CloseRequestFcn = @(source,event)(nop()) ;
        
      % Add some controls to the UI that we can set up before there is a project
      obj.initialize_menu_track_backend_config_() ;

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
      obj.listeners_(end+1) = ...@
        addlistener(labeler,'didSetCurrTracker',@(source,event)(obj.cbkCurrTrackerChanged()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLastLabelChangeTS',@(source,event)(obj.cbkLastLabelChangeTS()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackParams',@(source,event)(obj.cbkParameterChange()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackDLBackEnd', @(src,evt)(obj.update_menu_track_backend_config_()) ) ;

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
      
      arrayfun(@(x)zoom(x,'off'),handles.figs_all); % Cannot set KPF if zoom or pan is on
      arrayfun(@(x)pan(x,'off'),handles.figs_all);
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
    
    function cbkTrackerShowVizReplicatesChanged(obj)
      lObj = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      handles.menu_track_cpr_show_replicates.Checked = onIff(lObj.tracker.showVizReplicates) ;
    end  % function
    
    function cbkTrackerStoreFullTrackingChanged(obj)
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      handles = guidata(mainFigure) ;
      sft = labeler.tracker.storeFullTracking;
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
    end  % function
    
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
        'CellSelectionCallback',@(src,evt)cbkTblFramesCellSelection(src,evt),...
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
      hPUM = handles.pumTrack;
      hPUM.Value = labeler.trackModeIdx;
      try %#ok<TRYNC>
        fullstrings = getappdata(hPUM,'FullStrings');
        set(handles.text_framestotrackinfo,'String',fullstrings{hPUM.Value});
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
      hPUM = handles.pumTrack;
      hPUM.String = menustrs_compact;
      setappdata(hPUM,'FullStrings',menustrs);
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
        labeler.videoZoom(labeler.targetZoomRadiusDefault);
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
      % record state for txUnsavedChanges
      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1)+hPnlPrev.Position(3);
      obj.pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_ = ...
        uiPnlPrevRightEdge-hTx.Position(1);
      obj.pxTxUnsavedChangesWidth_ = hTx.Position(3);
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      
      pumTrack = handles.pumTrack;

      % Iss #116. Appears nec to get proper resize behavior
      pumTrack.Max = 2;
      
      obj.pumTrackInitFontSize_ = pumTrack.FontSize;
      obj.pumTrackInitHeight_ = pumTrack.Position(4);
    end
    
    function resize(obj)
      mainFigure = obj.mainFigure_ ;  
      handles = guidata(mainFigure) ;

      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1)+hPnlPrev.Position(3);
      hTx.Position(1) = uiPnlPrevRightEdge-obj.pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_;
      hTx.Position(3) = obj.pxTxUnsavedChangesWidth_;
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      obj.updateStatus() ;
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

  end  % methods  
end  % classdef
