classdef LabelerController < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    mainFigure_  % the GH to the main figure
    listeners_
    satellites_ = gobjects(1,0)  % handles of dialogs, figures, etc that will get deleted when this object is deleted
    waitbarFigure_ = gobjects(1,0)  % a GH to a waitbar() figure, or empty
    trackingMonitorVisualizer_  % a subcontroller
    trainingMonitorVisualizer_  % a subcontroller
    movieManagerController_
    backendTestController_
    pxTxUnsavedChangesWidth_  
      % We will record the width (in pixels) of txUnsavedChanges here, so we can keep it fixed when we resize
    isPlaying_ = false  % whether a video is currently playing or not
    labelTLInfo  % an InfoTimelineController object
    splashScreenFigureOrEmpty_  % GH to the splash screen figure, or empty
  end

  properties  % private/protected by convention
    tvTrx_  % scalar TrackingVisualizerTrx
    isInYodaMode_ = false  
      % Set to true to allow control actuation to happen *ouside* or a try/catch
      % block.  Useful for debugging.  "Do, or do not.  There is no try." --Yoda
  end

  properties  % these are all the things that used to be in the main figure's guidata, but are not simple controls
    figs_all
    axes_all
    images_all
  end

  properties (Constant)
    busystatuscolor = [ 1 0 1 ]
    idlestatuscolor = [ 0 1 0 ]
    minTblFramesRows = 12;
    minTblTrxRows = 12;
  end

  properties  % these are all the things that used to be in the main figure's guidata
    axes_curr
    axes_occ
    axes_prev
    axes_timeline_islabeled
    axes_timeline_manual
    cropHRect
    edit_frame
    % h_addpoints_only
    % h_ma_only
    % h_multiview_only
    % h_nonma_only
    % h_singleview_only
    image_curr
    image_prev
    labelMode2SetupMenu
    menu_debug
    menu_debug_generate_db
    menu_evaluate
    menu_evaluate_crossvalidate
    menu_evaluate_gtcomputeperf
    menu_evaluate_gtcomputeperfimported
    menu_evaluate_gtexportresults
    menu_evaluate_gtloadsuggestions
    menu_evaluate_gtsavesuggestions
    menu_evaluate_gtmode
    menu_evaluate_gt_frames
    menu_evaluate_gtsetsuggestions
    menu_file
    menu_file_bundle_tempdir
    menu_file_clean_tempdir
    menu_file_clear_imported
    menu_file_crop_mode
    menu_file_export_all_movies
    menu_file_export_labels2_trk_curr_mov
    menu_file_export_labels_table
    menu_file_export_labels_cocojson
    menu_file_import_labels_cocojson
    menu_file_export_labels_trks
    menu_file_import_export_advanced
    menu_file_import_labels2_trk_curr_mov
    menu_file_import_labels_table
    menu_file_import_labels_trk_curr_mov
    menu_file_import
    menu_file_export
    menu_file_load
    menu_file_managemovies
    menu_file_new
    menu_file_quit
    menu_file_save
    menu_file_saveas
    menu_file_shortcuts
    menu_go
    menu_go_movies_summary
    menu_go_nav_prefs
    menu_go_targets_summary
    menu_help
    menu_help_about
    menu_help_doc
    menu_help_labeling_actions
    menu_labeling_setup
    menu_quit_but_dont_delete_temp_folder
    % menu_setup_createtemplate
    menu_setup_highthroughput_mode
    menu_setup_label_outliers
    menu_setup_label_overlay_montage
    menu_setup_load_calibration_file
    menu_setup_ma_twoclick_align
    menu_setup_multianimal_mode
    % menu_setup_multiview_calibrated_mode
    menu_setup_multiview_calibrated_mode_2
    menu_setup_sequential_add_mode
    menu_setup_sequential_mode
    menu_setup_set_labeling_point
    menu_setup_set_nframe_skip
    menu_setup_streamlined
    menu_setup_template_mode
    % menu_setup_tracking_correction_mode
    menu_setup_use_calibration
    menu_start_tracking_but_dont_call_python
    menu_start_training_but_dont_call_python
    menu_track
    menu_track_all_movies
    menu_track_auto_params_update
    menu_track_batch_track
    menu_track_clear_tracking_results
    menu_track_current_movie
    menu_track_delete_current_tracker
    menu_track_delete_old_trackers
    menu_track_edit_skeleton
    menu_track_set_labels
    menu_track_setparametersfile
    menu_track_settrackparams
    menu_track_tracker_history
    menu_track_tracking_algorithm
    menu_track_trainincremental
    menu_track_viz_dataaug
    menu_view
    menu_view_adjustbrightness
    menu_view_converttograyscale
    menu_view_fit_entire_image
    menu_view_fps
    menu_view_flip
    menu_view_flip_fliplr
    menu_view_flip_flipud
    menu_view_flip_flipud_movie_only
    menu_view_gammacorrect
    menu_view_showhide_imported_predictions
    menu_view_showhide_imported_preds_all
    menu_view_showhide_imported_preds_curr_target_only
    menu_view_showhide_imported_preds_none
    menu_view_hide_imported_predictions
    menu_view_hide_labels
    menu_view_showhide_predictions
    menu_view_showhide_trajectories
    menu_view_keypoint_appearance
    menu_view_landmark_label_colors
    menu_view_landmark_prediction_colors
    menu_view_occluded_points_box
    menu_view_pan_toggle
    menu_view_reset_views
    menu_view_rotate_video_target_up
    % menu_view_show_axes_toolbar
    menu_view_show_grid
    menu_view_showhide_preds_all_targets
    menu_view_showhide_preds_curr_target_only
    menu_view_showhide_preds_none
    menu_view_hide_predictions
    menu_view_show_tick_labels
    menu_view_showhide_labelrois
    menu_view_showhide_maroi
    menu_view_showhide_maroiaux
    menu_view_showhide_skeleton
    % menu_view_trajectories
    menu_view_trajectories_centervideoontarget
    menu_view_trajectories_dontshow
    menu_view_trajectories_showall
    menu_view_trajectories_showcurrent
    menu_view_hide_trajectories
    menu_view_zoom_toggle
    pbClear
    pbClearAllCrops
    pbClearSelection
    pbPlay
    pbPlaySeg
    pbPlaySegRev
    pbRecallZoom
    pbResetZoom
    pbSetZoom
    pbTrack
    pbTrain
    pnlStatus
    popupmenu_prevmode
    pumInfo
    pumInfo_labels
    pumTrack
    pushbutton_exitcropmode
    pushbutton_freezetemplate
    scribeOverlay
    setupMenu2LabelMode
    sldZoom
    slider_frame
    tbAccept
    tbAdjustCropSize
    % tbAdjustCropSizeBGColor0
    % tbAdjustCropSizeBGColor1
    % tbAdjustCropSizeString0
    % tbAdjustCropSizeString1
    tbTLSelectMode
    tblFrames
    tblTrx
    text_framestotrack
    text_framestotrackinfo
    text_occludedpoints
    text_trackerinfo
    % toolbar
    txBGTrain
    txCropMode
    txGTMode
    txLblCoreAux
    txMoviename
    txPrevIm
    txStatus
    txTotalFramesLabeled
    txTotalFramesLabeledLabel
    txUnsavedChanges
    tx_timeline_islabeled
    uipanel_targets
    uipanel_targetzoom
    uipanel_frames
    uipanel_cropcontrols
    uipanel_curr
    uipanel_prev
  end

  properties
    axesesHighlightManager_
    hLinkPrevCurr
    newProjAxLimsSetInConfig
    h_ignore_arrows
    GTManagerFigure  % the ground truth manager *figure*
    shortcutkeys
    shortcutfns
    fakeMenuTags
    menu_track_backend_config
    menu_track_backend_config_jrc
    menu_track_backend_config_aws
    menu_track_backend_config_docker
    menu_track_backend_config_conda
    menu_track_backend_settings
    menu_track_backend_config_moreinfo
    menu_track_backend_config_test
  end

  methods
    function obj = LabelerController(varargin)
      % Process args that have to be dealt with before creating the Labeler
      [isInDebugMode, isInAwsDebugMode, isInYodaMode] = ...
        myparse_nocheck(varargin, ...
                        'isInDebugMode',false, ...
                        'isInAwsDebugMode',false, ...
                        'isInYodaMode', false) ;

      % Create the splash screen figure
      % (Do this after creation of main figure so splash screen figure is on top.)
      obj.splashScreenFigureOrEmpty_ = createSplashScreenFigure() ;
      oc = onCleanup(@()(obj.deleteSpashScreenFigureIfItExists_())) ;

      % Create the labeler, tell it there will be a GUI attached
      labeler = Labeler('isgui', true, 'isInDebugMode', isInDebugMode,  'isInAwsDebugMode', isInAwsDebugMode) ;  

      % Bring the splash screen to the foreground
      figure(obj.splashScreenFigureOrEmpty_);

      % Set up the main instance variables
      obj.labeler_ = labeler ;
      mainFigure = createLabelerMainFigure() ;
      obj.mainFigure_ = mainFigure ;
      obj.labeler_.registerController(obj) ;  % hack
      obj.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      obj.isInYodaMode_ = isInYodaMode ;  
        % If in yoda mode, we don't wrap GUI-event function calls in a try..catch.
        % Useful for debugging.
              
      % Initialize all the instance vars that will hold references to GUI controls
      handles = guihandles(mainFigure) ;
      tags = fieldnames(handles) ;
      for i = 1 : numel(tags) 
        tag = tags{i} ;
        if strcmp(tag, 'main_figure') 
          % We allready have a property for the main figure
          continue
        end
        if isprop(obj, tag) ,
          obj.(tag) = handles.(tag) ;
        end
      end
      
      % Set these things
      obj.figs_all = obj.mainFigure_ ;
      obj.axes_all = obj.axes_curr ;
      obj.images_all = obj.image_curr ;
      
      % Set up this resize thing
      obj.initializeResizeInfo_() ;
      mainFigure.SizeChangedFcn = @(src,evt)(obj.resize()) ;
      obj.resize() ;

      % Update the controls enablement  
      obj.updateEnablementOfManyControls() ;
      
      % Update the status
      obj.updateStatusAndPointer() ;

      % % Populate the callbacks of the controls in the main figure---someday
      % apt.populate_callbacks_bang(mainFigure, obj) ;

      % Create the waitbar on demand
      % obj.waitbarFigure_ = waitbar(0, '', ...
      %                              'Visible', 'off', ...
      %                              'CreateCancelBtn', @(source,event)(obj.didCancelWaitbar())) ;
      % obj.waitbarFigure_.CloseRequestFcn = @(source,event)(nop()) ;
        
      % Add some controls to the UI that we can set up before there is a project
      obj.initialize_menu_track_backend_config_() ;
      
      % Create the InfoTimelineController object to help manage the timeline axes, and
      % populate the two popup menus that determine what is shown in the timeline
      % axes.
      itm = labeler.infoTimelineModel ;
      obj.labelTLInfo = InfoTimelineController(labeler, obj.axes_timeline_manual , obj.axes_timeline_islabeled) ;
      set(obj.pumInfo,...
          'String',itm.getPropsDisp(),...
          'Value',itm.curprop);
      set(obj.pumInfo_labels,...
          'String',itm.getPropTypesDisp(),...
          'Value',itm.curproptype);

      % Misc labelmode/Setup menu
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
      obj.labelMode2SetupMenu = struct(tmp{:});
      tmp = LABELMODE_SETUPMENU_MAP(2:end,[2 1]);
      tmp = tmp';
      obj.setupMenu2LabelMode = struct(tmp{:});

      % Make the debug menu visible, if called for
      obj.menu_debug.Visible = onIff(labeler.isInDebugMode) ;      

      % Set up some custom callbacks
      %obj.controller = obj ;
      set(obj.tblTrx, 'CellSelectionCallback', @(s,e)(obj.controlActuated('tblTrx', s, e))) ;
      set(obj.tblFrames, 'CellSelectionCallback',@(s,e)(obj.controlActuated('tblFrames', s, e))) ;
      hZ = zoom(mainFigure);  % hZ is a "zoom object"
      hZ.ActionPostCallback = @(s,e)(obj.cbkPostZoom(s,e)) ;
      hP = pan(mainFigure);  % hP is a "pan object"
      hP.ActionPostCallback = @(s,e)(obj.cbkPostPan(s,e)) ;
      set(mainFigure, 'CloseRequestFcn', @(s,e)(obj.quitRequested())) ;
      obj.axes_timeline_manual.ButtonDownFcn = @(src,evt)obj.timelineButtonDown(src,evt);
      obj.axes_timeline_islabeled.ButtonDownFcn = @(src,evt)obj.timelineButtonDown(src,evt);    
      
      % Set up the figure callbacks to call obj, using the tag to determine the
      % method name.
      visit_children(mainFigure, @set_standard_callback_if_none_bang, obj) ;

      % Manually clear ones that use ContinuousValueChange events
      obj.slider_frame.Callback = [] ;
      obj.sldZoom.Callback = [] ;

      % Add the listeners
      obj.listeners_ = event.listener.empty(1,0) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateDoesNeedSave', @(source,event)(obj.updateDoesNeedSave(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateStatusAndPointer', @(source,event)(obj.updateStatusAndPointer())) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'didSetTrx', @(source,event)(obj.didSetTrx(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateTrxSetShowTrue', @(source,event)(obj.updateTrxSetShowTrue(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateTrxSetShowFalse', @(source,event)(obj.updateTrxSetShowFalse(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'didSpawnTrackingForGT', @(source,event)(obj.showDialogAfterSpawningTrackingForGT(source, event))) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'didComputeGTResults', @(source,event)(obj.showGTResults(source, event))) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didLoadProject',@(source,event)(obj.didLoadProject()));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'update_text_trackerinfo',@(source,event)(obj.update_text_trackerinfo()));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'refreshTrackMonitorViz',@(source,event)(obj.refreshTrackMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTrackMonitorViz',@(source,event)(obj.updateTrackMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'refreshTrainMonitorViz',@(source,event)(obj.refreshTrainMonitorViz()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTrainMonitorViz',@(source,event)(obj.updateTrainMonitorViz()));      
      % obj.listeners_(end+1) = ...
      %   addlistener(labeler,'raiseTrainingStoppedDialog',@(source,event)(obj.raiseTrainingEndedDialog_()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'newProject',@(source,event)(obj.didCreateNewProject()));
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetProjectName',@(source,event)(obj.didChangeProjectName()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetProjFSInfo',@(source,event)(obj.didChangeProjFSInfo()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieInvert',@(source,event)(obj.didChangeMovieInvert()));      
      % obj.listeners_(end+1) = ...
      %   addlistener(labeler,'update_menu_track_tracking_algorithm_quick',@(source,event)(obj.update_menu_track_tracking_algorithm_quick()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'update_menu_track_tracker_history',@(source,event)(obj.update_menu_track_tracker_history()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetCurrTracker',@(source,event)(obj.cbkCurrTrackerChanged()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetLastLabelChangeTS',@(source,event)(obj.cbkLastLabelChangeTS()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackParams',@(source,event)(obj.cbkParameterChange()));            
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetTrackDLBackEnd', @(src,evt)(obj.update_menu_track_backend_config()) ) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTargetCentrationAndZoom', @(src,evt)(obj.updateTargetCentrationAndZoom()) ) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTrainingMonitor', @(src,evt) (obj.updateTrainingMonitor())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'trainEnd', @(src,evt) (obj.cbkTrackerTrainEnd())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateTrackingMonitor', @(src,evt) (obj.updateTrackingMonitor())) ;
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
        addlistener(labeler,'didSetLabelMode',@(s,e)(obj.cbkLabelModeChanged()));
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
        addlistener(labeler,'gtIsGTModeChanged',@(s,e)(obj.didSetGTMode())) ;
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
        addlistener(obj.axes_curr,'XLim','PostSet',@(s,e)(obj.axescurrXLimChanged(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.axes_curr,'XDir','PostSet',@(s,e)(obj.axescurrXDirChanged(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.axes_curr,'YDir','PostSet',@(s,e)(obj.axescurrYDirChanged(s,e))) ;

      % obj.listeners_(end+1) = ...
      %   addlistener(obj.labeler_,'didSetTimelineSelectMode',@(s,e)(obj.cbklabelTLInfoSelectOn(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.labeler_,'updateTimeline',@(s,e)(obj.updateTimeline())) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.labeler_,'updateTimelineStatThresh',@(s,e)(obj.updateTimelineStatThresh())) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.labeler_,'updateTimelineLabels',@(s,e)(obj.updateTimelineLabels())) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.labeler_,'updateTimelineLandmarkColors',@(s,e)(obj.updateTimelineLandmarkColors())) ;

      obj.listeners_(end+1) = ...
        addlistener(obj.slider_frame,'ContinuousValueChange',@(s,e)(obj.controlActuated('slider_frame', s, e))) ;
      obj.listeners_(end+1) = ...
        addlistener(obj.sldZoom,'ContinuousValueChange',@(s,e)(obj.controlActuated('sldZoom', s, e))) ;

      obj.listeners_(end+1) = ...
        addlistener(labeler,'updateMainAxisHighlight',@(s,e)(obj.updateHighlightingOfAxes())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler,'update',@(s,e)(obj.update())) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'gtSuggUpdated', @(s,e)(obj.cbkGTSuggUpdated(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'gtSuggMFTableLbledUpdated', @(s,e)(obj.cbkGTSuggMFTableLbledUpdated())) ;
      % obj.listeners_(end+1) = ...
      %   addlistener(labeler, 'gtResUpdated', @(s,e)(obj.cbkGTResUpdated(s,e))) ;
      obj.listeners_(end+1) = ...
        addlistener(labeler, 'updateAfterCurrentFrameSet', @(s,e)(obj.updateAfterCurrentFrameSet())) ;


      obj.fakeMenuTags = {
        'menu_view_zoom_toggle'
        'menu_view_pan_toggle'
        'menu_view_hide_trajectories'
        'menu_view_hide_predictions'
        'menu_view_hide_imported_predictions'
        };

      % % Stash the guidata
      % guidata(mainFigure, obj) ;
      
      % Update things that need updating at startup
      obj.update() ;

      % Do this once listeners are set up
      obj.controlActuated('handleCreationTimeAdditionalArgumentsGUI', [], [], varargin{:}) ;
      % This will lead to 
      %   obj.labeler_.handleCreationTimeAdditionalArgumentsGUI_(varargin{:})
      % getting called, but we call it via obj.controlActuated() b/c we want to
      % be able to throw errors in the model method and have them get handled via
      % a dialog box vs the error percolating up to the top, depending on whether
      % a LabelerController is present.
    end

    function delete(obj)
      % Having the figure without a controller would be bad, so we make sure to
      % delete the figure (and subfigures) in our destructor.
      % We also delete the model.
      deleteValidGraphicsHandles(obj.satellites_) ;
      deleteValidGraphicsHandles(obj.waitbarFigure_) ;
      delete(obj.trackingMonitorVisualizer_) ;
      delete(obj.trainingMonitorVisualizer_) ;
      try
        deleteValidGraphicsHandles(obj.movieManagerController_.hFig) ;
      catch % fail silently :)
      end
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
      hTx = obj.txUnsavedChanges ;
      if doesNeedSave
        set(hTx,'Visible','on');
      else
        set(hTx,'Visible','off');
      end
    end

    function updateStatusAndPointer(obj)
      % Update the status text box to reflect the current model state.
      labeler = obj.labeler_ ;
      is_busy = labeler.isStatusBusy ;
      pointer = fif(is_busy, 'watch', 'arrow') ;
      valid_figs_all = obj.figs_all(isgraphics(obj.figs_all)) ;
      % Seems like "set(valid_figs_all,'Pointer',pointer);" should be sufficient,
      % istead of the if-else clause below.  Is obj.figs_all not always kept up to
      % date?  Normally obj.mainFigure_ == obj.figs_all(1).
      if ~isempty(valid_figs_all) ,
        set(valid_figs_all,'Pointer',pointer);
      else
        mainFigure = obj.mainFigure_ ;
        if ~isempty(mainFigure) && isgraphics(mainFigure) ,
          set(mainFigure,'Pointer',pointer);
        end
      end
      statusColor = fif(is_busy, obj.busystatuscolor, obj.idlestatuscolor) ;
      set(obj.txStatus,'ForegroundColor',statusColor);      
      if ~isempty(obj.trainingMonitorVisualizer_) && isvalid(obj.trainingMonitorVisualizer_)
        obj.trainingMonitorVisualizer_.updatePointer() ;
      end
      if ~isempty(obj.trackingMonitorVisualizer_) && isvalid(obj.trackingMonitorVisualizer_)
        obj.trackingMonitorVisualizer_.updatePointer() ;
      end
      if ~isempty(obj.movieManagerController_) && obj.movieManagerController_.isValid()
        obj.movieManagerController_.updatePointer() ;
      end

      % Actually update the String in the status text box.  Use the shorter status
      % string from the labeler if the normal one is too long for the text box.
      raw_status_string = labeler.rawStatusString;
      has_project = labeler.hasProject ;
      project_file_path = labeler.projectfile ;
      status_string = ...
        interpolate_status_string(raw_status_string, has_project, project_file_path) ;
      set(obj.txStatus,'String',status_string) ;
      % If the textbox is overstuffed, change to the shorter status string
      extent = get(obj.txStatus,'Extent') ;  % reflects the size fo the String property
      position = get(obj.txStatus,'Position') ;  % reflects the size of the text box
      string_width = extent(3) ;
      box_width = position(3) ;
      if string_width > 0.95*box_width ,
        shorter_status_string = ...
          interpolate_shorter_status_string(raw_status_string, has_project, project_file_path) ;
        if isequal(shorter_status_string, status_string) ,
          % Sometimes the "shorter" status string is the same---don't change the
          % text box if that's the case
        else
          set(obj.txStatus,'String',shorter_status_string) ;
        end
      end

      % Make sure to update graphics now
      drawnow('nocallbacks') ;  
        % Please don't comment out the above drawnow() command!  We want the update of the
        % pointer to happen ASAP when we update the busy status.  This helps indicate
        % to the user that APT is working on something, and they don't need to actuate
        % the control again.  -- ALT, 2025-08-06
    end

    function updateBackgroundProcessingStatus_(obj)
      % Update obj.txBGTrain (the lower-right corner text box) is reflect the
      % current training/tracking bout.
      labeler = obj.labeler_ ;      
      isTrainingOrTracking = labeler.bgTrnIsRunning || labeler.bgTrkIsRunning ;
      if isTrainingOrTracking
        obj.txBGTrain.String = labeler.backgroundProcessingStatusString ;
        % obj.txBGTrain.ForegroundColor = LabelerController.busystatuscolor ;
        obj.txBGTrain.Visible = 'on' ;
      else
        % This below here is all wrong.  Should just be an update.
        % obj.txBGTrain.String = 'Idle' ;
        % obj.txBGTrain.ForegroundColor = obj.idlestatuscolor ;
        obj.txBGTrain.Visible = 'off' ;
      end
    end  % function

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
            obj.save();
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
      obj.menu_view_hide_labels.Checked = onIff(~lblCore.hideLabels) ;
    end
    
    function lblCoreStreamlinedChanged(obj)
      labeler = obj.labeler_ ;
      lblCore = labeler.lblCore ;
      obj.menu_setup_streamlined.Checked = onIff(lblCore.streamlined) ;
    end

    function pbTrack_actuated_(obj, source, event)
      obj.track_core_(source, event) ;
    end

    function menu_start_tracking_but_dont_call_python_actuated_(obj, source, event)
      obj.track_core_(source, event, 'do_call_apt_interface_dot_py', false) ;
    end
    
    function track_core_(obj, source, event, varargin)  %#ok<INUSD> 
      obj.labeler_.pushBusyStatus('Spawning tracking job...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus()));
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
      
      % Switch to watch cursor
      labeler = obj.labeler_ ;
      labeler.pushBusyStatus('Spawning training job...') ;  % Want to do this here, b/c the stuff in this method can take a while
      oc = onCleanup(@()(labeler.popBusyStatus()));
      drawnow;

      % Check for project, movie
      [doTheyExist, message] = labeler.doProjectAndMovieExist() ;
      if ~doTheyExist ,
        error(message) ;
      end
      if labeler.doesNeedSave ,
        res = questdlg('Project has unsaved changes. Save before training?','Save Project','Save As','No','Cancel','Save As') ;
        if strcmp(res,'Cancel')
          return
        elseif strcmp(res,'Save As')
          obj.menu_file_saveas_actuated_(source, event) ;
        end    
      end

      % See if the tracker is in a fit state to be trained
      [tfCanTrain, reason] = labeler.trackCanTrain() ;
      if ~tfCanTrain,
        error('Tracker not fit to be trained: %s', reason) ;
      end
      
      % See if the automatically-determined parameters differ from the currently set
      % ones.  If so, offer user the option to change to the auto-determined params.
      [~, ~, was_canceled] = obj.setAutoParams();
      if was_canceled 
        return
      end

      % Make sure we have enough GPU memory
      if ~labeler.trackCheckGPUMemGUI()
        return
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

    % function menu_track_tracking_algorithm_item_actuated_(obj, source, event)  %#ok<INUSD> 
    % 
    % end

    function menu_track_tracker_history_item_actuated_(obj, source, event)  %#ok<INUSD> 

      labeler = obj.labeler_;
      if labeler.tracker.bgTrnIsRunning
        uiwait(warndlg('Cannot switch tracker while training is in progress','Training in progress'));
        return;
      end
      if labeler.tracker.bgTrkIsRunning
        uiwait(warndlg('Cannot switch tracker while tracking is in progress.','Tracking in progress'));
        return;
      end

      obj.labeler_.pushBusyStatus('Switching tracker...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus()));
      drawnow;

      % Get the index of the tracker in the tracker history
      trackerHistoryIndex = source.UserData ;

      % Call the labeler method
      labeler = obj.labeler_ ;
      labeler.trackMakeExistingTrackerCurrentGivenIndex(trackerHistoryIndex) ;      
    end

    function showDialogAfterSpawningTrackingForGT(obj, source, event)  %#ok<INUSD> 
      % Event handler that gets called after the labeler successfully spawns jobs for GT.
      % Raises a non-modal dialog, and registers it as a 'satellite' window so we can delete
      % it when the main window closes.
      % labeler = obj.labeler_ ;
      DIALOGTTL = 'GT Tracking';
      msg = 'Tracking of GT frames spawned. GT results will be shown when tracking is complete.';
      h = msgbox(msg,DIALOGTTL);
      obj.addSatellite(h) ;  % register dialog to we can delete when main window closes
      %obj.satellites_(1,end+1) = h ;  % register dialog to we can delete when main window closes
    end

    function showGTResults(obj, source, event)  %#ok<INUSD> 
      % Event handler that gets called after the labeler finishes computing GT results.
      % Raises a dialog, and registers it as a 'satellite' window so we can delete
      % it when the main window closes.
      obj.createGTResultFigures_() ;
      % obj.satellites_(1,end+1) = h ;  % register dialog to we can delete when main window closes
    end

    function createGTResultFigures_(obj, varargin)
      labeler = obj.labeler_ ;      
      plotParams = labeler.gtPlotParams;
      t = labeler.gtTblRes;

      [fcnAggOverPts,~,~] = ...
        myparse(varargin,...
                'fcnAggOverPts',@(x)max(x,[],ndims(x)), ... % or eg @mean
                'aggLabel','Max', ...
                'lbli',1 ... % which example to plot
                );
      
      l2err = t.L2err;  % For MA, nframes x nanimals x npts.  For SA, nframes x npts
      fp = t.FP;
      fn = t.FN;
      % aggOverPtsL2err = fcnAggOverPts(l2err);  
      fcnAggOverPts(l2err);  
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
      nviews = labeler.nview;
      nphyspt = npts/nviews;

      if labeler.maIsMA,
        % note that we might have only a subset of views that are valid
        % for a given example, maybe should fix that

        % this reshape makes (nframes*maxnanimals) x npts
        l2err_reshaped = reshape(l2err,[],npts);  
        valid = ~all(isnan(l2err_reshaped),2);
        l2err_reshaped = reshape(l2err_reshaped,[],npts);
        % nvalidanimalframes x npts
        l2err_filtered = l2err_reshaped(valid,:);

        exampleLbl = t(1,:).pLbl(:,1,:);
        exampleLbl = reshape(exampleLbl,1,[]);
      else        
        % Why don't we need to filter for e.g. single-view SA?  -- ALT, 2024-11-21
        % probably don't need to filter for either, since computations are
        % nan-robust. But, the multi-animal mode has lots of extra nans
        % because it is represented as matrices with maxnanimals as one of
        % its dimensions
        % l2err is nframes x (nphyspt*nviews)
        valid = ~all(isnan(l2err),2);
        l2err_filtered = l2err(valid,:);
        exampleLbl = t(1,:).pLbl;
      end

      units = get(obj.mainFigure_,'Units');
      set(obj.mainFigure_,'Units','pixels');
      mainfig_pos = get(obj.mainFigure_,'Position');
      set(obj.mainFigure_,'Units',units);
      hmain = mainfig_pos(end);

      % circles around keypoints indicating prctiles of error
      fig_1 = figure('Name','Groundtruth error percentiles');
      %obj.satellites_(1,end+1) = fig_1 ;
      obj.addSatellite(fig_1) ;

      [allims,allpos] = labeler.cropTargetImageFromMovie(t.mov(1),t.frm(1),t.iTgt(1),exampleLbl);
      prcs = prctile(l2err_filtered,plotParams.prc_vals,1);
      prcs = reshape(prcs,[],nphyspt,nviews);
      nperkp = sum(~isnan(l2err_filtered),1);
      nperkp = reshape(nperkp,[nphyspt,nviews]);
      ntotal = sum(~all(isnan(reshape(l2err_filtered,[],nphyspt,nviews)),2),1);
      ntotal = reshape(ntotal,[nviews,1]);
      fp_all = sum(fp,'omitmissing');
      fn_all = sum(fn,'omitmissing');
      txtOffset = labeler.labelPointsPlotInfo.TextOffset;
      islight = plotPercentileCircles(allims,prcs,allpos,plotParams.prc_vals,fig_1,txtOffset,ntotal,fp_all,fn_all,labeler.maIsMA);
      figh = hmain*.75;
      hpx = max(cellfun(@(x) size(x,1),allims));
      wpx = sum(cellfun(@(x) size(x,2),allims));
      figw = figh*wpx/hpx+200;
      set(fig_1,'Position',[10,10,figw,figh]);
      centerfig(fig_1, obj.mainFigure_);

      % Err by landmark
      fig_2 = figure('Name','Groundtruth error per keypoint');
      %obj.satellites_(1,end+1) = fig_2 ;
      obj.addSatellite(fig_2) ;
      errs = reshape(l2err_filtered,[],nphyspt,nviews);
      PlotErrorHists(errs,'hparent',fig_2,'kpcolors',clrs,...
        'prcs',prcs,'prc_vals',plotParams.prc_vals,...
        'nbins',plotParams.nbins,'maxprctile',plotParams.prc_vals(end),...
        'kpnames',labeler.skelNames,'islight',islight,...
        'nperkp',nperkp,'fp',fp_all,'fn',fn_all,'isma',labeler.maIsMA,'ntotal',ntotal);
      figh = hmain;
      figw = figh/2*nviews;
      set(fig_2,'Position',[10,10,figw,figh]);
      centerfig(fig_2,obj.mainFigure_);
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
    
    function tfsucc = selectAwsInstanceGUI_(obj, varargin)
      [canLaunch,canConfigure,forceSelect] = ...
        myparse(varargin, ...
                'canlaunch',true,...
                'canconfigure',1,...
                'forceSelect',true);
            
      tfsucc = true;
      didLaunchNewInstance = false ;
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
          if ~tfsucc,
            return;
          end
          % For changing things in the model, we go through the top-level model object
          labeler.set_backend_property('awsPEM', pemFile) ;
          labeler.set_backend_property('awsKeyName', keyName) ;
          if ~awsec2.areCredentialsSet,
            reason = 'AWS EC2 instance is not configured.' ;
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
        opts = {'Attach to Existing','Cancel'};
        default = 'Cancel';
        if canLaunch,
          opts{end+1} = 'Launch New';
        end
        qstr = 'Launch a new instance or attach to an existing instance?';
        if ~awsec2.isInstanceIDSet,
          qstr = ['APT is not attached to an AWS EC2 instance. ',qstr];
        else
          qstr = sprintf('APT currently attached to AWS EC2 instance %s. %s',instanceID,qstr);
        end
        tstr = 'Specify AWS EC2 instance';
        btn = questdlg(qstr,tstr,opts{:},default);
        if isempty(btn) || strcmp(btn,'Cancel'),
          return;
        end
        while true,
          switch btn
            case 'Launch New'
              labeler.pushBusyStatus('Launching new AWS EC2 instance') ;
              drawnow;
              [didLaunchSucceed, instanceID] = labeler.launchNewAWSInstance() ;
              obj.labeler_.popBusyStatus() ;
              if ~didLaunchSucceed
                reason = 'Could not launch AWS EC2 instance.';
                error(reason) ;
              end
              didLaunchNewInstance = true ;
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
              else
                return
              end
              break
            otherwise
              % This is a cancel
              return
          end
        end
        % Set the instanceID in the model, if needed
        if didLaunchNewInstance
          % do nothing, the instance ID will already be set in the labeler
        else
          % For changing things in the model, we go through the top-level model object
          labeler.set_backend_property('awsInstanceID', instanceID) ;
        end
      end
    end  % function

    function exceptionMaybe = controlActuated(obj, controlName, source, event, varargin)  % public so that control actuation can be easily faked
      % The advantage of passing in the controlName, rather than,
      % say always storing it in the tag of the graphics object, and
      % then reading it out of the source arg, is that doing it this
      % way makes it easier to fake control actuations by calling
      % this function with the desired controlName and an empty
      % source and event.
      % obj.labeler_.pushBusyStatus(sprintf('Control %s actuated...', controlName)) ;
      % oc = onCleanup(@()(obj.labeler_.popBusyStatus())) ;
      if obj.isInYodaMode_ ,
        % "Do, or do not.  There is no try." --Yoda
        obj.controlActuatedCore_(controlName, source, event, varargin{:}) ;
        exceptionMaybe = {} ;
      else        
        try
          obj.controlActuatedCore_(controlName, source, event, varargin{:}) ;
          exceptionMaybe = {} ;
        catch exception
          obj.labeler_.popBusyStatus() ;
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

    function initWaitbar(obj)

      if ~isempty(obj.waitbarFigure_) && ishandle(obj.waitbarFigure_),
        return;
      end
      obj.waitbarFigure_ = waitbar(0, '', ...
                                   'Visible', 'off', ...
                                   'CreateCancelBtn', @(source,event)(obj.didCancelWaitbar())) ;
      obj.waitbarFigure_.CloseRequestFcn = @(source,event)(nop()) ;

    end


    function armWaitbar(obj)
      % When we arm, want to re-center figure on main window, then do a normal
      % update.
      obj.initWaitbar();
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
      obj.updateEnablementOfManyControls() ;
    end
    
    function updateTarget_(obj)
      main_figure = obj.mainFigure_ ;
      if isempty(main_figure) || ~isvalid(main_figure)
        return
      end      
      
      lObj = obj.labeler_ ;
      if (lObj.hasTrx || lObj.maIsMA) && ~lObj.isinit ,
        iTgt = lObj.currTarget;
        lObj.currImHud.updateTarget(iTgt);
          % lObj.currImHud is really a view object, but is stored in the Labeler for
          % historical reasons.  It should probably be stored in obj (the
          % LabelerController).  Someday we will move it, but right now it's referred to
          % by so many places in Labeler, and LabelCore, etc that I don't want to start
          % shaving that yak right now.  -- ALT, 2025-01-30
        obj.labelTLInfo.updateLabels();
        if lObj.gtIsGTMode
          tfHilite = lObj.gtCurrMovFrmTgtIsInGTSuggestions();
        else
          tfHilite = false;
        end
        obj.axesesHighlightManager_.setHighlight(tfHilite);
      end
    end  % function

    function updateEnablementOfManyControls(obj)
      % Enable/disable controls, as appropriate.

      % % Make sure the figure is legit
      % main_figure = obj.mainFigure_ ;
      % if isempty(main_figure) || ~isvalid(main_figure)
      %   return
      % end      

      % Determine the state from the state of the Labeler      
      labeler = obj.labeler_ ;
      hasProject = labeler.hasProject ;
      hasMovie = labeler.hasMovie ;  
        % Project has one or more movie specified.  
        % Note that hasMovie implies hasProject
      nview = labeler.nview ;
      isMultiView = nview>1 ;
      isSingleView = ~isMultiView ;
      isMA = labeler.maIsMA ;  % is a multi-animal project
        % Note that isMA implies isSingleView
      nLabelPointsAdd = labeler.nLabelPointsAdd ;
      isInCropMode = labeler.cropIsCropMode ;
      hasTracker = ~isempty(labeler.tracker);
        % Note that hasTracker implies hasProject
      isInGTMode = labeler.gtIsGTMode ;
        
      %
      % Update the enablement of the controls, depending on various aspects of the
      % label state
      %

      % Update the main menubar menus
      set(obj.menu_file,'Enable','on');
      set(obj.menu_view,'Enable',onIff(hasMovie));
      set(obj.menu_labeling_setup,'Enable',onIff(hasMovie));
      set(obj.menu_go,'Enable',onIff(hasMovie));
      set(obj.menu_track,'Enable',onIff(hasMovie));
      set(obj.menu_evaluate,'Enable',onIff(hasMovie||isInGTMode));
      set(obj.menu_help,'Enable','on');
      if ~isempty(obj.menu_debug) && isgraphics(obj.menu_debug)
        set(obj.menu_debug,'Enable',onIff(hasProject)) ;
      end

      % Update items in the File menu
      set(obj.menu_file_new,'Enable','on');
      set(obj.menu_file_save,'Enable',onIff(hasProject));
      set(obj.menu_file_saveas,'Enable',onIff(hasProject));
      set(obj.menu_file_load,'Enable','on');
      set(obj.menu_file_shortcuts,'Enable',onIff(hasProject));
      set(obj.menu_file_managemovies,'Enable',onIff(hasProject));
      set(obj.menu_file_import,'Enable',onIff(hasProject));
      set(obj.menu_file_export,'Enable',onIff(hasMovie));
      set(obj.menu_file_crop_mode,'Enable',onIff(hasMovie));
      set(obj.menu_file_clean_tempdir,'Enable',onIff(hasProject));
      set(obj.menu_file_bundle_tempdir,'Enable',onIff(hasProject));        
      set(obj.menu_file_quit,'Enable','on');
      
      % Update items in the View menu
      obj.updateTrxMenuCheckEnable();

      % Update setup menu item
      set(obj.menu_setup_label_outliers, 'Enable', onIff(hasMovie)) ;

      % These things
      set(obj.tbAdjustCropSize,'Enable',onIff(hasProject));
      set(obj.pbClearAllCrops,'Enable',onIff(hasProject));
      set(obj.pushbutton_exitcropmode,'Enable',onIff(hasProject));

      % Crop mode stuff
      set(obj.uipanel_cropcontrols,'Visible',onIff(hasProject && isInCropMode)) ;
      set(obj.text_trackerinfo,'Visible',onIff(hasProject && ~isInCropMode)) ;

      obj.updateTimeline() ;

      set(obj.pbClear,'Enable',onIff(hasProject));
      set(obj.tbAccept,'Enable',onIff(hasProject));
      set(obj.pbRecallZoom,'Enable',onIff(hasProject));
      set(obj.pbSetZoom,'Enable',onIff(hasProject));
      set(obj.pbResetZoom,'Enable',onIff(hasProject));
      set(obj.sldZoom,'Enable',onIff(hasProject));
      set(obj.pbPlaySeg,'Enable',onIff(hasProject));
      set(obj.pbPlaySegRev,'Enable',onIff(hasProject));
      set(obj.pbPlay,'Enable',onIff(hasProject));
      set(obj.slider_frame,'Enable',onIff(hasProject));
      set(obj.edit_frame,'Enable',onIff(hasProject));
      set(obj.popupmenu_prevmode,'Enable',onIff(hasProject));
      set(obj.pushbutton_freezetemplate,'Enable',onIff(hasProject));
      %set(obj.toolbar,'Visible',onIff(hasProject)) ;
      
      obj.menu_track.Enable = onIff(hasTracker);
      obj.pbTrain.Enable = onIff(hasTracker);
      obj.pbTrack.Enable = onIff(hasTracker);
      obj.menu_view_showhide_predictions.Enable = onIff(hasTracker);
      set(obj.menu_track_auto_params_update, 'Checked', hasProject && labeler.trackAutoSetParams) ;
      
      set(obj.menu_go_targets_summary,'Enable',onIff(hasProject && ~isInGTMode)) ;

      set(obj.menu_setup_sequential_mode,'Visible',onIff(hasMovie && isSingleView && ~isMA)) ;
      set(obj.menu_setup_template_mode,'Visible',onIff(hasMovie && isSingleView && ~isMA)) ;
      set(obj.menu_setup_highthroughput_mode,'Visible',onIff(hasMovie && isSingleView && ~isMA)) ;
      set(obj.menu_setup_multiview_calibrated_mode_2,'Visible',onIff(hasMovie && isMultiView));
      set(obj.menu_setup_multianimal_mode,'Visible',onIff( hasMovie && isMA));
      set(obj.menu_setup_sequential_add_mode, 'Visible', onIff(hasMovie && isSingleView && nLabelPointsAdd~=0)) ;
    end  % function

    function update_text_trackerinfo(obj)
      % Updates the tracker info string to match what's in 
      % obj.labeler_.tracker.trackerInfo.
      % Called via notify() when labeler.tracker.trackerInfo is changed.
      
      % Get the handles out of the main figure
      main_figure = obj.mainFigure_ ;
      if isempty(main_figure) || ~isvalid(main_figure)
        return
      end      
      
      % Update the relevant text object
      tracker = obj.labeler_.tracker ;
      if ~isempty(tracker) ,
        obj.text_trackerinfo.String = tracker.getTrackerInfoString();
      end
    end  % function

    function raiseTargetsTableFigure_(obj)
      labeler = obj.labeler_ ;
      labeler.pushBusyStatus('Making figure for big summary table...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus())) ;      
      drawnow;
      main_figure = obj.mainFigure_ ;
      [tfok,tblBig] = labeler.hlpTargetsTableUIgetBigTable();
      if ~tfok
        return
      end
      
      tblSumm = labeler.trackGetSummaryTable(tblBig) ;
      hF = figure('Name','Target Summary (click row to navigate)',...
                  'MenuBar','none','Visible','off', ...
                  'Tag', 'target_table_figure'); %#ok<*CPROP>
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

      obj.labeler_.pushBusyStatus('Switching target...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus()));
      drawnow;

      labeler = obj.labeler_ ;
      labeler.setMFTGUI(rowdata.mov,rowdata.frm1,rowdata.iTgt) ;
    end  % function

    function target_table_update_button_actuated_(obj, source, event)  %#ok<INUSD>
      % Does what needs doing when the target table update button is actuated.

      obj.labeler_.pushBusyStatus('Updating target table...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus()));
      drawnow;

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
    
    function result = isSatellite(obj, h)
      result = any(obj.satellites_ == h) ;
    end

    function h = findSatelliteByTag_(obj, query_tag)
      % Find the handle with Tag query_tag in obj.depHandles.
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
      nt.setData(tbl);
      hF.UserData = nt;
      obj.addSatellite(hF);
    end  % function

    function susp_frame_table_row_actuated_(obj, source, event, row, rowdata)  %#ok<INUSD>
      % Does what needs doing when the suspicious frame table row is selected.

      obj.labeler_.pushBusyStatus('Switching to suspicious [movie, frame, target]...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus()));
      drawnow;

      labeler = obj.labeler_ ;
      labeler.suspCbkTblNavedGUI(row) ;
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
      if ~isempty(obj.trackingMonitorVisualizer_) && isvalid(obj.trackingMonitorVisualizer_)
        labeler = obj.labeler_ ;
        pollingResult = labeler.tracker.bgTrkMonitor.pollingResult ;
        obj.trackingMonitorVisualizer_.resultsReceived(pollingResult) ;
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
      if ~isempty(obj.trainingMonitorVisualizer_) && isvalid(obj.trainingMonitorVisualizer_) 
        labeler = obj.labeler_ ;
        pollingResult = labeler.tracker.bgTrnMonitor.pollingResult ;
        obj.trainingMonitorVisualizer_.resultsReceived(pollingResult) ;
      end
    end  % function

    function addSatellite(obj, h)
      % Add a 'satellite' figure, so we don't lose track of them.

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

    function raiseTrainingEndedDialog_(obj)
      % Raise a dialog that reports how many training iterations have completed, and
      % ask if the user wants to save the project.  Normally called via event
      % notification after training ends.
      labeler = obj.labeler_ ;
      tracker = labeler.tracker ;
      iterCurr = tracker.trackerInfo.iterCurr ;  % a row vector, in general
      iterFinal = tracker.trackerInfo.iterFinal ;
      n_out_of_d_string = DeepTracker.printIter(iterCurr, iterFinal) ;
      if ~all(isnan(iterCurr)) ,
        if labeler.lastTrainEndCause == EndCause.complete
          question_string = sprintf('Training completed %s iterations.  Save project now?',...
                                    n_out_of_d_string) ;
        elseif labeler.lastTrainEndCause == EndCause.error
          question_string = sprintf('Training errored after %s iterations.  (See console for details.)  Save project now?',...
                                    n_out_of_d_string) ;
        elseif labeler.lastTrainEndCause == EndCause.abort
          question_string = sprintf('Training was aborted after %s iterations.  Save project now?',...
                                    n_out_of_d_string) ;
        else
          error('Internal error.  Please save your work if possible, restart APT, and report to the APT developers.') ;
        end        
        res = questdlg(question_string,'Save?','Save','Save as...','No','Save') ;  % modal
        if strcmpi(res,'Save'),
          obj.save();
        elseif strcmpi(res,'Save as...'),
          obj.saveAs();
        else
          % do nothing
        end  % if      
      else
        % all(isnan(iterCurr)) == true
        % This means there was an error or abort early in training.
        if labeler.lastTrainEndCause == EndCause.complete
          uiwait(errordlg(sprintf('Training allegedly completed, but after %s iterations.  Odd.', n_out_of_d_string), ...
                          'Strangeness', ...
                          'modal')) ;          
        elseif labeler.lastTrainEndCause == EndCause.error
          uiwait(errordlg(sprintf('Training errored after %s iterations.  See console for details.', n_out_of_d_string), ...
                          'Training Error', ...
                          'modal')) ;
        elseif labeler.lastTrainEndCause == EndCause.abort
          % just proceed on abort
        else
          error('Internal error.  Please save your work if possible, restart APT, and report to the APT developers.') ;
        end
      end
    end  % function

    function raiseTrackingEndedDialog_(obj)
      % Raise a dialog if tracking hit an error.  Normally called via event
      % notification after training ends.
      labeler = obj.labeler_ ;
      if labeler.lastTrackEndCause == EndCause.error
          uiwait(errordlg('Error while tracking.  See console for details.', ...
                          'Tracking Error', ...
                          'modal')) ;
      else
        % Don't want a dialog on abort or complete (or undefined).
      end        
    end  % function

    function didCreateNewProject(obj)
      labeler =  obj.labeler_ ;
      
      obj.clearSatellites() ;
      
      % Initialize the uitable of labeled frames
      obj.initTblFramesTrx_() ;
      
      % figs, axes, images
      deleteValidGraphicsHandles(obj.figs_all(2:end));
      obj.figs_all = obj.figs_all(1);
      obj.axes_all = obj.axes_all(1);
      obj.images_all = obj.images_all(1);
      obj.axes_occ = obj.axes_occ(1);
      
      nview = labeler.nview;
      figs = gobjects(1,nview);
      axs = gobjects(1,nview);
      ims = gobjects(1,nview);
      axsOcc = gobjects(1,nview);
      figs(1) = obj.figs_all;
      axs(1) = obj.axes_all;
      ims(1) = obj.images_all;
      axsOcc(1) = obj.axes_occ;
      
      % all occluded-axes will have ratios widthAxsOcc:widthAxs and 
      % heightAxsOcc:heightAxs equal to that of axsOcc(1):axs(1)
      axsOcc1Pos = axsOcc(1).Position;
      ax1Pos = axs(1).Position;
      axOccSzRatios = axsOcc1Pos(3:4)./ax1Pos(3:4);
      axOcc1XColor = axsOcc(1).XColor;
      
      set(ims(1),'CData',0); % reset image
      %controller = obj.controller ;
      for iView=2:nview
        thisfig = ...
          figure('CloseRequestFcn',@(s,e)(obj.cbkAuxFigCloseReq(s,e)),...
                 'Color',figs(1).Color, ...
                 'UserData',struct('view',iView), ...
                 'Tag', sprintf('figs_all(%d)', iView) ...
                 );
        figs(iView) = thisfig ;
        axs(iView) = axes('Parent', thisfig, 'Position', [0,0,1,1]) ;
        obj.addSatellite(thisfig) ;

        % Set up the figure toolbar how we want it
        makeFigureMenubarAndToolbarAPTAppropriateBang(thisfig) ;
        
        ims(iView) = imagesc(0,'Parent',axs(iView));  % N.B.: this clears any Tag property set on the axes...
        set(ims(iView),'PickableParts','none');
        %axisoff(axs(iView));
        hold(axs(iView),'on');  % Do we still need/want this?
        set(axs(iView),'Color',[0 0 0]);
        set(axs(iView),'Tag','axes_curr');
        
        axparent = axs(iView).Parent;
        axpos = axs(iView).Position;
        axunits = axs(iView).Units;
        axpos(3:4) = axpos(3:4).*axOccSzRatios;
        axsOcc(iView) = ...
          axes('Parent',axparent,'Position',axpos,'Units',axunits,...
               'Color',[0 0 0],'Box','on','XTick',[],'YTick',[],'XColor',axOcc1XColor,...
               'YColor',axOcc1XColor, ...
               'Tag', 'axes_occ') ;
        hold(axsOcc(iView),'on');
        axis(axsOcc(iView),'ij');

        % Hide axes toolbar
        axes_toolbar = axtoolbar(axs(iView), 'default');
        axes_toolbar.Visible = 'off';        
      end  % for loop over non-primary view figures
      obj.figs_all = figs;
      obj.axes_all = axs;
      obj.images_all = ims;
      obj.axes_occ = axsOcc;
      
      % AL 20191002 This is to enable labeling simple projs without the Image
      % toolbox (crop init uses imrect)
      try
        obj.cropInitImRects_() ;
      catch ME
        fprintf(2,'Crop Mode initialization error: %s\n',ME.message);
      end
      
      if ~isempty(obj.axesesHighlightManager_)
        % Explicit deletion not supposed to be nec
        delete(obj.axesesHighlightManager_);
      end
      obj.axesesHighlightManager_ = AxesHighlightManager(axs);
      
      axis(obj.axes_occ,[0 labeler.nLabelPoints+1 0 2]);
      
      % Delete obj.hLinkPrevCurr
      % The link destruction/recreation may not be necessary
      if ~isempty(obj.hLinkPrevCurr) && isvalid(obj.hLinkPrevCurr)
        delete(obj.hLinkPrevCurr);
        obj.hLinkPrevCurr = [] ;
      end

      % Configure the non-primary view windows
      viewCfg = labeler.projPrefs.View;
      obj.newProjAxLimsSetInConfig = ...
        obj.hlpSetConfigOnViews_(viewCfg, ...
                                 viewCfg(1).CenterOnTarget) ;  % lObj.CenterOnTarget is not set yet
      AX_LINKPROPS = {'XLim' 'YLim' 'XDir' 'YDir'};
      obj.hLinkPrevCurr = ...
        linkprop([obj.axes_curr,obj.axes_prev], AX_LINKPROPS) ;
      
      arrayfun(@(x)(colormap(x,gray())),figs);
      obj.updateGUIFigureNames() ;
      obj.updateMainAxesName();
      
      arrayfun(@(fig)zoom(fig,'off'),obj.figs_all);  % Cannot set KPF if zoom or pan is on
      arrayfun(@(fig)pan(fig,'off'),obj.figs_all);
      hTmp = findall(obj.figs_all,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',@(src,evt)(obj.cbkKPF(src,evt))) ;
      obj.h_ignore_arrows = [obj.slider_frame];
      %set(obj.figs_all,'WindowButtonMotionFcn',@(src,evt)cbkWBMF(src,evt,lObj));
      %set(obj.figs_all,'WindowButtonUpFcn',@(src,evt)cbkWBUF(src,evt,lObj));
      % if ispc
      %   set(obj.figs_all,'WindowScrollWheelFcn',@(src,evt)cbkWSWF(src,evt,lObj));
      % end
      
      % eg when going from proj-with-trx to proj-no-trx, targets table needs to
      % be cleared
      obj.setTblTrxData(cell(0,size(obj.tblTrx.ColumnName,2)));
      
      obj.setShortcuts_() ;
      
      obj.labelTLInfo.updateForNewProject();
      
      deleteValidGraphicsHandles(obj.movieManagerController_) ;
      obj.movieManagerController_ = [];
      % t0 = tic;
      % obj.movieManagerController_ = MovieManagerController(labeler) ;
      % fprintf('Creating movie manager takes %f s\n',toc(t0));
      % obj.movieManagerController_.setVisible(false);
      
      % obj.GTManagerFigure = GTManager(labeler);
      % obj.GTManagerFigure.Visible = 'off';
      % obj.addSatellite(obj.GTManagerFigure) ;
    end  % function

    function menu_file_new_actuated_(obj, ~, ~)
      % Create a new project
      labeler = obj.labeler_ ;
      if obj.raiseUnsavedChangesDialogIfNeeded() ,
        cfg = ProjectSetup(obj.mainFigure_);  % launches the project setup window
        if ~isempty(cfg)    
          labeler.projNew(cfg);
          if ~isempty(obj.movieManagerController_) && obj.movieManagerController_.isValid() ,
            obj.movieManagerController_.setVisible(true);
          else
            obj.movieManagerController_ = MovieManagerController(obj.labeler_);
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
      obj.updateStatusAndPointer() ;      
    end  % function

    function didChangeProjFSInfo(obj)
      obj.updateMainFigureName() ;
      obj.updateStatusAndPointer() ;      
    end  % function

    function didChangeMovieInvert(obj)
      obj.updateGUIFigureNames() ;
      obj.updateMainAxesName() ;
    end  % function

    function updateGUIFigureNames(obj)
      labeler = obj.labeler_ ;
      figs = obj.figs_all ;

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

    function updateMainAxesName(obj)
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
      set(obj.txMoviename,'String',str) ;
    end  % function
    
    function setShortcuts_(obj)
      labeler = obj.labeler_ ;
      main_figure = obj.mainFigure_ ;
      prefs = labeler.projPrefs;
      if ~isfield(prefs,'Shortcuts')
        return;
      end
      prefs = prefs.Shortcuts;
      fns = fieldnames(prefs);
      ismenu = false(1,numel(fns));
      for i = 1:numel(fns)
        h = findobj(main_figure,'Tag',fns{i},'-property','Accelerator');
        if isempty(h) || ~ishandle(h) || ...
            (ismember(fns{i},obj.fakeMenuTags) && isprop(h,'Visible') && strcmpi(h.Visible,'off')),
          continue;
        end
        ismenu(i) = true;
        set(h,'Accelerator',prefs.(fns{i}));
      end
      obj.shortcutkeys = cell(1,nnz(~ismenu));
      obj.shortcutfns = cell(1,nnz(~ismenu));
      idxnotmenu = find(~ismenu);
      for ii = 1:numel(idxnotmenu)
        i = idxnotmenu(ii);
        obj.shortcutkeys{ii} = prefs.(fns{i});
        obj.shortcutfns{ii} = fns{i};
      end
    end  % function

    function match = matchShortcut(obj,evt)
      match = {};
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      tfShft = any(strcmp('shift',modifier));
      if ~tfCtrl,
        return;
      end
      scs = obj.labeler_.getShortcuts();
      keys = struct2cell(scs);
      iscap = cellfun(@(x) strcmp(x,upper(x)),keys);
      ismatch = strcmp(keys,key);
      if ~any(ismatch),
        return;
      end
      if tfShft,
        i = find(iscap & ismatch);
      else
        i = find(~iscap & ismatch);
      end
      if isempty(i),
        return;
      end
      fns = fieldnames(scs);
      tagsmatch = fns(i);
      keysmatch = keys(i);
      match = [tagsmatch(:),keysmatch(:)];
    end

    function menu_file_shortcuts_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_;
      labeler.pushBusyStatus('Editing keyboard shortcuts...');
      oc = onCleanup(@()(labeler.popBusyStatus())) ;
      drawnow;
      uiwait(ShortcutsDialog(obj));

    end  % function

    function cropInitImRects_(obj)
      deleteValidGraphicsHandles(obj.cropHRect);
      obj.cropHRect = ...
        arrayfun(@(x)imrect(x,[nan nan nan nan]),obj.axes_all,'uni',0); %#ok<IMRECT>
      obj.cropHRect = cat(1,obj.cropHRect{:}); % ML 2016a ish can't concat imrects in arrayfun output
      arrayfun(@(x)set(x,'Visible','off','PickableParts','none','UserData',true),...
        obj.cropHRect); % userdata: see cropImRectSetPosnNoPosnCallback
      for ivw=1:numel(obj.axes_all)
        posnCallback = @(zpos)cbkCropPosn(obj,zpos,ivw);
        obj.cropHRect(ivw).addNewPositionCallback(posnCallback);
      end
    end  % function

    function cbkCropPosn(obj,posn,iview)
      labeler = obj.labeler_ ;
      tfSetPosnLabeler = get(obj.cropHRect(iview),'UserData');
      if tfSetPosnLabeler
        [roi,roiw,roih] = CropInfo.rectPos2roi(posn);
        tb = obj.tbAdjustCropSize;
        if tb.Value==tb.Max  % tbAdjustCropSizes depressed; using as proxy for, imrect is resizable
          fprintf('roi (width,height): (%d,%d)\n',roiw,roih);
        end
        labeler.cropSetNewRoiCurrMov(iview,roi);
      end
    end  % function

    function menu_view_reset_views_actuated_(obj, source, event)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      viewCfg = labeler.projPrefs.View;
      obj.hlpSetConfigOnViews_(viewCfg, labeler.movieCenterOnTarget) ;
      movInvert = ViewConfig.getMovieInvert(viewCfg);
      labeler.movieInvert = movInvert;
      labeler.movieCenterOnTarget = viewCfg(1).CenterOnTarget;
      labeler.movieRotateTargetUp = viewCfg(1).RotateTargetUp;
    end  % function
    
    function tfKPused = cbkKPF(obj, source, event)

      labeler = obj.labeler_ ;
      if ~labeler.isReady ,
        return
      end      
      tfKPused = false;
      isarrow = ismember(event.Key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'});
      if isarrow && ismember(source,obj.h_ignore_arrows),
        return
      end
      
      tfShift = any(strcmp('shift',event.Modifier));
      tfCtrl = any(strcmp('control',event.Modifier));
      
      lcore = labeler.lblCore;
      if ~isempty(lcore)
        tfKPused = lcore.kpf(source,event);
        if tfKPused
          return
        end
      end

      if ~isempty(obj.shortcutkeys) && ~isempty(obj.shortcutfns)
        % control key pressed?
        if tfCtrl && numel(event.Modifier)==1 && any(strcmpi(event.Key,obj.shortcutkeys))
          i = find(strcmpi(event.Key,obj.shortcutkeys),1);
          if ~ismember(obj.shortcutfns{i},labeler.lblCore.unsupportedKPFFns),
            h = findobj(obj.mainFigure_,'Tag',obj.shortcutfns{i});
            if isprop(h,'Callback'),
              cb = h.Callback;
            elseif isprop(h,'MenuSelectedFcn'),
              cb = h.MenuSelectedFcn;
            else
              cb = [];
            end
            if isempty(cb)
              fprintf('Unknown shortcut handle %s\n',obj.shortcutfns{i});
            else
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
      end
      if tfKPused
        return
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
                labeler.setFrameProtectedGUI(f);
              end
            else
              labeler.frameDownGUI(tfCtrl);
            end
            tfKPused = true;
          case 'rightarrow'
            if tfShift
              sam = labeler.movieShiftArrowNavMode;
              samth = labeler.movieShiftArrowNavModeThresh;
              samcmp = labeler.movieShiftArrowNavModeThreshCmp;
              [tffound,f] = sam.seekFrame(labeler,1,samth,samcmp);
              if tffound
                labeler.setFrameProtectedGUI(f);
              end
            else
              labeler.frameUpGUI(tfCtrl);
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
          
    % function menu_file_quick_open_actuated_(obj, source, event)  %#ok<INUSD>
    %   lObj = obj.labeler_ ;
    %   if obj.raiseUnsavedChangesDialogIfNeeded() ,
    %     [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(false);
    %     if ~tfsucc
    %       return;
    %     end
    % 
    %     movfile = movfile{1};
    %     trxfile = trxfile{1};
    % 
    %     cfg = Labeler.cfgGetLastProjectConfigNoView() ;
    %     if cfg.NumViews>1
    %       warndlg('Your last project had multiple views. Opening movie with single view.');
    %       cfg.NumViews = 1;
    %       cfg.ViewNames = cfg.ViewNames(1);
    %       cfg.View = cfg.View(1);
    %     end
    %     lm = LabelMode.(cfg.LabelMode);
    %     if lm.multiviewOnly
    %       cfg.LabelMode = char(LabelMode.TEMPLATE);
    %     end
    % 
    %     [~,projName,~] = fileparts(movfile);
    %     cfg.ProjectName = projName ;
    %     lObj.projNew(cfg);
    %     lObj.movieAdd(movfile,trxfile);
    %     lObj.movieSetGUI(1,'isFirstMovie',true);      
    %   end
    % end  % function
    
    function projAddLandmarks(obj, nadd)
      % Function to add new kinds of landmarks to an existing project.  E.g. If you
      % had a fly .lbl file where you weren't tracking the wing tips, but then you
      % wanted to start tracking the wingtips, you would call this function.
      % Currently not exposed in the GUI, probably should be eventually.

      labeler = obj.labeler_ ;

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
                                       obj.axes_prev,labeler.labelPointsPlotInfo);
      if ~isempty(labeler.prevAxesModeInfo)
        labeler.prevAxesLabelsRedraw();
      end
      
      % init info timeline
      obj.labelTLInfo.updateForNewProject();
      obj.labelTLInfo.updateLabels();
      
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
      set(obj.menu_setup_sequential_add_mode,'Visible','on');
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
      set(obj.menu_setup_sequential_add_mode,'Visible','off');
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
    
    % function update_menu_track_tracking_algorithm_quick(obj)
    %   % Update the Track > 'Tracking algorithm' submenu.
    %   % This essentially means updating what elements are checked or not.
    % 
    %   % Get out the main objects
    %   labeler = obj.labeler_ ;
    %   if labeler.isinit || ~labeler.hasProject ,
    %     return
    %   end
    % 
    %   % Remake the submenu items
    %   menus = obj.menu_track_tracking_algorithm.Children ;
    %   trackers = labeler.trackersAll ;
    %   trackerCount = numel(trackers) ;
    %   isMatch = labeler.doesCurrentTrackerMatchFromTrackersAllIndex() ;
    %   for i=1:trackerCount
    %     menu = menus(i) ;
    %     menuTrackersAllIndex = menu.UserData ;
    %     menu.Checked = onIff(isMatch(menuTrackersAllIndex)) ;
    %   end
    % end  % function

    function update_menu_track_tracker_history(obj)
      % Populate the Track > 'Tracking algorithm' submenu.
      % This deletes all the menu items and then remakes them.

      % Get out the main objects
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end      

      % Delete the old submenu items
      menu_track_tracker_history = obj.menu_track_tracker_history ;
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
               'Tag',tag, ...
               'UserData',i, ...
               'Position',i, ...
               'Checked', onIff(i==1)) ;  
          % The first element of labeler.trackerHistory is always the current one
      end

      % Set up the figure callbacks to call obj, using the tag to determine the
      % method name.
      visit_children(menu_track_tracker_history, @set_standard_callback_if_none_bang, obj) ;      
    end  % function

    % function update_menu_track_tracker_history_(obj)
    %   % Populate the Track > 'Tracking algorithm' submenu.
    % 
    %   % Get out the main objects
    %   labeler = obj.labeler_ ;
    %   if labeler.isinit ,
    %     return
    %   end      
    % 
    %   % Delete the old submenu items
    %   menu_track_tracker_history = obj.menu_track_tracker_history ;
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
      
      if ~isempty(obj.menu_track_backend_config_jrc)
        % set up first time only, should not change
        return
      end
      % moved this to createLabelerMainFigure
      % obj.menu_track_backend_config = uimenu( ...
      %   'Parent',obj.menu_track,...
      %   'Label','Backend configuration',...
      %   'Visible','on',...
      %   'Tag','menu_track_backend_config');
      % moveMenuItemAfter(obj.menu_track_backend_config, obj.menu_track_tracker_history) ;
      obj.menu_track_backend_config_docker = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','Docker',...
        'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
        'Tag','menu_track_backend_config_docker',...
        'userdata',DLBackEnd.Docker);  
      obj.menu_track_backend_config_conda = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','Conda',...
        'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
        'Tag','menu_track_backend_config_conda',...
        'userdata',DLBackEnd.Conda,...
        'Visible',true,...
        'Enable','on');
      obj.menu_track_backend_config_jrc = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','JRC Cluster',...
        'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
        'Tag','menu_track_backend_config_jrc',...
        'userdata',DLBackEnd.Bsub);
      obj.menu_track_backend_config_aws = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','AWS Cloud',...
        'Callback',@(s,e)(obj.cbkTrackerBackendMenu(s,e)),...
        'Tag','menu_track_backend_config_aws',...
        'userdata',DLBackEnd.AWS);

      obj.menu_track_backend_settings = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','Settings...',...
        'Callback',@(s,e)(obj.cbkTrackerBackendSettings(s,e)),...
        'Tag','menu_track_backend_settings',...
        'Separator','on');

      obj.menu_track_backend_config_test = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','Test backend configuration',...
        'Tag','menu_track_backend_config_test');

      % KB added menu item to get more info about how to set up
      obj.menu_track_backend_config_moreinfo = uimenu( ...
        'Parent',obj.menu_track_backend_config,...
        'Label','More information...',...
        'Callback',@(s,e)(obj.cbkTrackerBackendMenuMoreInfo()),...
        'Tag','menu_track_backend_config_moreinfo');   

      % Set up the figure callbacks to call obj, using the tag to determine the
      % method name.  (Most have custom callbacks, but a few use standard ones.
      visit_children(obj.menu_track_backend_config, @set_standard_callback_if_none_bang, obj) ;            
    end  % function

    function cbkTrackerBackendMenu(obj, source, event)  %#ok<INUSD>
      lObj = obj.labeler_ ;
      beType = source.UserData;
      lObj.set_backend_property('type', beType) ;
    end  % function

    function cbkTrackerBackendSettings(obj, varargin)
      labeler = obj.labeler_;
      
      beType = labeler.trackDLBackEnd.type;
      if beType==DLBackEnd.Bsub,
        obj.cbkTrackerBackendJRCSettings(varargin{:});
      elseif beType==DLBackEnd.Docker,
        obj.cbkTrackerBackendDockerSettings(varargin{:});
      elseif beType==DLBackEnd.Conda,
        obj.cbkTrackerBackendSetCondaEnv();
      elseif beType == DLBackEnd.AWS,
        obj.selectAwsInstanceGUI_('canlaunch',true,...
          'canconfigure',2, ...
          'forceSelect',true) ;
      else
        error('Unknown backend %s',beType);
      end

    end

    function cbkTrackerBackendJRCSettings(obj,varargin)
      uiwait(JRCBackEndSettingsDialog(obj));
    end

    function cbkTrackerBackendDockerSettings(obj,varargin)
      uiwait(DockerBackEndSettingsDialog(obj));
    end


    function cbkTrackerBackendMenuMoreInfo(obj)
      lObj = obj.labeler_ ;
      res = web(lObj.DLCONFIGINFOURL,'-new');
      if res ~= 0,
        msgbox({'Information on configuring Deep Learning GPU/Backends can be found at'
                'https://github.com/kristinbranson/APT/wiki/Deep-Neural-Network-Tracking.'},...
                'Deep Learning GPU/Backend Information','replace');
      end
    end  % function

    function menu_track_backend_config_test_actuated_(obj, ~, ~)
      obj.labeler_.pushBusyStatus('Testing backend...');
      oc = onCleanup(@()(obj.labeler_.popBusyStatus()));
      obj.backendTestController_ = BackendTestController(obj, obj.labeler_) ;
      obj.labeler_.testBackendConfig() ;
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
      % changes.

      % Get the objects we need to mess with
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end 
      tracker = labeler.tracker ;

      % Enable/disable controls that depend on whether a tracker is available
      tfTracker = ~isempty(tracker) ;
      onOrOff = onIff(tfTracker && labeler.isReady) ;
      obj.menu_track.Enable = onOrOff;
      obj.pbTrain.Enable = onOrOff;
      obj.pbTrack.Enable = onOrOff;
      obj.menu_view_showhide_predictions.Enable = onOrOff;

      % % Remake the tracker history submenu
      % obj.update_menu_track_tracker_history_() ;

      % Update the check marks in menu_track_backend_config menu
      obj.update_menu_track_backend_config();

      % Update the InfoTimelineController
      obj.labelTLInfo.updateLabels();
    end  % function
    
    function updateTrainingMonitor(obj)
      obj.trainingMonitorVisualizer_.update() ;
    end  % function

    function cbkTrackerTrainEnd(obj)
      labeler = obj.labeler_ ;
      if ~labeler.silent ,
        obj.raiseTrainingEndedDialog_() ;
      end
      obj.update() ;
    end  % function

    function updateTrackingMonitor(obj)
      obj.trackingMonitorVisualizer_.update() ;
    end  % function

    function cbkTrackerEnd(obj)
      labeler = obj.labeler_ ;
      if ~labeler.silent ,
        obj.raiseTrackingEndedDialog_() ;
      end
      obj.update() ;
    end  % function

    function updateShowPredMenus(obj)
      lObj = obj.labeler_ ;
      tracker = lObj.tracker ;
      if isempty(tracker),
        return;
      end
      obj.menu_view_showhide_preds_all_targets.Checked = onIff(~tracker.hideViz && ~tracker.showPredsCurrTargetOnly) ;
      obj.menu_view_showhide_preds_curr_target_only.Checked = onIff(~tracker.hideViz && tracker.showPredsCurrTargetOnly) ;
      obj.menu_view_showhide_preds_none.Checked = onIff(tracker.hideViz) ;
    end

    function updateShowImportedPredMenus(obj,src,evt) %#ok<INUSD>
      labeler = obj.labeler_ ;
      if nargin < 2,
        src = nan;
      end
      % during initiatialization, these have not been set to bools yet
      if isempty(labeler.labels2ShowCurrTargetOnly),
        showcurrent = false;
      else
        showcurrent = labeler.labels2ShowCurrTargetOnly;
      end
      if isempty(labeler.labels2Hide),
        hide = false;
      else
        hide = labeler.labels2Hide;
      end
      if src ~= obj.menu_view_showhide_imported_preds_all,
        obj.menu_view_showhide_imported_preds_all.Checked = onIff(~hide && ~showcurrent) ;
      end
      if src ~= obj.menu_view_showhide_imported_preds_curr_target_only,
        obj.menu_view_showhide_imported_preds_curr_target_only.Checked = onIff(~hide && showcurrent);
      end
      if src ~= obj.menu_view_showhide_imported_preds_none
        obj.menu_view_showhide_imported_preds_none.Checked = onIff(hide) ;
      end
      
    end



    function cbkTrackerHideVizChanged(obj)
      % lObj = obj.labeler_ ;
      % tracker = lObj.tracker ;
      obj.updateShowPredMenus()
    end  % function

    function cbkTrackerShowPredsCurrTargetOnlyChanged(obj)
      % lObj = obj.labeler_ ;
      % tracker = lObj.tracker ;
      obj.updateShowPredMenus();
      % obj.menu_view_showhide_preds_curr_target_only.Checked = onIff(tracker.showPredsCurrTargetOnly) ;
      % obj.menu_view_showhide_preds_all_targets.Checked = onIff(~tracker.showPredsCurrTargetOnly) ;
      % obj.menu_view_showhide_preds_none.Checked = onIff(~tracker.showPredsCurrTargetOnly) ;
    end  % function

    function update_menu_track_backend_config(obj)
      labeler = obj.labeler_ ;
      if isempty(obj.menu_track_backend_config_jrc) 
        % Early return if the menus have not been set up yet
        return
      end      
      if ~labeler.hasProject
        % The whole menu_track should be disabled already in this case
        return
      end
      beType = labeler.trackDLBackEnd.type;
      oiBsub = onIff(beType==DLBackEnd.Bsub);
      oiDckr = onIff(beType==DLBackEnd.Docker);
      oiCnda = onIff(beType==DLBackEnd.Conda);
      oiAWS = onIff(beType==DLBackEnd.AWS);
      set(obj.menu_track_backend_config_jrc,'checked',oiBsub);
      set(obj.menu_track_backend_config_docker,'checked',oiDckr);
      set(obj.menu_track_backend_config_conda,'checked',oiCnda, 'Enable', onIff(~ispc()));
      set(obj.menu_track_backend_config_aws,'checked',oiAWS);
      set(obj.menu_track_backend_settings,'Enable','on');
    end  % function
    
    % function cbkTrackerBackendSetDockerSSH(obj)
    %   lObj = obj.labeler_ ;
    %   assert(lObj.trackDLBackEnd.type==DLBackEnd.Docker);
    %   drh = lObj.trackDLBackEnd.dockerremotehost;
    %   if isempty(drh),
    %     defans = 'Local';
    %   else
    %     defans = 'Remote';
    %   end
    % 
    %   res = questdlg('Run docker on your Local machine, or SSH to a Remote machine?',...
    %     'Set Docker Remote Host','Local','Remote','Cancel',defans);
    %   if strcmpi(res,'Cancel'),
    %     return;
    %   end
    %   if strcmpi(res,'Remote'),
    %     res = inputdlg({'Remote Host Name:'},'Set Docker Remote Host',1,{drh});
    %     if isempty(res) || isempty(res{1}),
    %       return;
    %     end
    %     lObj.trackDLBackEnd.dockerremotehost = res{1};
    %   else
    %     lObj.trackDLBackEnd.dockerremotehost = '';
    %   end
    % 
    %   ischange = ~strcmp(drh,lObj.trackDLBackEnd.dockerremotehost);
    % 
    %   if ischange,
    %     res = questdlg('Test new Docker configuration now?','Test Docker configuration','Yes','No','Yes');
    %     if strcmpi(res,'Yes'),
    %       try
    %         tfsucc = lObj.trackDLBackEnd.testDockerConfig();
    %       catch ME,
    %         tfsucc = false;
    %         disp(getReport(ME));
    %       end
    %       if ~tfsucc,
    %         res = questdlg('Test failed. Revert to previous Docker settings?','Backend test failed','Yes','No','Yes');
    %         if strcmpi(res,'Yes'),
    %           lObj.trackDLBackEnd.dockerremotehost = drh;
    %         end
    %       end
    %     end
    %   end
    % end  % function
    % 
    % function cbkTrackerBackendSetDockerImageSpec(obj)
    %   lObj = obj.labeler_ ;      
    %   original_full_image_spec = lObj.get_backend_property('dockerimgfull') ;
    %   dialog_result = inputdlg({'Docker Image Spec:'},'Set image spec...',1,{original_full_image_spec},'on');
    %   if isempty(dialog_result)
    %     return
    %   end
    %   new_full_image_spec = dialog_result{1};
    %   try
    %     lObj.set_backend_property('dockerimgfull', new_full_image_spec) ;
    %   catch exception
    %     if strcmp(exception.identifier, 'APT:invalidValue') ,
    %       uiwait(errordlg(exception.message));
    %     else
    %       rethrow(exception);
    %     end
    %   end
    % end  % function

    function cbkTrackerBackendSetCondaEnv(obj)
      lObj = obj.labeler_ ;      
      original_value = lObj.get_backend_property('condaEnv') ;
      dialog_result = inputdlg({'Conda environment:'},'Set environment...',[1 50],{original_value});
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
      if ~isempty(lObj.tracker) ,
        obj.text_trackerinfo.String = lObj.tracker.getTrackerInfoString() ;
      end
    end  % function
    
    function cbkParameterChange(obj)
      lObj = obj.labeler_ ;      
      if isempty(lObj.tracker) ,
        return
      end
      obj.text_trackerinfo.String = lObj.tracker.getTrackerInfoString() ;
    end  % function
    
    function initTblFramesTrx_(obj)
      % Initialize the uitable of labeled frames in the 'Labeled Frames' window.

      labeler = obj.labeler_ ;

      isMA = labeler.maIsMA ;
      hasTrx = labeler.projectHasTrx;
      obj.tblFrames.Units = 'normalized';
      obj.tblFrames.RowName = '';
      showtargets = isMA || hasTrx;
      if isMA
        COLNAMES = {'Frame' 'N Tgts' 'N Pts' 'N ROIs'};
        %COLWIDTH = {'2x','1x','1x','1x'};
      elseif hasTrx,
        COLNAMES = {'Frame' 'N Tgts' 'N Pts'};
        %COLWIDTH = {'2x','1x','1x'};
      else
        COLNAMES = {'Frame' 'N Pts'};
        %COLWIDTH = {'2x','1x'};
      end
      set(obj.tblFrames,...
        'ColumnName',COLNAMES,...
        'Data',cell(obj.minTblFramesRows,numel(COLNAMES)),...
        'BackgroundColor',[.3 .3 .3; .45 .45 .45]);

      obj.uipanel_targets.Visible = onIff(showtargets);
      obj.uipanel_targetzoom.Visible = onIff(showtargets);
      postargets = obj.uipanel_targets.Position;
      posframes = obj.uipanel_frames.Position;

      if showtargets,
        posframes(1) = postargets(1)+postargets(3);
        obj.uipanel_frames.Position = posframes;        
      else
        posframes(1) = postargets(1);
        obj.uipanel_frames.Position = posframes;        
      end
      obj.resizeTblFramesTrx_();
    end  % function


    function resizeTblFramesTrx_(obj)

      minwidth = 5;
      for tbl0 = [obj.tblFrames, obj.tblTrx],
        colnames = tbl0.ColumnName;
        ncols = numel(colnames);
        u = tbl0.Units;
        tbl0.Units = 'pixel';
        tw = tbl0.InnerPosition(3);
        tbl0.Units = u;
        COLWIDTH = num2cell(repmat(max(minwidth,(tw-20)/ncols),[1,ncols]));
        tbl0.ColumnWidth = COLWIDTH;
      end
    end

    function data = getTblFramesData(obj)
      data = obj.tblFrames.Data;
      tfpad = cellfun(@isempty,data(:,1));
      data = data(~tfpad,:);
    end

    function setTblFramesData(obj,data)
      if size(data,1) < obj.minTblFramesRows,
        data = [data;cell(obj.minTblFramesRows-size(data,1),size(data,2))];
      end
      obj.tblFrames.Data = data;
    end
    
    function data = getTblTrxData(obj)
      data = obj.tblTrx.Data;
      tfpad = cellfun(@isempty,data(:,1));
      data = data(~tfpad,:);
    end

    function setTblTrxData(obj,data)
      if size(data,1) < obj.minTblTrxRows,
        if ~iscell(data),
          data = num2cell(data);
        end
        data = [data;cell(obj.minTblTrxRows-size(data,1),size(data,2))];
      end
      obj.tblTrx.Data = data;
    end

    function tfAxLimsSpecifiedInCfg = hlpSetConfigOnViews_(obj, viewCfg, centerOnTarget)
      % Configure the figures and axes showing the different views of the animal(s)
      % according to the specification in viewCfg.

      %labeler = obj.labeler_ ;
      axes_all = obj.axes_all;
      tfAxLimsSpecifiedInCfg = ...
        ViewConfig.setCfgOnViews(viewCfg, ...
                                 obj.figs_all, ...
                                 axes_all, ...
                                 obj.images_all, ...
                                 obj.axes_prev) ;
      if ~centerOnTarget
        [axes_all.CameraUpVectorMode] = deal('auto');
        [axes_all.CameraViewAngleMode] = deal('auto');
        [axes_all.CameraTargetMode] = deal('auto');
        [axes_all.CameraPositionMode] = deal('auto');
      end
      [axes_all.DataAspectRatio] = deal([1 1 1]);
      obj.menu_view_show_tick_labels.Checked = onIff(~isempty(axes_all(1).XTickLabel));
      obj.menu_view_show_grid.Checked = axes_all(1).XGrid;
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
    
    function menu_track_delete_current_tracker_actuated_(obj, source, event)  %#ok<INUSD>

      labeler = obj.labeler_;
      if labeler.tracker.bgTrnIsRunning
        uiwait(warndlg('Cannot delete current tracker while training is in progress. Cancel training first.','Training in progress'));
        return;
      end
      if labeler.tracker.bgTrkIsRunning
        uiwait(warndlg('Cannot delete current tracker while tracking is in progress. Cancel tracking first.','Tracking in progress'));
        return;
      end
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
      if (labeler.hasTrx || labeler.maIsMA) && ~labeler.isinit ,
        iTgt = labeler.currTarget;
        labeler.currImHud.updateTarget(iTgt);
        obj.labelTLInfo.updateLabels();
        obj.updateHighlightingOfAxes();
      end
    end  % function

    function menuSetupLabelModeHelp_(obj, labelMode)
      % Set .Checked for menu_setup_<variousLabelModes> based on labelMode
      menus = fieldnames(obj.setupMenu2LabelMode);
      for m = menus(:)',m=m{1}; %#ok<FXSET>
        obj.(m).Checked = 'off';
      end
      if isempty(labelMode) ,
        return
      end
      hMenu = obj.labelMode2SetupMenu.(char(labelMode));
      hMenu.Checked = 'on';
    end

    function cbkLabelModeChanged(obj)
      labeler = obj.labeler_ ;
      lblMode = labeler.labelMode;
      if isempty(lblMode) ,
        return
      end
      obj.menuSetupLabelModeHelp_(lblMode) ;
      switch lblMode
        case LabelMode.SEQUENTIAL
          obj.menu_setup_set_labeling_point.Visible = 'off';
          obj.menu_setup_set_nframe_skip.Visible = 'off';
          obj.menu_setup_streamlined.Visible = 'off';
          obj.menu_setup_load_calibration_file.Visible = 'off';
          obj.menu_setup_use_calibration.Visible = 'off';
          obj.menu_setup_ma_twoclick_align.Visible = 'off';
          obj.menu_view_zoom_toggle.Visible = 'off';
          obj.menu_view_pan_toggle.Visible = 'off';
          obj.menu_view_showhide_labelrois.Visible = 'off';
        case LabelMode.SEQUENTIALADD
          obj.menu_setup_set_labeling_point.Visible = 'off';
          obj.menu_setup_set_nframe_skip.Visible = 'off';
          obj.menu_setup_streamlined.Visible = 'off';
          obj.menu_setup_load_calibration_file.Visible = 'off';
          obj.menu_setup_use_calibration.Visible = 'off';
          obj.menu_setup_ma_twoclick_align.Visible = 'off';
          obj.menu_view_zoom_toggle.Visible = 'off';
          obj.menu_view_pan_toggle.Visible = 'off';
          obj.menu_view_showhide_labelrois.Visible = 'off';
        case LabelMode.MULTIANIMAL
          obj.menu_setup_set_labeling_point.Visible = 'off';
          obj.menu_setup_set_nframe_skip.Visible = 'off';
          obj.menu_setup_streamlined.Visible = 'off';
          obj.menu_setup_load_calibration_file.Visible = 'off';
          obj.menu_setup_use_calibration.Visible = 'off';
          obj.menu_setup_ma_twoclick_align.Visible = 'on';
          obj.menu_setup_ma_twoclick_align.Checked = labeler.isTwoClickAlign;
          obj.menu_view_zoom_toggle.Visible = 'on';
          obj.menu_view_pan_toggle.Visible = 'on';
          obj.menu_view_showhide_labelrois.Visible = 'on';
        case LabelMode.TEMPLATE
          %     obj.menu_setup_createtemplate.Visible = 'on';
          obj.menu_setup_set_labeling_point.Visible = 'off';
          obj.menu_setup_set_nframe_skip.Visible = 'off';
          obj.menu_setup_streamlined.Visible = 'off';
          obj.menu_setup_load_calibration_file.Visible = 'off';
          obj.menu_setup_use_calibration.Visible = 'off';
          obj.menu_setup_ma_twoclick_align.Visible = 'off';
          obj.menu_view_zoom_toggle.Visible = 'off';
          obj.menu_view_pan_toggle.Visible = 'off';
          obj.menu_view_showhide_labelrois.Visible = 'off';
        case LabelMode.HIGHTHROUGHPUT
          %     obj.menu_setup_createtemplate.Visible = 'off';
          obj.menu_setup_set_labeling_point.Visible = 'on';
          obj.menu_setup_set_nframe_skip.Visible = 'on';
          obj.menu_setup_streamlined.Visible = 'off';
          obj.menu_setup_load_calibration_file.Visible = 'off';
          obj.menu_setup_use_calibration.Visible = 'off';
          obj.menu_setup_ma_twoclick_align.Visible = 'off';
          obj.menu_view_zoom_toggle.Visible = 'off';
          obj.menu_view_pan_toggle.Visible = 'off';
          obj.menu_view_showhide_labelrois.Visible = 'off';
        case LabelMode.MULTIVIEWCALIBRATED2
          obj.menu_setup_set_labeling_point.Visible = 'off';
          obj.menu_setup_set_nframe_skip.Visible = 'off';
          obj.menu_setup_streamlined.Visible = 'on';
          obj.menu_setup_load_calibration_file.Visible = 'on';
          obj.menu_setup_use_calibration.Visible = 'on';
          obj.menu_setup_ma_twoclick_align.Visible = 'off';
          obj.menu_view_zoom_toggle.Visible = 'off';
          obj.menu_view_pan_toggle.Visible = 'off';
          obj.menu_view_showhide_labelrois.Visible = 'off';
      end
    end  % function

    function cbkLabels2HideChanged(obj, varargin)  
      % labeler = obj.labeler_ ;
      obj.updateShowImportedPredMenus(varargin{:});
    end  % function

    function cbkLabels2ShowCurrTargetOnlyChanged(obj, varargin)  
      % labeler = obj.labeler_ ;       
      obj.updateShowImportedPredMenus(varargin{:});
    end  % function

    function updateTrxMenuCheckEnable(obj,src,evt) %#ok<INUSD>
      if nargin < 2,
        src = nan;
      end
      labeler = obj.labeler_;
      if src ~= obj.menu_view_showhide_trajectories,
        set(obj.menu_view_showhide_trajectories, ...
          'Enable', onIff(labeler.hasProject && ~labeler.maIsMA && labeler.hasTrx));
      end
      if src ~= obj.menu_view_trajectories_showall,
        set(obj.menu_view_trajectories_showall, ...
          'Checked', onIff(labeler.hasProject && ~labeler.maIsMA && labeler.hasTrx && labeler.showTrx && ~labeler.showTrxCurrTargetOnly) ) ;
      end
      if src ~= obj.menu_view_trajectories_showcurrent,
        set(obj.menu_view_trajectories_showcurrent, ...
          'Checked', onIff(labeler.hasProject && ~labeler.maIsMA && labeler.hasTrx && labeler.showTrxCurrTargetOnly) ) ;
      end
      if src ~= obj.menu_view_trajectories_dontshow,
        set(obj.menu_view_trajectories_dontshow, ...
          'Checked', onIff(labeler.hasProject && ~labeler.maIsMA && labeler.hasTrx && ~labeler.showTrx && ~labeler.showTrxCurrTargetOnly) ) ;
      end
    end

    function cbkShowTrxChanged(obj, varargin)
      % labeler = obj.labeler_ ;
      obj.updateTrxMenuCheckEnable(varargin{:});
    end  % function

    function cbkShowOccludedBoxChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      onOff = onIff(labeler.showOccludedBox);
      obj.menu_view_occluded_points_box.Checked = onOff;
      set([obj.text_occludedpoints,obj.axes_occ],'Visible',onOff);
    end  % function

    function cbkShowTrxCurrTargetOnlyChanged(obj, varargin)
      % labeler = obj.labeler_ ;
      obj.updateTrxMenuCheckEnable(varargin{:});
    end  % function

    function cbkTrackModeIdxChanged(obj, src, evt)  %#ok<INUSD>
      obj.updatePUMTrackAndFriend() ;
    end  % function

    function cbkTrackerNFramesChanged(obj, src, evt)  %#ok<INUSD>
      obj.updatePUMTrackAndFriend() ;
    end  % function

    function updatePUMTrackAndFriend(obj)
      labeler = obj.labeler_ ;       
      if labeler.isinit || ~labeler.hasProject ,
        return
      end
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
      pumTrack = obj.pumTrack;
      rawMenuIndex = labeler.trackModeIdx ;
      if 1<=rawMenuIndex && rawMenuIndex<=numel(menustrs) ,
        menuIndex = rawMenuIndex ;
      else
        % This seems to happen sometimes when loading very old projects
        menuIndex = 1 ;
      end
      set(pumTrack, 'String', menustrs_compact, 'Value', menuIndex) ;  
        % Set these at same time to avoid possibility of Value out of range for String
      set(obj.text_framestotrackinfo,'String',menustrs{menuIndex});
    end  % function

    function cbkMovieCenterOnTargetChanged(obj, src, evt)   %#ok<INUSD>
      labeler = obj.labeler_ ;       
      tf = labeler.movieCenterOnTarget;
      mnu = obj.menu_view_trajectories_centervideoontarget;
      mnu.Checked = onIff(tf);
      if tf,
        obj.videoZoom(labeler.targetZoomRadiusDefault);
      end
    end  % function

    function cbkMovieRotateTargetUpChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      tf = labeler.movieRotateTargetUp;
      if tf
        ax = obj.axes_curr;
        warnst = warning('off','LabelerGUI:axDir');
        % When axis is in image mode, ydir should be reversed!
        ax.XDir = 'normal';
        ax.YDir = 'reverse';
        warning(warnst);
      end
      mnu = obj.menu_view_rotate_video_target_up;
      mnu.Checked = onIff(tf);
      labeler.UpdatePrevAxesDirections();
    end  % function
    
    function cbkMovieForceGrayscaleChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      tf = labeler.movieForceGrayscale;
      mnu = obj.menu_view_converttograyscale;
      mnu.Checked = onIff(tf);
    end  % function

    function cbkMovieViewBGsubbedChanged(obj, src, evt)  %#ok<INUSD>
    end  % function

    function didSetGTMode(obj)       
      % Updates the controls that depend upon whether the Labeler is in GT mode or
      % not, and then also brings the movie manager window to the fore if we just
      % switched to GT mode.
      obj.updateGTModeRelatedControls() ;
      % mmc = obj.movieManagerController_ ;
      % if ~isempty(mmc) ,
      %   labeler = obj.labeler_ ;     
      %   gt = labeler.gtIsGTMode ;
      %   if gt
      %     mmc.bringWindowToFront() ;
      %   end
      % end      
    end

    function updateGTModeRelatedControls(obj)
      % Updates the controls that depend upon whether the Labeler is in GT mode or
      % not.
      labeler = obj.labeler_ ;       
      gt = labeler.gtIsGTMode;
      onIffGT = onIff(gt);
      obj.menu_evaluate_gt_frames.Visible = onIffGT;
      obj.update_menu_evaluate() ;
      obj.txGTMode.Visible = onIffGT;
      % if ~isempty(obj.GTManagerFigure)
      %   obj.GTManagerFigure.Visible = onIffGT;
      % end
      obj.updateHighlightingOfAxes();
      obj.labelTLInfo.updateGTModeRelatedControls() ;
      % mmc = obj.movieManagerController_ ;
      % if ~isempty(mmc) ,
      %   mmc.lblerLstnCbkGTMode() ;
      % end
    end

    function update_menu_evaluate(obj)
      labeler = obj.labeler_ ;       
      gt = labeler.gtIsGTMode ;
      onIffGT = onIff(gt) ;
      obj.menu_evaluate_gtmode.Checked = onIffGT;
      obj.menu_evaluate_gtloadsuggestions.Visible = onIffGT;
      obj.menu_evaluate_gtsavesuggestions.Visible = onIffGT;
      obj.menu_evaluate_gtsetsuggestions.Visible = onIffGT;
      obj.menu_evaluate_gtcomputeperf.Visible = onIffGT;
      obj.menu_evaluate_gtcomputeperfimported.Visible = onIffGT;
      obj.menu_evaluate_gtexportresults.Visible = onIffGT;      
    end

    function cbkCropIsCropModeChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      obj.cropReactNewCropMode_();
      if labeler.hasProject && labeler.hasMovie
        labeler.setFrameGUI(labeler.currFrame,'tfforcereadmovie',true);
      end
    end  % function

    function cbkUpdateCropGUITools(obj, src, evt)  %#ok<INUSD>
      obj.cropReactNewCropMode_() ;
    end  % function
    
    function cbkCropCropsChanged(obj, src, evt)  %#ok<INUSD>
      obj.cropUpdateCropHRects_();
    end  % function

    function cbkNewMovie(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       

      %movRdrs = labeler.movieReader;
      %ims = arrayfun(@(x)x.readframe(1),movRdrs,'uni',0);
      hAxs = obj.axes_all;
      hIms = obj.images_all; % Labeler has already loaded with first frame
      assert(isequal(labeler.nview,numel(hAxs),numel(hIms)));

      tfResetAxLims = evt.isFirstMovieOfProject || labeler.movieRotateTargetUp;
      tfResetAxLims = repmat(tfResetAxLims,labeler.nview,1);
      % if isfield(handles,'newProjAxLimsSetInConfig')
      %   % AL20170520 Legacy projects did not save their axis lims in the .lbl
      %   % file.
      %   tfResetAxLims = tfResetAxLims | ~obj.newProjAxLimsSetInConfig;
      %   handles = rmfield(handles,'newProjAxLimsSetInConfig');
      % end

      % if labeler.hasMovie && evt.isFirstMovieOfProject,
      obj.updateEnablementOfManyControls() ;
      % end

      if ~labeler.gtIsGTMode,
        set(obj.menu_go_targets_summary,'Enable','on');
      else
        set(obj.menu_go_targets_summary,'Enable','off');
      end

      wbmf = @(src,evt)(obj.cbkWBMF(src,evt));
      wbuf = @(src,evt)(obj.cbkWBUF(src,evt));
      movnr = labeler.movienr;
      movnc = labeler.movienc;
      figs = obj.figs_all;
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

      obj.labelTLInfo.updateForNewMovie(obj.tbTLSelectMode.BackgroundColor);
      obj.labelTLInfo.updateLabels();

      nframes = labeler.nframes;
      sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
      set(obj.slider_frame,'Value',0,'SliderStep',sliderstep);

      tfHasMovie = labeler.currMovie>0;
      if tfHasMovie
        minzoomrad = 10;
        maxzoomrad = (labeler.movienc(1)+labeler.movienr(1))/4;
        obj.sldZoom.UserData = log([minzoomrad maxzoomrad]);
      end

      TRX_MENUS = {...
        'menu_view_trajectories_centervideoontarget'
        'menu_view_rotate_video_target_up'};
      %  'menu_setup_label_overlay_montage_trx_centered'};
      tftblon = labeler.hasTrx || labeler.maIsMA;
      onOff = onIff(tftblon);
      cellfun(@(x)set(obj.(x),'Enable',onOff),TRX_MENUS);
      hTbl = obj.tblTrx;
      set(hTbl,'Enable',onOff);

      obj.updatePUMTrackAndFriend() ;

      % See note in AxesHighlightManager: Trx vs noTrx, Axes vs Panels
      obj.axesesHighlightManager_.setHighlightPanel(labeler.hasTrx) ;

      obj.updateHighlightingOfAxes();

      if labeler.cropIsCropMode
        obj.cropUpdateCropHRects_() ;
      end
      obj.menu_file_crop_mode.Enable = onIff(~labeler.hasTrx);

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
      set(obj.txMoviename,'String',str);

      % by default, use calibration if there is calibration for this movie
      lc = labeler.lblCore;
      if ~isempty(lc) && lc.supportsCalibration,
        obj.menu_setup_use_calibration.Checked = onIff(lc.isCalRig && lc.showCalibration);
      end
    end  % function

    function cbkDataImported(obj, src, evt)  %#ok<INUSD>
      obj.labelTLInfo.updateLabels();  % Using this as a "refresh" for now
    end  % function

    function cbkShowSkeletonChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      hasSkeleton = ~isempty(labeler.skeletonEdges) ;
      isChecked = onIff(hasSkeleton && labeler.showSkeleton) ;
      set(obj.menu_view_showhide_skeleton, 'Enable', hasSkeleton, 'Checked', isChecked) ;
    end  % function

    function cbkShowMaRoiChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      onOff = onIff(labeler.showMaRoi);
      obj.menu_view_showhide_maroi.Checked = onOff;
    end  % function

    function cbkShowMaRoiAuxChanged(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;       
      onOff = onIff(labeler.showMaRoiAux);
      obj.menu_view_showhide_maroiaux.Checked = onOff;
    end  % function
    
    function initializeResizeInfo_(obj)

      % Record the width of txUnsavedChanges, so we can keep it fixed
      hTx = obj.txUnsavedChanges;
      hPnlPrev = obj.uipanel_prev;
      
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

      % Take steps to keep right edge of unsaved changes text box aligned with right
      % edge of the previous/reference frame panel
      pxTxUnsavedChangesWidth = obj.pxTxUnsavedChangesWidth_ ;
      hTx = obj.txUnsavedChanges;
      hPnlPrev = obj.uipanel_prev;
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1) + hPnlPrev.Position(3) ;
      hTx.Position(1) = uiPnlPrevRightEdge - pxTxUnsavedChangesWidth ;
      hTx.Position(3) = pxTxUnsavedChangesWidth ;
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      obj.resizeTblFramesTrx_();

      %obj.updateStatus() ;  % do we need this here?
    end
    
    function cropReactNewCropMode_(obj)
      labeler = obj.labeler_ ;
      isInCropMode = labeler.cropIsCropMode ;

      if isempty(isInCropMode) ,
        return
      end

      REGCONTROLS = {
        'pbClear'
        'tbAccept'
        'pbTrain'
        'pbTrack'
        'pumTrack'};

      onIfIsInCropMode = onIff(isInCropMode);
      offIfIsInCropMode = onIff(~isInCropMode);

      %cellfun(@(x)set(obj.(x),'Visible',onIfTrue),CROPCONTROLS);
      set(obj.uipanel_cropcontrols,'Visible',onIfIsInCropMode);
      set(obj.text_trackerinfo,'Visible',offIfIsInCropMode);

      cellfun(@(x)set(obj.(x),'Visible',offIfIsInCropMode),REGCONTROLS);
      obj.menu_file_crop_mode.Checked = onIfIsInCropMode;

      obj.cropUpdateCropHRects_() ;
      obj.cropUpdateCropAdjustingCropSize_(false) ;
    end
    
    function cropUpdateCropHRects_(obj)
      % Update obj.cropHRect from lObj.cropIsCropMode, lObj.currMovie and
      % lObj.movieFilesAll*cropInfo
      %
      % rect props set:
      % - position
      % - visibility, pickableparts
      %
      % rect props NOT set:
      % - resizeability.

      lObj = obj.labeler_ ;

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
          h = obj.cropHRect(ivw);
          cropImRectSetPosnNoPosnBang(h,CropInfo.roi2RectPos(roi(ivw,:)));
          set(h,'Visible','on','PickableParts','all');
          fcn = makeConstrainToRectFcn('imrect',[1 imnc(ivw)],[1 imnr(ivw)]);
          h.setPositionConstraintFcn(fcn);
        end
      else
        arrayfun(@(rect)cropImRectSetPosnNoPosnBang(rect,[nan nan nan nan]),...
                 obj.cropHRect);
        arrayfun(@(x)set(x,'Visible','off','PickableParts','none'),obj.cropHRect);
      end
    end

    function cropUpdateCropAdjustingCropSize_(obj, tfAdjust)
      tb = obj.tbAdjustCropSize;
      if nargin<2
        tfAdjust = tb.Value==tb.Max; % tb depressed
      end

      if tfAdjust
        tb.Value = tb.Max;
        tb.String = 'Done Adjusting' ;
        tb.BackgroundColor = [ 1 0 0 ] ;
      else
        tb.Value = tb.Min;
        tb.String = 'Adjust Size' ;
        tb.BackgroundColor = [ 0 0.450980392156863 0.741176470588235 ] ;
      end
      arrayfun(@(x)x.setResizable(tfAdjust),obj.cropHRect);
    end
    
    function cbkWBMF(obj, src, evt)
      labeler = obj.labeler_ ;      
      lcore = labeler.lblCore;
      if ~isempty(lcore)
        lcore.wbmf(src,evt) ;
      end
    end
    
    function cbkWBUF(obj, src, evt)
      labeler = obj.labeler_ ;      
      if ~isempty(labeler.lblCore)
        labeler.lblCore.wbuf(src,evt) ;
      end
    end
    
    function scroll_callback(obj, hObject, eventdata)
      %labeler = obj.labeler_ ;
      
      ivw = find(hObject==obj.figs_all);
      ax = obj.axes_all(ivw);
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

      % how big is the axes in pixels?
      units = get(ax,'Units');
      set(ax,'Units','pixels');
      axpos_px = get(ax,'Position');
      set(ax,'Units',units);
      szpx = axpos_px(3:4);
      
      dx = xlim(2)-xlim(1);
      dy = ylim(2)-ylim(1);
      xscale = szpx(1)/dx;
      yscale = szpx(2)/dy;
      scale = min(xscale,yscale); % px per data unit      
      him = obj.images_all(ivw);
      imglimx = get(him,'XData');
      imglimy = get(him,'YData');

      % how big would the xlim, ylim be if we went beyond the edges of the
      % image
      vdx = szpx(1)/scale;
      vdy = szpx(2)/scale;
      x0 = (xlim(1)+xlim(2))/2;
      y0 = (ylim(1)+ylim(2))/2;
      vxlim = [x0-vdx/2,x0+vdx/2];
      vylim = [y0-vdy/2,y0+vdy/2];

      xlim(1) = curp(1,1)-(curp(1,1)-vxlim(1))/scrl;
      xlim(2) = curp(1,1)+(vxlim(2)-curp(1,1))/scrl;
      ylim(1) = curp(1,2)-(curp(1,2)-vylim(1))/scrl;
      ylim(2) = curp(1,2)+(vylim(2)-curp(1,2))/scrl;
      xlim(1) = max(xlim(1),imglimx(1));
      xlim(2) = min(xlim(2),imglimx(2));
      ylim(1) = max(ylim(1),imglimy(1));
      ylim(2) = min(ylim(2),imglimy(2));
      set(ax,'XLim',xlim,'YLim',ylim);
      %set(ax,'DataAspectRatioMode','auto');
      %axis(ax,[xlim(1),xlim(2),ylim(1),ylim(2)]);
      %set(ax,'DataAspectRatio',[1,1,1]);
      % fprintf('Scrolling %d!!\n',eventdata.VerticalScrollAmount)
    end

    function closeImContrast(obj, iAxRead, iAxApply)
      % ReadClim from axRead and apply to axApply

      labeler = obj.labeler_ ;
      
      axAll = obj.axes_all;
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
      		set(obj.axes_prev,'CLim',clim);
      	end
      end
    end

    function [tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(obj)

      labeler = obj.labeler_ ;
      
      if ~labeler.isMultiView
      	tfproceed = 1;
      	iAxRead = 1;
      	iAxApply = 1;
      else
        fignames = {obj.figs_all.Name}';
        fignames{1} = obj.txMoviename.String;
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
              iAxApply = 1:numel(obj.axes_all);
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
    
    function hlpRemoveFocus_(obj)
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
      obj.mainFigure_.CurrentObject = obj.axes_curr ;
      %uicontrol(obj.txStatus);
    end

    function tblFrames_cell_selected_(obj, src, evt)
      labeler = obj.labeler_ ;
      row = evt.Indices;
      if ~isempty(row)
        row = row(1);
        dat = get(src,'Data');
        if ~isempty(dat{row,1}),
          labeler.setFrameGUI(dat{row,1},'changeTgtsIfNec',true);
        end
      end
      obj.hlpRemoveFocus_() ;
    end

    function axescurrXLimChanged(obj, hObject, eventdata)  %#ok<INUSD>
      %labeler = obj.labeler_ ;
      ax = eventdata.AffectedObject;
      radius = diff(ax.XLim)/2;
      hSld = obj.sldZoom;
      if ~isempty(hSld.UserData) % empty during init
        userdata = hSld.UserData;
        logzoomradmin = userdata(1);
        logzoomradmax = userdata(2);
        sldval = (log(radius)-logzoomradmax)/(logzoomradmin-logzoomradmax);
        sldval = min(max(sldval,0),1);
        hSld.Value = sldval;
      end
    end

    function videoRotateTargetUpAxisDirCheckWarn_(obj)
      ax = obj.axes_curr;
      if (strcmp(ax.XDir,'reverse') || strcmp(ax.YDir,'normal')) && obj.labeler_.movieRotateTargetUp
        warningNoTrace(...
          'LabelerGUI:axDir', ...
          'Main axis ''XDir'' or ''YDir'' is set to be flipped and .movieRotateTargetUp is set. Graphics behavior may be unexpected; proceed at your own risk.');
      end
    end

    function axescurrXDirChanged(obj, hObject, eventdata)  %#ok<INUSD>
      obj.videoRotateTargetUpAxisDirCheckWarn_() ;
    end

    function axescurrYDirChanged(obj, hObject, eventdata)  %#ok<INUSD>
      obj.videoRotateTargetUpAxisDirCheckWarn_() ;
    end
    
    function cbkPostZoom(obj,src,evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if evt.Axes == obj.axes_prev,
        labeler.UpdatePrevAxesLimits();
      end
    end

    function cbkPostPan(obj,src,evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if evt.Axes == obj.axes_prev,
        labeler.UpdatePrevAxesLimits();
      end
    end

    % function cbklabelTLInfoSelectOn(obj, src, evt)  %#ok<INUSD>
    %   obj.labelTLInfo.didSetTimelineSelectMode();      
    %   labeler = obj.labeler_ ;
    %   itm = labeler.infoTimelineModel ;
    %   tb = obj.tbTLSelectMode;  % the togglebutton
    %   tb.Value = itm.selectOn;
    % end

    function updateTimeline(obj)
      % Update the props dropdown menu and timeline.
      labeler = obj.labeler_ ;
      hasProject = labeler.hasProject ;
      hasMovie = labeler.hasMovie ;        
      itm = labeler.infoTimelineModel ;
      props = itm.getPropsDisp(itm.curproptype);
      proptypes = itm.getPropTypesDisp();
      if ~isempty(obj.labelTLInfo)
        obj.labelTLInfo.update();
      end
      tb = obj.tbTLSelectMode;  % the togglebutton
      tb.Value = itm.selectOn;      
      set(obj.pbClearSelection,'Enable',onIff(hasProject && hasMovie && labeler.areAnyFramesSelected())) ;
      set(obj.pumInfo,'String',props,'Value',itm.curprop,'Enable',onIff(hasProject));
      set(obj.pumInfo_labels,'String',proptypes,'Value',itm.curproptype,'Enable',onIff(hasProject));
      set(obj.tbTLSelectMode,'Enable',onIff(hasProject));
      set(obj.pumTrack,'Enable',onIff(hasProject));
    end

    function updateTimelineStatThresh(obj)
      % Update the timeline statistic threshold display.
      if ~isempty(obj.labelTLInfo)
        obj.labelTLInfo.updateStatThresh();
      end
    end

    function updateTimelineLabels(obj)
      % Update the timeline labels display.
      if ~isempty(obj.labelTLInfo)
        obj.labelTLInfo.updateLabels();
      end
    end

    function updateTimelineLandmarkColors(obj)
      % Update the timeline landmark colors.
      if ~isempty(obj.labelTLInfo)
        obj.labelTLInfo.updateLandmarkColors();
      end
    end

    % function cbklabelTLInfoPropTypesUpdated(obj, src, evt)  %#ok<INUSD>
    %   % Update the props dropdown menu and timeline.
    %   labeler = obj.labeler_ ;
    %   itm = labeler.infoTimelineModel ;
    %   proptypes = itm.getPropTypesDisp();
    %   set(obj.pumInfo_labels,'String',proptypes);
    % end
    
    function menuSetupLabelModeCbkGeneric(obj, src, evt)  %#ok<INUSD>
      lblMode = obj.setupMenu2LabelMode.(src.Tag);
      obj.labeler_.labelingInit('labelMode',lblMode);
    end
    
    % function figure_CloseRequestFcn(obj, src, evt)  %#ok<INUSD>
    %   obj.quitRequested() ;
    % end

    function videoZoom(obj,zoomRadius)
      % Zoom to square window over current frame center with given radius.
      [x0,y0] = obj.videoCurrentCenter();
      lims = [x0-zoomRadius,x0+zoomRadius,y0-zoomRadius,y0+zoomRadius];
      axis(obj.axes_curr,lims);
    end    

    function [xsz,ysz] = videoCurrentSize(obj)
      %labeler = obj.labeler_ ;
      
      v = axis(obj.axes_curr);
      xsz = v(2)-v(1);
      ysz = v(4)-v(3);
    end

    function [x0,y0] = videoCurrentCenter(obj)
      x0 = mean(get(obj.axes_curr,'XLim'));
      y0 = mean(get(obj.axes_curr,'YLim'));
    end

    function v = videoCurrentAxis(obj)
      %labeler = obj.labeler_ ;
      
      v = axis(obj.axes_curr);
    end

    function videoSetAxis(obj,lims,resetcamera)
      if nargin<3
        resetcamera = true;
      end
      %labeler = obj.labeler_ ;
      
      % resets camera view too
      ax = obj.axes_curr;
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
      % labeler = obj.labeler_ ;
      
      [xsz,ysz] = obj.videoCurrentSize();
      lims = [x-xsz/2,x+xsz/2,y-ysz/2,y+ysz/2];
      axis(obj.axes_curr,lims);      
    end
    
    function xy = videoClipToVideo(obj,xy)
      % Clip coords to video size.
      %
      % xy (in): [nx2] xy-coords
      %
      % xy (out): [nx2] xy-coords, clipped so that x in [1,nc] and y in [1,nr]
      
      labeler = obj.labeler_ ;
      
      xy = CropInfo.roiClipXY(labeler.movieroi,xy);
    end

    function dxdy = videoCurrentUpVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "up" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 
      
      labeler = obj.labeler_ ;
      
      ax = obj.axes_curr;
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
      
      ax = obj.axes_curr;
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
            
      obj.videoPlaySegmentCore(labeler.currFrame,labeler.nframes,...
        'setFrameArgs',{'updateTables',false});
    end
    
    % function videoPlaySegment(obj)
    %   % Play segment centererd at .currFrame
    % 
    %   labeler = obj.labeler_ ;
    % 
    %   f = labeler.currFrame;
    %   df = labeler.moviePlaySegRadius;
    %   fstart = max(1,f-df);
    %   fend = min(labeler.nframes,f+df);
    %   obj.videoPlaySegmentCore(fstart,fend,'freset',f,...
    %     'setFrameArgs',{'updateTables',false,'updateLabels',false});
    % end

    function videoPlaySegFwdEnding(obj)
      % Play segment ending at .currFrame
      labeler = obj.labeler_ ;
            
      f = labeler.currFrame;
      df = labeler.moviePlaySegRadius;
      fstart = max(1,f-df);
      fend = f;
      obj.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end
    
    function videoPlaySegRevEnding(obj)
      % Play segment (reversed) ending at .currFrame
      labeler = obj.labeler_ ;
      
      f = labeler.currFrame;
      df = labeler.moviePlaySegRadius;
      fstart = min(f+df,labeler.nframes);
      fend = f;
      obj.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end


    function videoPlaySegmentCore(obj,fstart,fend,varargin)
      
      labeler = obj.labeler_ ;
      
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

        labeler.setFrameGUI(f,setFrameArgs{:});
        drawnow('limitrate');
      end
      
      if tfreset
        % AL20170619 passing setFrameArgs a bit fragile; needed for current
        % callers (don't update labels in videoPlaySegment)
        labeler.setFrameGUI(freset,setFrameArgs{:}); 
      end
      
      % - icon managed by caller      
    end

    function videoCenterOnCurrTargetPoint(obj)
      labeler = obj.labeler_ ;
      
      [tfsucc,xy] = labeler.videoCenterOnCurrTargetPointHelp();
      if tfsucc
        [x0,y0] = obj.videoCurrentCenter();
        dx = xy(1)-x0;
        dy = xy(2)-y0;
        ax = obj.axes_curr;
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
      ax = obj.axes_curr;
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
    
    function updateTargetCentrationAndZoom(obj)
      labeler = obj.labeler_ ;      
      if labeler.isinit || ~labeler.hasProject ,
        return
      end      
      if (labeler.hasTrx || labeler.maIsMA) && labeler.movieCenterOnTarget && ~labeler.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTarget();
        obj.videoZoom(labeler.targetZoomRadiusDefault);
      elseif labeler.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTargetPoint();
      end
    end  % function

    function play_(obj, playMethodName)
      if obj.isPlaying_ ,
        obj.isPlaying_ = false ;  
          % setting this this will cause the already-running video playback loop from the previous cal to play_() to exit
        return
      end
      oc = onCleanup(@()(obj.playCleanup_())) ;
      obj.isPlaying_ = true ;
      obj.pbPlay.CData = Icons.ims.stop ;
      obj.(playMethodName) ;
    end  % function

    function playCleanup_(obj)
      obj.pbPlay.CData = Icons.ims.play ;
      obj.isPlaying_ = false ;
    end  % function

    function tblTrx_cell_selected_(obj, src, evt) %#ok<*DEFNU>
      % Current/last row selection is maintained in hObject.UserData

      labeler = obj.labeler_ ;

      if ~(labeler.hasTrx || labeler.maIsMA)
        obj.hlpRemoveFocus_();
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

      obj.hlpRemoveFocus_();
    end

    %
    % This is where the insertion of the dispatchMainFigureCallback.m methods
    % starts.
    %



    function pumTrack_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.trackModeIdx = src.Value;
    end



    function slider_frame_actuated_(obj, src,evt,varargin)  %#ok<INUSD>



      labeler = obj.labeler_ ;

      % Hints: get(src,'Value') returns position of slider
      %        get(src,'Min') and get(src,'Max') to determine range of slider

      debugtiming = false;
      if debugtiming,
        starttime = tic() ;  %#ok<UNRCH>
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

      cmod = obj.mainFigure_.CurrentModifier;
      if ~isempty(cmod) && any(strcmp(cmod{1},{'control' 'shift'}))
        if f>labeler.currFrame
          tfSetOccurred = labeler.frameUpGUI(true);
        else
          tfSetOccurred = labeler.frameDownGUI(true);
        end
      else
        tfSetOccurred = labeler.setFrameProtectedGUI(f);
      end

      if ~tfSetOccurred
        sldval = (labeler.currFrame-1)/(labeler.nframes-1);
        if isnan(sldval)
          sldval = 0;
        end
        set(src,'Value',sldval);
      end

      if debugtiming,
        fprintf('Slider callback setting to frame %d took %f seconds\n',f,toc(starttime));  %#ok<UNRCH>
      end


    end



    function edit_frame_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;

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
        labeler.setFrameGUI(f)
      end



    end



    function pbClear_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if ~labeler.doProjectAndMovieExist()
        return;
      end
      labeler.lblCore.clearLabels();
      labeler.CheckPrevAxesTemplate();
    end



    function tbAccept_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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
      if ~labeler.doProjectAndMovieExist()
        return
      end
      v = src.Value;
      userdata = src.UserData;
      logzoomrad = userdata(2)+v*(userdata(1)-userdata(2));
      zoomRad = exp(logzoomrad);
      obj.videoZoom(zoomRad);
      obj.hlpRemoveFocus_() ;
    end



    function pbResetZoom_actuated_(obj, src, evt)  %#ok<INUSD>
      hAxs = obj.axes_all;
      hIms = obj.images_all;
      assert(numel(hAxs)==numel(hIms));
      arrayfun(@zoomOutFullView,hAxs,hIms,false(1,numel(hIms)));
    end



    function pbSetZoom_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.targetZoomRadiusDefault = diff(obj.axes_curr.XLim)/2;
    end



    function pbRecallZoom_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      % TODO this is broken!!
      obj.videoCenterOnCurrTarget();
      obj.videoZoom(labeler.targetZoomRadiusDefault);
    end



    function tbTLSelectMode_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.setTimelineSelectMode(src.Value) ;
    end



    function pbClearSelection_actuated_(obj, src, evt)  %#ok<INUSD>
      % Clear the current selection of frames as shown in the timeline axes.
      labeler = obj.labeler_ ;
      if ~labeler.doProjectAndMovieExist()
        return
      end
      labeler.clearSelectedFrames() ;
    end



    function menu_file_save_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.save();
    end



    function menu_file_saveas_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.saveAs();
    end



    function menu_file_load_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.load() ;
    end

    function load(obj)
      labeler = obj.labeler_ ;
      if obj.raiseUnsavedChangesDialogIfNeeded() ,
        % currMovInfo = labeler.projLoadGUI();
        labeler.projLoadGUI();
        % if ~isempty(currMovInfo)
        %   obj.movieManagerController_.setVisible(true);
        %   wstr = ...
        %     sprintf(strcatg('Could not find file for movie(set) %d: %s.\n\nProject opened with no movie selected. ', ...
        %                     'Double-click a row in the MovieManager or use the ''Switch to Movie'' button to start working on a movie.'), ...
        %             currMovInfo.iMov, ...
        %             currMovInfo.badfile);
        %   warndlg(wstr,'Movie not found','modal');
        % end
      end
    end

    function menu_go_movies_summary_actuated_(obj, src, evt)
      obj.menu_file_managemovies_actuated_(src, evt) ;
    end

    function ShowMovieManager(obj)

      labeler = obj.labeler_ ;

      labeler.pushBusyStatus('Opening Movie Manager...') ;  % Want to do this here, b/c the stuff in this method can take a while
      oc = onCleanup(@()(labeler.popBusyStatus()));
      drawnow;

      if ~isempty(obj.movieManagerController_) && obj.movieManagerController_.isValid() ,
        obj.movieManagerController_.setVisible(true);
      else
        obj.movieManagerController_ = MovieManagerController(obj.labeler_);
      end

    end

    function menu_file_managemovies_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.ShowMovieManager();

    end

    function menu_file_import_labels_trk_curr_mov_actuated_(obj, src, evt)  %#ok<INUSD>

      labeler = obj.labeler_ ;
      if ~labeler.hasMovie
        error('LabelerGUI:noMovie','No movie is loaded.');
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
      labeler.labelImportTrkPromptGenericSimple(iMov,'labelImportTrk','gtok',false) ;
    end

    function menu_file_import_labels2_trk_curr_mov_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      iMov = labeler.currMovie; % gt-aware
      labeler.labelImportTrkPromptGenericSimple(iMov,'labels2ImportTrk','gtok',true);
    end

    function menu_file_export_labels_trks_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      [tfok,rawtrkname] = obj.getExportTrkRawNameUI('labels',true);
      if ~tfok
        return;
      end
      labeler.labelExportTrkGUI(1:labeler.nmoviesGTaware,'rawtrkname',rawtrkname);
    end

    function menu_file_export_labels_table_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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

    function menu_file_export_cocojson_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      [tfCanExport,reason] = labeler.trackCanExport();
      if ~tfCanExport,
        uiwait(warndlg(reason,'Cannot export labels'));
        return;
      end
      fname = labeler.getDefaultFilenameExportCOCOJson();
      [f,p] = uiputfile(fname,'Export File');
      if isequal(f,0)
        return;
      end
      fname = fullfile(p,f);
      fprintf('Exporting COCO json file and labeled images for current tracker to %s...\n',fname);
      labeler.tracker.export_coco_db(fname);
      fprintf('Done.\n');
    end

    function menu_file_import_labels_cocojson_actuated_(obj, src, evt)  %#ok<INUSD>
      % callback for importing labels from coco json

      res = questdlg('WARNING! Importing labels will overwrite labels in your current project. Proceed?','Warning','Yes','No','Cancel','No');
      if ~strcmpi(res,'yes'),
        return;
      end

      labeler = obj.labeler_ ;
      fname = labeler.getDefaultFilenameImportCOCOJson();
      [f,p] = uigetfile(fname,'Import COCO Json File');
      if isequal(f,0)
        return;
      end
      cocojsonfile = fullfile(p,f);
      if ~exist(cocojsonfile,'file'),
        errordlg(sprintf('File %s does not exist',cocojsonfile),'Error importing COCO labels');
        return;
      end
      try
        cocos = TrnPack.hlpLoadJson(cocojsonfile);
      catch ME,
        warningNoTrace('Error loading json file %s:\n%s\n',cocojsonfile,getReport(ME));
        errordlg(sprintf('Error loading json file %s',cocojsonfile),'Error importing COCO labels');
        return;
      end
      if ~isfield(cocos,'images') || ~isfield(cocos,'annotations'),
        warningNoTrace('COCO json file must contain entries "images" and "annotations"');
        errordlg('COCO json file must contain entries "images" and "annotations"','Bad COCO json file','modal');
        return;
      end
      hasmovies = isfield(cocos,'info') && isfield(cocos.info,'movies');
      % we will create a fake movie directory, where should we put it?
      if ~hasmovies,
        if isempty(cocos.images) || isempty(cocos.annotations),
          warningNoTrace('No annotations/images to import');
          return;
        end
        % see if the images are named in a way parsable by
        % get_readframe_fcn
        [p1,filename,imext] = fileparts(cocos.images(1).file_name);
        outimdir = fullfile(p,p1);
        m = regexp(filename,'^(.*[^\d])(\d+)$','tokens','once');
        isseq = false;
        if ~isempty(m),
          imname = m{1};
          basename = fullfile(p1,imname);
          imfiles = {cocos.images.file_name};
          if numel(imfiles) == 1,
            isseq = true;
          else
            framenum = regexp(imfiles,[basename,'(\d+)\',imext,'$'],'tokens','once');
            if ~any(cellfun(@isempty,framenum)),
              framenum = cellfun(@str2double,framenum);
              sortedframenum = sort(framenum);
              if all(diff(sortedframenum)==1),
                isseq = true;
              end
            end
          end
        end
        if isseq,          
          args = {'outimdir',outimdir,'overwrite',false,'imname',imname,'cocojsonfile',cocojsonfile,'copyims',false};
        else
          outimdirparent = uigetdir(p,'Folder to output movie frames to');
          if ~ischar(outimdirparent),
            return;
          end
          outdirname = 'movie';
          imname = 'frame';
          % if the name of the directory is movie, then assume that we want
          % to use this directory to output images to
          [~,n] = fileparts(outimdirparent);
          if strcmp(n,outdirname),
            outimdir = outimdirparent;
          else
            % if this is an empty directory, also assume we want to output
            % here
            dircontents = mydir(outimdirparent);
            if isempty(dircontents),
              outimdir = outimdirparent;
            else
              % assume we should create a new directory named movie in this
              % directory
              outimdir = fullfile(outimdirparent,outdirname);
            end
          end
          overwrite = true;
          if exist(outimdir,'dir'),
            [~,~,imext] = fileparts(cocos.images(1).file_name);
            dircontents = mydir(fullfile(outimdir,[imname,'*',imext]));
            if ~isempty(dircontents),
              res = questdlg(sprintf('Images exist in %s, overwrite?',imname),'Overwrite?','Yes','No','Cancel','Yes');
              if strcmpi(res,'Cancel'),
                return;
              end
              overwrite = strcmpi(res,'Yes');
            end
          end
          args = {'outimdir',outimdir,'overwrite',overwrite,'imname',imname,'cocojsonfile',cocojsonfile};
        end
      else
        args = {};
      end
      fprintf('Importing labels from %s...\n',cocojsonfile);
      labeler.labelPosBulkImportCOCOJson(cocos,args{:});
      fprintf('Done.\n');
    end

    function menu_file_import_labels_table_actuated_(obj, src, evt)  %#ok<INUSD>

      res = questdlg('WARNING! Importing labels will overwrite labels in your current project. Proceed?','Warning!','Yes','No','Cancel','No');
      if ~strcmpi(res,'yes'),
        return;
      end

      labeler = obj.labeler_ ;
      lastFile = labeler.rcGetProp('lastLabelMatfile');
      if isempty(lastFile)
        lastFile = pwd;
      end
      [fname,pth] = uigetfile('*.mat','Load Labels',lastFile);
      if isequal(fname,0)
        return
      end
      fname = fullfile(pth,fname);
      t = loadSingleVariableMatfile(fname);
      labeler.labelPosBulkImportTbl(t);
      fprintf('Loaded %d labeled frames from file ''%s''.\n',height(t),fname);
    end

    function menu_file_export_stripped_lbl_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      fname = labeler.getDefaultFilenameExportStrippedLbl();
      [f,p] = uiputfile(fname,'Export File');
      if isequal(f,0)
        return
      end
      fname = fullfile(p,f);
      labeler.projExportTrainData(fname)
    end

    function menu_file_crop_mode_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if ~labeler.hasMovie ,
        error('Can''t do that without a movie') ;
      end
      if ~isempty(labeler.tracker) && ~labeler.gtIsGTMode && labeler.labelPosMovieHasLabels(labeler.currMovie),
        res = questdlg('Frames of the current movie are labeled. Editing the crop region for this movie will cause trackers to be reset. Continue?');
        if ~strcmpi(res,'Yes'),
          return;
        end
      end
      labeler.cropSetCropMode(~labeler.cropIsCropMode);
    end

    function menu_file_clean_tempdir_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if isempty(labeler.projTempDir),
        rootdir = APT.getdotaptdirpath() ;  % native path
      else
        rootdir = fileparts(labeler.projTempDir);
      end
      if ~exist(rootdir,'dir'),
        return
      end
      todelete = mydir(rootdir,'isdir',true);
      if ~isempty(labeler.projTempDir),
        i = find(strcmp(todelete,labeler.projTempDir));
        assert(~isempty(i));
        todelete(i) = [];
      end
      if isempty(todelete),
        uiwait(msgbox('No temp directories to remove.','All clear!'));
        return
      end
      res = questdlg(sprintf('Delete %d temp directories? Only do this if no other instances of APT are open.',numel(todelete)));
      if ~strcmpi(res,'Yes'),
        return
      end
      labeler.projRemoveOtherTempDirs(todelete) ;
    end    
    
    function menu_file_bundle_tempdir_actuated_(obj, src, evt)  %#ok<INUSD>
      [fname,pname,~] = uiputfile('*.tar','File to save the training bundle as...');
      if isnumeric(fname)
        return
      end
      tfile = fullfile(pname,fname);
      labeler = obj.labeler_ ;
      labeler.projBundleTempDir(tfile);
    end

    function menu_help_actuated_(obj, src, evt)  %#ok<INUSD>
    end

    function menu_help_labeling_actions_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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
      about(labeler);
    end



    function menu_setup_sequential_mode_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.menuSetupLabelModeCbkGeneric(src);
    end



    function menu_setup_sequential_add_mode_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.menuSetupLabelModeCbkGeneric(src);
    end



    function menu_setup_template_mode_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.menuSetupLabelModeCbkGeneric(src);
    end



    function menu_setup_highthroughput_mode_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.menuSetupLabelModeCbkGeneric(src);
    end



    function menu_setup_multiview_calibrated_mode_2_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.menuSetupLabelModeCbkGeneric(src);
    end



    function menu_setup_multianimal_mode_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.menuSetupLabelModeCbkGeneric(src);
    end



    function menu_setup_label_overlay_montage_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;            
      labeler.pushBusyStatus('Plotting all labels on one axes to visualize label distribution...');
      oc = onCleanup(@()(labeler.popBusyStatus())) ;
      drawnow;
      if labeler.hasTrx
        labeler.labelOverlayMontageGUI();
        labeler.labelOverlayMontageGUI('ctrMeth','trx');
        labeler.labelOverlayMontageGUI('ctrMeth','trx','rotAlignMeth','trxtheta');
        % could also use headtail for centering/alignment but skip for now.
      else % labeler.maIsMA, or SA-non-trx
        labeler.labelOverlayMontageGUI();
        if ~labeler.isMultiView
          labeler.labelOverlayMontageGUI('ctrMeth','centroid');
          if labeler.maIsMA
            prms = labeler.trackParams;
            if ~isempty(prms)
              if isfield(prms.ROOT.MultiAnimal.TargetCrop,'multi_scale_by_bbox')
                tfScale = prms.ROOT.MultiAnimal.TargetCrop.multi_scale_by_bbox;
              else
                tfScale = false;
              end
            end
          else
            tfScale = false;
          end
%           if tfScale
%             labeler.labelOverlayMontageGUI('ctrMeth','centroid','scale',true);          
%           end
          tfHTdefined = ~isempty(labeler.skelHead) && ~isempty(labeler.skelTail);
          if tfHTdefined
            labeler.labelOverlayMontageGUI('ctrMeth','centroid','rotAlignMeth','headtail');
            if tfScale
              labeler.labelOverlayMontageGUI('ctrMeth','centroid','rotAlignMeth','headtail','scale',true);
            end
          else
            labeler.labelOverlayMontageGUI('ctrMeth','centroid','scale',true);
            warningNoTrace('For aligned overlays, define head/tail points in Track>Landmark Paraneters.');
          end
        end
      end
    end



    function menu_setup_label_outliers_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      label_outlier_gui(labeler) ;
    end



    function menu_setup_set_nframe_skip_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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


      lc = labeler.lblCore;
      assert(isa(lc,'LabelCoreMultiViewCalibrated2'));
      lc.streamlined = ~lc.streamlined;

    end



    function menu_setup_ma_twoclick_align_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;


      lc = labeler.lblCore;
      tftc = ~lc.tcOn;
      labeler.isTwoClickAlign = tftc; % store the state
      lc.setTwoClickOn(tftc);
      src.Checked = onIff(tftc); % skip listener business for now

    end



    function menu_setup_set_labeling_point_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;


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


      lastCalFile = labeler.rcGetProp('lastCalibrationFile');
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
        labeler.viewCalSetProjWideGUI(crObj);%,'tfSetViewSizes',tfSetViewSizes);
      else
        labeler.viewCalSetCurrMovie(crObj);%,'tfSetViewSizes',tfSetViewSizes);
      end


      lc = labeler.lblCore;
      if lc.supportsCalibration,
        lc.setShowCalibration(true);
      end
      obj.menu_setup_use_calibration.Checked = onIff(lc.showCalibration);
      labeler.rcSaveProp('lastCalibrationFile',fname);
    end

    function menu_view_adjustbrightness_actuated_(obj, src, evt)  %#ok<INUSD>



      [tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        try
        	hConstrast = imcontrast_kb(obj.axes_all(iAxRead));
        catch ME
          switch ME.identifier
            case 'images:imcontrast:unsupportedImageType'
              error(ME.identifier,'%s %s',ME.message,'Try View > Display in grayscale.');
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

      tf = ~strcmp(src.Checked,'on');

      labeler.movieForceGrayscale = tf;
      if labeler.hasMovie
        % Pure convenience: update image for user rather than wait for next
        % frame-switch. Could also put this in Labeler.set.movieForceGrayscale.
        labeler.setFrameGUI(labeler.currFrame,'tfforcereadmovie',true);
      end
    end



    function menu_view_gammacorrect_actuated_(obj, src, evt)  %#ok<INUSD>
      [tfok,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if ~tfok
      	return;
      end
      ud = obj.axes_all(iAxApply(1)).UserData;
      if isstruct(ud) && isfield(ud,'gamma') && ~isempty(ud.gamma),
        def = ud.gamma;
      else
        def = 1;        
      end
      val = inputdlg('Gamma value (0 < Gamma < 1 reduces contrast, Gamma > 1 increases contrast)','Gamma correction',[1,50],{num2str(def)});
      if isempty(val)
        return;
      end
      gamma = str2double(val{1});
      if ~(gamma > 0),
        errordlg('Gamma must be a positive number. (0 < Gamma < 1 reduces contrast, Gamma > 1 increases contrast.','Bad value for gamma');
        return;
      end
      ViewConfig.applyGammaCorrection(obj.images_all,obj.axes_all,...
                                      obj.axes_prev,iAxApply,gamma);
    end

    function menu_file_quit_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.quitRequested() ;
    end

    function menu_view_hide_trajectories_actuated_(obj, src, evt) %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.setShowTrx(~labeler.showTrx);
      obj.updateTrxMenuCheckEnable(src);      
    end

    function menu_view_trajectories_showall_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.setShowTrx(true);
      labeler.setShowTrxCurrTargetOnly(false);
      obj.updateTrxMenuCheckEnable(src);
    end

    function menu_view_trajectories_showcurrent_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.setShowTrxCurrTargetOnly(true); 
      obj.updateTrxMenuCheckEnable(src);
    end

    function menu_view_trajectories_dontshow_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.setShowTrx(false); 
      labeler.setShowTrxCurrTargetOnly(false); 
      obj.updateTrxMenuCheckEnable(src);
    end

    function menu_view_trajectories_centervideoontarget_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.movieCenterOnTarget = ~labeler.movieCenterOnTarget;
    end



    function menu_view_rotate_video_target_up_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.movieRotateTargetUp = ~labeler.movieRotateTargetUp;
    end

    function menu_view_fps_actuated_(obj,src,evt)  %#ok<INUSD>
      % redundant with Go > Navigation preferences, but hard to find
      labeler = obj.labeler_ ;
      labeler.navPrefsUI();
    end


    function menu_view_flip_flipud_movie_only_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      [tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        labeler.movieInvert(iAxApply) = ~labeler.movieInvert(iAxApply);
        if labeler.hasMovie
          labeler.setFrameGUI(labeler.currFrame,'tfforcereadmovie',true);
        end
        if ~labeler.isMultiView,
          toggleOnOff(obj.menu_view_flip_flipud_movie_only,'Checked');
        end
      end
    end



    function menu_view_flip_flipud_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;

      [tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        for iAx = iAxApply(:)'
          ax = obj.axes_all(iAx);
          ax.YDir = toggleAxisDir(ax.YDir);
        end
        labeler.UpdatePrevAxesDirections();
        if ~labeler.isMultiView,
          toggleOnOff(obj.menu_view_flip_flipud,'Checked');
        end
      end
    end



    function menu_view_flip_fliplr_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;

      [tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(obj);
      if tfproceed
        for iAx = iAxApply(:)'
          ax = obj.axes_all(iAx);
          ax.XDir = toggleAxisDir(ax.XDir);
          %     if ax==obj.axes_curr
          %       ax2 = obj.axes_prev;
          %       ax2.XDir = toggleAxisDir(ax2.XDir);
          %     end
          labeler.UpdatePrevAxesDirections();
          toggleOnOff(obj.menu_view_flip_flipud,'Checked');
        end
        if ~labeler.isMultiView,
          toggleOnOff(obj.menu_view_flip_fliplr,'Checked');
        end

      end
    end

    function updateFlipMenus(obj)
      labeler = obj.labeler_;
      if labeler.isMultiView,
        return;
      end
      if isempty(labeler.projPrefs),
        return;
      end
      viewCfg = labeler.projPrefs.View;

      movieInvert = ViewConfig.getMovieInvert(viewCfg);
      obj.menu_view_flip_flipud_movie_only.Checked = onIff(any(movieInvert));

      for i = 1:numel(obj.axes_all),
        if strcmpi(obj.axes_all(i).XDir,'reverse'),
          obj.menu_view_flip_fliplr.Checked = 'on';
          break;
        end
      end

      for i = 1:numel(obj.axes_all),
        if strcmpi(obj.axes_all(i).YDir,'normal'),
          obj.menu_view_flip_fliplr.Checked = 'on';
          break;
        end
      end
    end

    % function menu_view_show_axes_toolbar_actuated_(obj, src, evt)  %#ok<INUSD>
    %   ax = obj.axes_curr;
    %   onoff = fif(strcmp(src.Checked,'on'), 'off', 'on') ;  % toggle it
    %   ax.Toolbar.Visible = onoff;
    %   src.Checked = onoff;
    % end



    function menu_view_fit_entire_image_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;

      hAxs = obj.axes_all;
      hIms = obj.images_all;
      assert(numel(hAxs)==numel(hIms));
      arrayfun(@zoomOutFullView,hAxs,hIms,true(1,numel(hAxs)));
      labeler.movieCenterOnTarget = false;


    end



    function menu_view_hide_labels_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      lblCore = labeler.lblCore ;
      if ~isempty(lblCore)
        lblCore.labelsHideToggle() ;
      end
    end



    function menu_view_showhide_preds_all_targets_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;


      tracker = labeler.tracker;
      if ~isempty(tracker)
        tracker.setHideViz(false); % show tracking
        tracker.setShowPredsCurrTargetOnly(false); % not only current target
      end

      obj.updateShowPredMenus();

    end



    function menu_view_showhide_preds_curr_target_only_actuated_(obj, src, evt) %#ok<INUSD>

      

      labeler = obj.labeler_ ;


      tracker = labeler.tracker;
      if ~isempty(tracker)
        tracker.setHideViz(false); % show tracking
        tracker.setShowPredsCurrTargetOnly(true);
        obj.updateShowPredMenus();
      end


    end

    function menu_view_showhide_preds_none_actuated_(obj, src, evt) %#ok<INUSD>



      labeler = obj.labeler_ ;


      tracker = labeler.tracker;
      if ~isempty(tracker)
        tracker.setHideViz(true); % do not show tracking
        obj.updateShowPredMenus();
      end

    end

    function menu_view_hide_predictions_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tracker = labeler.tracker;
      if ~isempty(tracker)
        tracker.setHideViz(~tracker.hideViz); % toggle
        obj.updateShowPredMenus();
      end
    end  % function


    function menu_view_showhide_imported_preds_all_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      labeler.labels2VizShow();
      labeler.labels2VizSetShowCurrTargetOnly(false);
      obj.updateShowImportedPredMenus(src);


    end



    function menu_view_showhide_imported_preds_curr_target_only_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      labeler.labels2VizShow();
      labeler.labels2VizSetShowCurrTargetOnly(true);
      obj.updateShowImportedPredMenus(src);


    end


    function menu_view_showhide_imported_preds_none_actuated_(obj, src, evt)  %#ok<INUSD>



      labeler = obj.labeler_ ;
      labeler.labels2VizHide();
      obj.updateShowImportedPredMenus(src);


    end

    function menu_view_hide_imported_predictions_actuated_(obj,src,evt)  %#ok<INUSD>

      labeler = obj.labeler_ ;
      labeler.labels2VizToggle();
      obj.updateShowImportedPredMenus(src);      

    end

    function menu_view_show_tick_labels_actuated_(obj, src, evt)  %#ok<INUSD>
      % just use checked state of menu for now, no other state
      toggleOnOff(src,'Checked');
      hlpTickGridBang(obj.axes_all, obj.menu_view_show_tick_labels, obj.menu_view_show_grid) ;
    end



    function menu_view_show_grid_actuated_(obj, src, evt)  %#ok<INUSD>
      % just use checked state of menu for now, no other state
      toggleOnOff(src,'Checked');
      hlpTickGridBang(obj.axes_all, obj.menu_view_show_tick_labels, obj.menu_view_show_grid) ;
    end



    function menu_track_setparametersfile_actuated_(obj, src, evt)  %#ok<INUSD>
      % Really, "configure parameters"
      labeler = obj.labeler_ ;
      if any(labeler.bgTrnIsRunningFromTrackerIndex()),
        warndlg('Cannot change training parameters while trackers are training.','Training in progress','modal');
        return
      end
      
      % Actually takes a while for first response to happen, so show busy
      obj.labeler_.pushBusyStatus('Setting training parameters...') ;
      oc = onCleanup(@()(obj.labeler_.popBusyStatus())) ;
      drawnow;

      % Compute the automatic parameters, give user chance to accept/reject them.
      % did_update will be true iff they accepted them.
      % tPrm will we be the current parameter tree, whether or not it incorporates
      % the automatically-generated suggestions.
      [tPrm, did_update, was_canceled] = obj.setAutoParams();
      if was_canceled ,
        return
      end

      % Show the GUI window that allows users to set parameters.  sPrmNew will be
      % empty if user mode no changes, otherwise will be parameter structure holding
      % the new parameters (which have not yet been 'written' to the model).
      sPrmNew = ParameterSetup(obj.mainFigure_,tPrm,'labelerObj',labeler,'name','Training parameters');  % modal

      % Write the parameters to the labeler, if called for.  Set doesNeedSave in the
      % labeler, as needed.     
      if isempty(sPrmNew)
        if did_update
          labeler.setDoesNeedSave(true,'Parameters changed') ;
        end
      else
        labeler.trackSetTrainingParams(sPrmNew);
        labeler.setDoesNeedSave(true,'Parameters changed') ;
      end
    end  % function



    function menu_track_settrackparams_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tPrm = labeler.trackGetTrackParams();
      sPrmTrack = ParameterSetup(obj.mainFigure_, tPrm, 'labelerObj', labeler,'name','Tracking parameters');  % modal
      labeler.setTrackingParameters(sPrmTrack) ;
    end



    function menu_track_auto_params_update_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      checked = get(src,'Checked') ;
      set(src,'Checked',~checked) ;  % No no no no no
      labeler.trackAutoSetParams = ~checked;
      labeler.setDoesNeedSave(true, 'Auto compute training parameters changed') ;
    end



    function menu_track_use_all_labels_to_train_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tObj = labeler.tracker;
      if isempty(tObj)
        error('LabelerController:tracker','No tracker for this project.');
      end
      if tObj.hasTrained && tObj.trnDataDownSamp
        resp = ...
          questdlg('A tracker has already been trained with downsampled training data. Proceeding will clear all previous trained/tracked results. OK?',...
                   'Clear Existing Tracker','Yes, clear previous tracker','Cancel', ...
                   'Cancel');
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
      labeler.trainIncremental();
    end



    function menu_go_targets_summary_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if labeler.maIsMA
        TrkInfoUI(labeler);
      else
        obj.raiseTargetsTableFigure_();
      end
    end



    function menu_go_nav_prefs_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.navPrefsUI();
    end



    function menu_evaluate_gt_frames_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.gtShowGTManager();
    end



    function menu_evaluate_crossvalidate_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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
        error('LabelerGUI:xvalid', 'Number of folds must be a positive integer greater than one.') ;
      end
      tbl.split = ceil(nfold*rand(n,1));
      t = labeler.tracker;
      t.trainsplit(tbl);
    end



    function menu_track_clear_tracking_results_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      % legacy behavior not sure why; maybe b/c the user is prob wanting to increase avail mem
      %labeler.preProcInitData();
      res = questdlg('Are you sure you want to clear tracking results?');
      if ~strcmpi(res,'yes'),
        return;
      end
      labeler.clearTrackingResults();
    end



    function menu_track_batch_track_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tbobj = TrackBatchGUI(labeler);
      tbobj.run();
    end



    function menu_track_all_movies_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mIdx = labeler.allMovIdx();
      toTrackIn = labeler.mIdx2TrackListGUI(mIdx);
      tbobj = TrackBatchGUI(labeler,'toTrack',toTrackIn);
      % [toTrackOut] = tbobj.run();
      tbobj.run();
      % todo: import predictions
    end



    function menu_track_current_movie_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      mainFigure = obj.mainFigure_ ;
      mIdx = labeler.currMovIdx;
      toTrackIn = labeler.mIdx2TrackListGUI(mIdx);
      mdobj = SpecifyMovieToTrackGUI(labeler,mainFigure,toTrackIn);
      [toTrackOut,dostore] = mdobj.run();
      if ~dostore,
        return;
      end
      labeler.trackBatch('toTrack',toTrackOut);
    end



    function menu_file_clear_imported_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.labels2Clear();
    end



    function menu_file_export_all_movies_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      nMov = labeler.nmoviesGTaware;
      if nMov==0
        error('LabelerGUI:noMov','No movies in project.');
      end
      iMov = 1:nMov;
      [tfok,rawtrkname] = obj.getExportTrkRawNameUI();
      if ~tfok
        return;
      end
      labeler.trackExportResultsGUI(iMov,'rawtrkname',rawtrkname);
    end



    function menu_track_set_labels_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;

      tracker = labeler.tracker;
      if labeler.gtIsGTMode
        error('LabelerGUI:gt','Unsupported in GT mode.');
      end

      frm = labeler.currFrame;
      if ~isempty(tracker) && tracker.hasBeenTrained() && (~labeler.maIsMA)
        % single animal. Use prediction if available else use imported below
        [tfhaspred,xy,tfocc] = tracker.getTrackingResultsCurrFrm(); %#ok<ASGLU>
        itgt = labeler.currTarget;

        if ~tfhaspred(itgt)
          if (labeler.nTrx>1)
            msgbox('No predictions for current frame.');
            return;
          else % for single animal use imported predictions if available
            iMov = labeler.currMovie;
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

        labeler.lblCore.newFrame(frm,frm,1,true);

      else
        iMov = labeler.currMovie;
        frm = labeler.currFrame;
        if iMov==0
          error('LabelerGUI:setLabels','No movie open.');
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
          if isempty(tracker)
            usePred = false;
          elseif isempty(tracker.trkVizer)
            usePred = false;
          else
            [tfhaspred,xy,tfocc] = tracker.getTrackingResultsCurrFrm(); %#ok<ASGLU>
            iTgtPred = tracker.trkVizer.currTrklet;
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
            iTgt = tracker.trkVizer.currTrklet;
            [~,xy,tfocc] = tracker.getTrackingResultsCurrFrm();
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
            error('LabelerGUI:setLabels','Unsupported for multiple targets.');
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
    end  % function



    function menu_track_background_predict_start_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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



    function menu_track_background_predict_end_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tObj = labeler.tracker;
      if tObj.asyncIsPrepared
        tObj.asyncStopBgRunner();
      else
        warndlg('Background worker is not running.','Background tracking');
      end
    end



    function menu_track_background_predict_stats_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tObj = labeler.tracker;
      if tObj.asyncIsPrepared
        tObj.asyncComputeStats();
      else
        warningNoTrace('LabelerGUI:bgTrack',...
          'No background tracking information available.','Background tracking');
      end
    end



    function menu_evaluate_gtmode_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.gtToggleGTMode() ;
      if labeler.gtIsGTMode,
        obj.gtShowGTManager();
        if ~obj.labeler_.hasMovie,
          obj.ShowMovieManager();
        end
      else
        obj.gtCloseGTManager();
      end
    end



    function menu_evaluate_gtloadsuggestions_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      LabelerGT.loadSuggestionsUI(labeler);
    end


    function menu_evaluate_gtsavesuggestions_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      LabelerGT.saveSuggestionsUI(labeler);
    end

    function menu_evaluate_gtsetsuggestions_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      LabelerGT.setSuggestionsToLabeledUI(labeler);
    end



    function menu_evaluate_gtcomputeperf_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      assert(labeler.gtIsGTMode);
      response = obj.askAboutUnrequestedGTLabelsIfNeeded_() ;
      if strcmp(response, 'cancel')
        return
      end      
      whichlabels = response ;
      labeler.gtComputeGTPerformance('whichlabels',whichlabels);
    end

    function response = askAboutUnrequestedGTLabelsIfNeeded_(obj)
      labeler = obj.labeler_ ;
      assert(labeler.gtIsGTMode);
      nNewLbls = labeler.gtComputeNewLabelCount() ;
      if nNewLbls == 0
         response = 'suggestonly' ;  % will be ignored when the rubber meets the road, but that's ok
         return
      end
      res = questdlg(sprintf('%d labeled frames were not in the to-label list, include them in analysis?',nNewLbls),'Update to-label list?','Yes','No','Cancel','Yes');
      if strcmpi(res,'Cancel'),
        response = 'cancel' ;
      elseif strcmpi(res,'No'),
        response = 'suggestonly' ;
      elseif strcmpi(res,'Yes'),
        response = 'all' ;
      else
        error('Internal error: The dialog returned an unanticpated value') ;
      end
    end  % function

    function menu_evaluate_gtcomputeperfimported_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      assert(labeler.gtIsGTMode);

      response = obj.askAboutUnrequestedGTLabelsIfNeeded_() ;
      if strcmp(response, 'cancel')
        return
      end      
      whichlabels = response ;

      labeler.gtComputeGTPerformance('whichlabels',whichlabels,'useLabels2',true);
    end

    function menu_evaluate_gtexportresults_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tblRes = labeler.gtTblRes;
      if isempty(tblRes)
        errordlg('No GT results are currently available.','Export GT Results');
        return;
      end
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
      % Set the current property to the one with index get(get,'Value').  Handles
      % the case where this is a custom feature: Pops up a dialog in this case.

      % Get the value
      iprop = get(src,'Value');

      % Do the core stuff
      labeler = obj.labeler_ ;
      itm = labeler.infoTimelineModel ;
      if itm.getCurPropTypeIsAllFrames() && strcmpi(itm.props_allframes(iprop).name,'Add custom...')
        movfile = labeler.getMovieFilesAllFullMovIdx(labeler.currMovIdx);
        defaultpath = fileparts(movfile{1});
        [f,p] = uigetfile('*.mat','Select .mat file with a feature value for each frame for current movie',defaultpath);
        if ~ischar(f)
          return
        end
        file = fullfile(p,f);
        labeler.addCustomTimelineFeatureGivenFileName(file) ;
      else
        labeler.setTimelineCurrentPropertyType(itm.curproptype, iprop) ;
      end

      % This does something important
      obj.hlpRemoveFocus_() ;
    end

    function menu_InfoTimeline_SetNumFramesShown_actuated_(obj, src, evt)  %#ok<INUSD>
      % Pop up a dialog to ask user for the number of frames to be shown in the
      % timeline, then call a Labeler method to make it so. 
      frmRad = obj.labeler_.projPrefs.InfoTimelines.FrameRadius;
      answer = inputdlg('Number of frames (0 to show full movie)',...
                        'Timeline',1,{num2str(2*frmRad)});
      if ~isempty(answer)
        nframes = str2double(answer{1});
        obj.labeler_.setTimelineFramesInView(nframes) ;
      end
    end  % function

    function menu_InfoTimeline_ClearBout_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.labeler_.clearBoutInTimeline() ;
    end

    function menu_InfoTimeline_ToggleThresholdViz_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.labeler_.toggleTimelineIsStatThreshVisible();
    end

    function pbPlaySeg_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if ~labeler.doProjectAndMovieExist()
        return
      end
      obj.play_('videoPlaySegFwdEnding') ;
    end



    function pbPlaySegRev_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if ~labeler.doProjectAndMovieExist()
        return
      end
      obj.play_('videoPlaySegRevEnding') ;
    end



    function pbPlay_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      if ~labeler.doProjectAndMovieExist()
        return
      end
      obj.play_('videoPlay') ;
    end



    function tbAdjustCropSize_actuated_(obj, src, evt)  %#ok<INUSD>
      obj.cropUpdateCropAdjustingCropSize_() ;
      tb = obj.tbAdjustCropSize;
      if tb.Value==tb.Min
        % user clicked "Done Adjusting"
        warningNoTrace('All movies in a given view must share the same crop size. The sizes of all crops have been updated as necessary.');
      elseif tb.Value==tb.Max
        % user clicked "Adjust Crop Size"
        labeler = obj.labeler_ ;
        if ~labeler.cropProjHasCrops
          labeler.cropInitCropsAllMovies;
          fprintf(1,'Default crop initialized for all movies.\n');
          obj.cropUpdateCropHRects_();
        end
      end
    end



    function pbClearAllCrops_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.cropClearAllCrops();
    end



    function menu_file_export_labels2_trk_curr_mov_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      iMov = labeler.currMovie;
      if iMov==0
        error('LabelerGUI:noMov','No movie currently set.');
      end
      [tfok,rawtrkname] = obj.getExportTrkRawNameUI();
      if ~tfok
        return;
      end
      labeler.trackExportResultsGUI(iMov,'rawtrkname',rawtrkname);
    end

    function menu_file_import_export_advanced_actuated_(obj, src, evt)  %#ok<INUSD>
    end

    function menu_track_tracking_algorithm_actuated_(obj, src, evt)  %#ok<INUSD>

      labeler = obj.labeler_;
      if labeler.tracker.bgTrnIsRunning
        uiwait(warndlg('Cannot change tracker while training is in progress.','Training in progress'));
        return;
      end
      if labeler.tracker.bgTrkIsRunning
        uiwait(warndlg('Cannot change tracker while tracking is in progress.','Tracking in progress'));
        return;
      end

      labeler.pushBusyStatus('Creating new tracker...') ; 
      oc = onCleanup(@()(labeler.popBusyStatus()));
      drawnow;
      SelectTrackingAlgorithm(labeler,obj.mainFigure_);

    end

    function menu_view_keypoint_appearance_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      cbkApply = @(varargin)(labeler.hlpApplyCosmetics(varargin{:})) ;
      LandmarkColors(labeler,cbkApply);
      % AL 20220217: changes now applied immediately
      % if ischange
      %   cbkApply(savedres.colorSpecs,savedres.markerSpecs,savedres.skelSpecs);
      % end
    end

    function menu_track_edit_skeleton_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      landmark_specs('lObj',labeler);
    end

    function menu_track_viz_dataaug_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.trainAugOnly() ;
    end

    function menu_view_showhide_skeleton_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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

      if strcmpi(get(src,'Checked'),'off'),
        labeler.setShowMaRoi(true);
      else
        labeler.setShowMaRoi(false);
      end
    end

    function menu_view_showhide_maroiaux_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      tf = strcmpi(get(src,'Checked'),'off');
      labeler.setShowMaRoiAux(tf);
    end

    function popupmenu_prevmode_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
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
      labeler.setPrevAxesMode(PrevAxesMode.FROZEN);
    end

    function pushbutton_exitcropmode_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.cropSetCropMode(false);
    end

    function menu_view_occluded_points_box_actuated_(obj, src, evt)  %#ok<INUSD>
      labeler = obj.labeler_ ;
      labeler.setShowOccludedBox(~labeler.showOccludedBox);
      if labeler.showOccludedBox,
        labeler.lblCore.showOcc();
      else
        labeler.lblCore.hideOcc();
      end
    end

    function pumInfo_labels_actuated_(obj, src, evt)  %#ok<INUSD>
      ipropType = get(src,'Value');
      iprop = get(obj.pumInfo,'Value');
      labeler = obj.labeler_ ;
      itm = labeler.infoTimelineModel ;
      props = itm.getPropsDisp(ipropType);
      if iprop > numel(props),
        iprop = 1;
      end
      set(obj.pumInfo,'String',props,'Value',iprop);  
        % Will happen via update event, but this is faster, for immediate feedback (?)
      obj.labeler_.setTimelineCurrentPropertyType(ipropType,iprop);
    end  % function

    function updateHighlightingOfAxes(obj)
      labeler = obj.labeler_ ;
      if labeler.isinit ,
        return
      end
      if labeler.gtIsGTMode
        tfHilite = labeler.gtCurrMovFrmTgtIsInGTSuggestions();
      else
        tfHilite = false;
      end
      if ~isempty(obj.axesesHighlightManager_) ,
        obj.axesesHighlightManager_.setHighlight(tfHilite);
      end
    end
    
    function cbkGTSuggUpdated(obj, ~, ~)
      % Update the main window controls when the GT suggestions change.
      obj.labelTLInfo.updateGTModeRelatedControls() ;
    end

    % function cbkGTResUpdated(obj, s, e)
    %   % i think there are listeners in the GTManager, not sure why we need
    %   % this too
    %   % if ~exist('s', 'var') ,
    %   %   s = [] ;
    %   % end
    %   % if ~exist('e', 'var') ,
    %   %   e = [] ;
    %   % end
    % end

    function gtGoToNextUnlabeled(obj)
      
      labeler = obj.labeler_;
      assert(labeler.gtIsGTMode);

      nextmft = labeler.gtNextUnlabeledMFT();
      if isempty(nextmft),
        msgbox('No more unlabeled frames in to-label list.','','modal');
        return;
      end

      iMov = nextmft.mov.get();
      if iMov~=labeler.currMovie
        labeler.movieSetGUI(iMov);
      end
      labeler.setFrameAndTargetGUI(nextmft.frm,nextmft.iTgt);
    end


    function update(obj)
      % Intended to be a full update of all GUI controls to bring them into sync
      % with obj.labeler_.  Currently a work in progress.
      obj.updateEnablementOfManyControls() ;
      obj.cbkLabelModeChanged() ;
      obj.cbkShowTrxChanged() ;
      obj.cbkShowTrxCurrTargetOnlyChanged() ;
      obj.updatePUMTrackAndFriend() ;
      obj.updateTargetCentrationAndZoom() ;
      obj.cbkMovieCenterOnTargetChanged() ;
      obj.cbkMovieForceGrayscaleChanged() ;
      obj.updateMainFigureName() ;
      obj.updateMainAxesName() ;
      obj.updateGUIFigureNames() ;
      obj.updateMainFigureName() ;
      obj.cbkShowOccludedBoxChanged() ;
      obj.cbkUpdateCropGUITools() ;
      obj.updateGTModeRelatedControls() ;
      if ~isempty(obj.movieManagerController_) && obj.movieManagerController_.isValid(),
        obj.movieManagerController_.lblerLstnCbkGTMode() ; % todo check if needed
      end
      obj.updateShowPredMenus();
      obj.updateShowImportedPredMenus();
      obj.updateFlipMenus();
      obj.update_menu_track_tracker_history() ;
      obj.update_menu_track_backend_config();
      obj.update_text_trackerinfo() ;
      obj.updateStatusAndPointer() ;
      obj.updateBackgroundProcessingStatus_() ;
      obj.cbkGTSuggUpdated() ;
      % obj.cbkGTResUpdated() ;
      obj.cbkCurrTrackerChanged() ;
      if ~isempty(obj.movieManagerController_) && obj.movieManagerController_.isValid(),
        obj.movieManagerController_.hlpLblerLstnCbkUpdateTable() ; % todo check if needed
      end
      sendMaybe(obj.trainingMonitorVisualizer_, 'updateStopButton') ;
      sendMaybe(obj.trackingMonitorVisualizer_, 'updateStopButton') ;
    end
    
    function save(obj)
      % Try to save to current project; if there is no project file name specified
      % yet, do a saveas.
      labeler = obj.labeler_ ;
      lblfname = labeler.projectfile;
      if isempty(lblfname)
        obj.saveAs();
      else
        labeler.projSave(lblfname);
      end
    end  % function
    
    function saveAs(obj)
      % Saves a .lbl file, prompting user for filename
      labeler = obj.labeler_ ;
      if ~isempty(labeler.projectfile)
        filterspec = labeler.projectfile;
      else
        % Guess a path/location for save
        lastLblFile = labeler.rcGetProp('lastLblFile');
        if isempty(lastLblFile)
          if labeler.hasMovie
            savepath = fileparts(labeler.moviefile);
          else
            savepath = pwd;
          end
        else
          savepath = fileparts(lastLblFile);
        end
        
        if ~isempty(labeler.projname)
          projfile = sprintf(labeler.DEFAULT_LBLFILENAME,labeler.projname);
        else
          projfile = sprintf(labeler.DEFAULT_LBLFILENAME,'APTProject');
        end
        filterspec = fullfile(savepath,projfile);
      end
      
      [lblfname,pth] = uiputfile(filterspec,'Save label file');
      if isequal(lblfname,0)
        return
      end
      lblFilePath = fullfile(pth, lblfname) ;

      labeler.projSave(lblFilePath) ;
    end  % function
    
    function labels2ImportTrkPromptAuto(obj, iMovs)
      % See labelImportTrkPromptAuto().
      % iMovs: works per current GT mode
      
      labeler = obj.labeler_ ;      
      if exist('iMovs','var')==0
        iMovs = 1:labeler.nmoviesGTaware;
      end      
      obj.labelImportTrkPromptGenericAuto(iMovs,'labels2ImportTrk');
    end
    
    function labelImportTrkPromptGenericAuto(obj,iMovs,importFcn)
      % Come up with trkfiles based on iMovs and then call importFcn.
      % 
      % iMovs: index into .movieFilesAllGTAware
      
      labeler = obj.labeler_ ;      
      PROPS = labeler.gtGetSharedProps();
      movfiles = labeler.(PROPS.MFAF)(iMovs,:);
      [tfsucc,trkfilesUse] = LabelerController.labelImportTrkFindTrkFilesPrompt(movfiles);
      if tfsucc
        feval(importFcn,labeler,iMovs,trkfilesUse);
      else
        if isscalar(iMovs) && labeler.nview==1
          % In this case (single movie, single view) failure can occur if 
          % no trkfile is found alongside movie, or if user cancels during
          % a prompt.
          
          lastTrkFileImported = labeler.rcGetProp('lastTrkFileImported');
          if isempty(lastTrkFileImported)
            lastTrkFileImported = pwd;
          end
          [fname,pth] = uigetfile('*.trk','Import trkfile',lastTrkFileImported);
          if isequal(fname,0)
            return;
          end
          trkfile = fullfile(pth,fname);
          feval(importFcn,labeler,iMovs,{trkfile});
        end
      end      
    end

    function tf = isGTManagerFigure(obj)
      hGTMgr = obj.GTManagerFigure ;
      tf = ~isempty(hGTMgr) && ishandle(hGTMgr);
    end

    function gtShowGTManager(obj)
      if obj.isGTManagerFigure()
        %hGTMgr.Visible = 'on';
        figure(obj.GTManagerFigure);
      else
        obj.GTManagerFigure = GTManager(obj.labeler_);
      end
    end

    function gtCloseGTManager(obj)
      if obj.isGTManagerFigure(),
        close(obj.GTManagerFigure);
      end
    end

  end  % methods

  methods (Static)
    function [tfsucc,trkfilesUse] = labelImportTrkFindTrkFilesPrompt(movfiles)
      % Find trkfiles present for given movies. Prompt user to pick a set
      % if more than one exists.
      %
      % movfiles: [nTrials x nview] cellstr
      %
      % tfsucc: if true, trkfilesUse is valid; if false, trkfilesUse is
      % intedeterminate
      % trkfilesUse: cellstr, same size as movfiles. Full paths to trkfiles
      % present/selected for import
      
      [trkfilesCommon,kwCommon] = Labeler.getTrkFileNamesForImport(movfiles);
      nCommon = numel(kwCommon);
      
      tfsucc = false;
      trkfilesUse = [];
      switch nCommon
        case 0
          warningNoTrace('Labeler:labelImportTrkPrompt',...
            'No consistently-named trk files found across %d given movies.',numel(movfiles));
          return;
        case 1
          trkfilesUseIdx = 1;
        otherwise
          msg = sprintf('Multiple consistently-named trkfiles found. Select trkfile pattern to import.');
          uiwait(msgbox(msg,'Multiple trkfiles found','modal'));
          trkfileExamples = trkfilesCommon{1};
          for i=1:numel(trkfileExamples)
            [~,trkfileExamples{i}] = myfileparts(trkfileExamples{i});
          end
          [sel,ok] = listdlg(...
            'Name','Select trkfiles',...
            'Promptstring','Select a trkfile (pattern) to import.',...
            'SelectionMode','single',...
            'listsize',[300 300],...
            'liststring',trkfileExamples);
          if ok
            trkfilesUseIdx = sel;
          else
            return;
          end
      end
      trkfilesUse = cellfun(@(x)x{trkfilesUseIdx},trkfilesCommon,'uni',0);
      tfsucc = true;
    end  % function
  end  % methods (Static)

  methods
    function updateAfterCurrentFrameSet(obj)
      labeler = obj.labeler_ ;
      obj.labelTLInfo.updateAfterCurrentFrameSet();
      set(obj.edit_frame,'String',num2str(labeler.currFrame));
      sldval = (labeler.currFrame-1)/(labeler.nframes-1);
      if isnan(sldval)
        sldval = 0;
      end
      set(obj.slider_frame,'Value',sldval);
      hasProject = labeler.hasProject ;
      hasMovie = labeler.hasMovie ;        
      set(obj.pbClearSelection,'Enable',onIff(hasProject && hasMovie && labeler.areAnyFramesSelected())) ;      
      obj.updateHighlightingOfAxes() ;      
    end  % function

    function deleteSpashScreenFigureIfItExists_(obj)
      hfigsplash = obj.splashScreenFigureOrEmpty_ ;
      if isempty(hfigsplash) || ~ishghandle(hfigsplash) 
        obj.splashScreenFigureOrEmpty_ = [] ;
        return
      end
      % main_figure = obj.mainFigure_ ;
      % refocusSplashScreen(hfigsplash, main_figure) ;  % why refocus on the splash screen just before deleting it?  -- ALT, 2025-07-08
      delete(hfigsplash) ;
      obj.splashScreenFigureOrEmpty_ = [] ;
    end

    function handleCreationTimeAdditionalArgumentsGUI_actuated_(obj, ~, ~, varargin)
      obj.labeler_.handleCreationTimeAdditionalArgumentsGUI_(varargin{:}) ;
    end

    function trainMonitorVizCloseRequested(obj)
      doReallyClose = false ;
      tfbatch = batchStartupOptionUsed() ; % ci
      trainMonitorViz = obj.trainingMonitorVisualizer_ ;
      if tfbatch ,
        doReallyClose = true ;
      else        
        trainMonitorFig = trainMonitorViz.hfig ;
        handles = guidata(trainMonitorFig) ;
  
        mode = get(handles.pushbutton_startstop,'UserData');  % this is not a good way to store application state.
  
        if strcmpi(mode,'stop') ,
          res = questdlg({'Training currently in progress. Please stop training before'
                          'closing this monitor. If you have already clicked Stop training,'
                          'please wait for training processes to be killed before closing'
                          'this monitor.'
                          'Only override this warning if you know what you are doing.'} , ...
                         'Stop training before closing monitor', ...
                         'Ok','Override and close anyways', ...
                         'Ok');
          if ~strcmpi(res,'Ok'),
            doReallyClose = true ;
          end
        elseif strcmpi(mode,'start') || strcmpi(mode,'done') ,
          doReallyClose = true ;
        else
          % sanity check
          error('Internal error: Bad userdata value for pushbutton_startstop');
        end
      end

      if doReallyClose ,
        delete(trainMonitorViz);
        obj.trainingMonitorVisualizer_ = [] ;
      end        
    end  % function

    function trackMonitorVizCloseRequested(obj)
      doReallyClose = false ;
      tfbatch = batchStartupOptionUsed() ; % ci
      trackMonitorViz = obj.trackingMonitorVisualizer_ ;
      if tfbatch ,
        doReallyClose = true ;
      else        
        trackMonitorFig = trackMonitorViz.hfig ;
        handles = guidata(trackMonitorFig) ;
  
        mode = get(handles.pushbutton_startstop,'UserData');  % this is not a good way to store application state.
  
        if strcmpi(mode,'stop') ,
          res = questdlg({'Tracking currently in progress. Please stop tracking before'
                          'closing this monitor. If you have already clicked Stop tracking,'
                          'please wait for tracking processes to be killed before closing'
                          'this monitor.'
                          'Only override this warning if you know what you are doing.'} , ...
                         'Stop tracking before closing monitor', ...
                         'Ok','Override and close anyways', ...
                         'Ok');
          if ~strcmpi(res,'Ok'),
            doReallyClose = true ;
          end
        elseif strcmpi(mode,'start') || strcmpi(mode,'done') ,
          doReallyClose = true ;
        else
          % sanity check
          error('Internal error: Bad userdata value for pushbutton_startstop');
        end
      end

      if doReallyClose ,
        delete(trackMonitorViz);
        obj.trackingMonitorVisualizer_ = [] ;
      end        
    end  % function

    function [tPrm, did_update, was_canceled] = setAutoParams(obj)
      % Compute auto parameters and update them based on user feedback.
      %
      % AL: note this sets the project-level params based on the current
      % tracker; if a user uses multiple tracker types (eg: MA-BU and 
      % MA-TD) and switches between them, the behavior may be odd (eg the
      % user may get prompted constantly about "changed suggestions" etc)

      % On exit, returns the current parameter tree in the labeler in tPrm (whether
      % modified or not).  do_update is a logical scalar that is true iff the
      % suggested automatically-determined paramters were applied to the labeler.

      labeler = obj.labeler_ ;
        
      sPrmCurrent = labeler.trackGetTrainingParams();
      % Future todo: if sPrm0 is empty (or partially-so), read "last params" in 
      % eg RC/lastCPRAPTParams. Previously we had an impl but it was messy, start
      % over.
      
      % Start with default "new" parameter tree/specification
      tPrm = APTParameters.defaultParamsTree() ;
      % Overlay our starting point
      tPrm.structapply(sPrmCurrent) ;
      
      if labeler.isMultiView        
        warningNoTrace('Multiview project: not auto-setting params.');
        did_update = false;
        was_canceled = false ;
        return
      end      
      
      if labeler.trackerIsTwoStage && ~labeler.trackerIsObjDet && isempty(labeler.skelHead)
        uiwait(warndlg('For head-tail based tracking method please select the head and tail landmarks', [], 'modal')) ;
        landmark_specs('lObj',labeler,'waiton_ui',true);
        if isempty(labeler.skelHead)
          uiwait(warndlg('Head Tail landmarks are not specified to enable auto setting of training parameters. Using the default parameters', ...
                         [], ...
                         'modal'));
          did_update = false;
          was_canceled = false ;        
          return
        end
      end
      
      [tPrm, was_canceled, do_update] = APTParameters.autosetparamsGUI(tPrm, labeler) ;
      if was_canceled
        did_update = false ;
        return
      end

      % Finally, apply the update, if called for.
      if do_update
        sPrmNew = tPrm.structize() ;
        labeler.trackSetTrainingParams(sPrmNew);
        did_update = true ;
      else
        did_update = false ;
      end
    end  % function
        
    function [docontinue, stg1ctorargs, stg2ctorargs] = raiseDialogsToChooseStageAlgosForCustomTopDownTracker(obj, stg1mode, stg2mode)
      % What it says on the tin.
      dlnets = enumeration('DLNetType') ;
      isma = [dlnets.isMultiAnimal] ;
      stg2nets = dlnets(~isma) ;
      
      is_bbox = false(1,numel(dlnets)) ;
      for dndx = 1:numel(dlnets)          
        is_bbox(dndx) = dlnets(dndx).isMultiAnimal && startsWith(char(dlnets(dndx)),'detect_') ;
      end  % for
      
      stg1nets_ht = dlnets(isma & ~is_bbox) ;
      stg1nets_bbox = dlnets(isma & is_bbox) ;
      if stg1mode == DLNetMode.multiAnimalTDDetectHT
        stg1nets = stg1nets_ht ;
      else
        stg1nets = stg1nets_bbox ;
      end
      [stg1net, stg2net] = apt.get_custom_two_stage_tracker_nets_ui(obj.mainFigure_, stg1nets, stg2nets) ;

      docontinue = ~isempty(stg1net) ;
      if docontinue
        stg1ctorargs = {'trnNetMode', stg1mode, 'trnNetType', stg1net} ;
        stg2ctorargs = {'trnNetMode', stg2mode, 'trnNetType', stg2net} ;
      else
        stg1ctorargs = [] ;
        stg2ctorargs = [] ;
      end      
    end  % function

    function cbkGTSuggMFTableLbledUpdated(obj)
      % React to incremental update to labeler.gtSuggMFTableLbled
      obj.labelTLInfo.updateGTModeRelatedControlsLight();
    end

    function timelineButtonDown(obj, src, evt)
      labeler = obj.labeler_;
      if ~labeler.isReady || ~labeler.hasProject || ~labeler.hasMovie
        return
      end

      if evt.Button==1
        % Navigate to clicked frame        
        pos = get(src,'CurrentPoint');
        if labeler.hasTrx,
          [sf,ef] = labeler.trxGetFrameLimits();
        else
          sf = 1;
          ef = labeler.nframes ;
        end
        frm = round(pos(1,1));
        frm = min(max(frm,sf),ef);
        labeler.setFrameGUI(frm);
      end
    end  % function

    function backendTestFigureCloseRequested(obj)
      delete(obj.backendTestController_) ;
      obj.backendTestController_ = [] ;
    end

  end  % methods  
end  % classdef
