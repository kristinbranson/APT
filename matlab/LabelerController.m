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
      obj.mainFigure_ = LabelerGUI(labeler, obj) ;
      obj.labeler_.registerController(obj) ;  % hack
      obj.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      obj.isInYodaMode_ = isInYodaMode ;  
        % If in yoda mode, we don't wrap GUI-event function calls in a try..catch.
        % Useful for debugging.

      % Create the waitbar figure, which we re-use  
      obj.waitbarFigure_ = waitbar(0, '', ...
                                   'Visible', 'off', ...
                                   'CreateCancelBtn', @(source,event)(obj.didCancelWaitbar())) ;
      obj.waitbarFigure_.CloseRequestFcn = @(source,event)(nop()) ;
        
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
        addlistener(labeler,'updateTrackerInfoText',@(source,event)(obj.updateTrackerInfoText()));
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
        addlistener(labeler,'didSetProjname',@(source,event)(obj.didChangeProjectName()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetProjFSInfo',@(source,event)(obj.cbkProjFSInfoChanged()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler,'didSetMovieInvert',@(source,event)(obj.cbkMovieInvertChanged()));      
      obj.listeners_(end+1) = ...
        addlistener(labeler.progressMeter, 'didArm', @(source,event)(obj.armWaitbar())) ;      
      obj.listeners_(end+1) = ...
        addlistener(labeler.progressMeter, 'update', @(source,event)(obj.updateWaitbar())) ;      

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
        if isfield(handles,'figs_all') && any(ishandle(handles.figs_all)),
          set(handles.figs_all(ishandle(handles.figs_all)),'Pointer','watch');
        else
          set(obj.mainFigure_,'Pointer','watch');
        end
      else
        color = handles.idlestatuscolor;
        if isfield(handles,'figs_all') && any(ishandle(handles.figs_all)),
          set(handles.figs_all(ishandle(handles.figs_all)),'Pointer','arrow');
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
    
    function didSetLblCore(obj)
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

    function pbTrack_actuated(obj, source, event)
      obj.track_core_(source, event) ;
    end

    function menu_start_tracking_but_dont_call_apt_interface_dot_py_actuated(obj, source, event)
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

    function menu_debug_generate_db_actuated(obj, source, event)
      obj.train_core_(source, event, 'do_just_generate_db', true) ;
    end

    function pbTrain_actuated(obj, source, event)
      obj.train_core_(source, event) ;
    end

    function menu_start_training_but_dont_call_apt_interface_dot_py_actuated(obj, source, event)
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
        res = questdlg('Project has unsaved changes. Save before training?','Save Project','Save As','No','Cancel','Save As');
        if strcmp(res,'Cancel')
          return
        elseif strcmp(res,'Save As')
          menu_file_saveas_Callback(source, event, guidata(source))
        end    
      end
      
      labeler.setStatus('Training...');
      drawnow;
      [tfCanTrain,reason] = labeler.trackCanTrain();
      if ~tfCanTrain,
        errordlg(['Error training tracker: ',reason],'Error training tracker');
        labeler.clearStatus();
        return
      end
      
      fprintf('Training started at %s...\n',datestr(now()));
      oc1 = onCleanup(@()(labeler.clearStatus()));
      labeler.train(...
        'trainArgs',{}, ...
        'do_just_generate_db', do_just_generate_db, ...
        'do_call_apt_interface_dot_py', do_call_apt_interface_dot_py) ;
    end  % method

    function menu_quit_but_dont_delete_temp_folder_actuated(obj, source, event)  %#ok<INUSD> 
      obj.labeler_.projTempDirDontClearOnDestructor = true ;
      obj.quitRequested() ;
    end  % method    

    function menu_track_backend_config_aws_configure_actuated(obj, source, event)  %#ok<INUSD> 
      obj.selectAwsInstance_('canlaunch',1,...
                             'canconfigure',2, ...
                             'forceSelect',1) ;
    end

    function menu_track_backend_config_aws_setinstance_actuated(obj, source, event)  %#ok<INUSD> 
      obj.selectAwsInstance_() ;
    end

    function menu_track_algorithm_actuated(obj, source, event)  %#ok<INUSD> 
      % Get the tracker index
      tracker_index = source.UserData;

      % Validate it
      tAll = obj.labeler_.trackersAll;
      tracker_count = numel(tAll) ;
      if isnumeric(tracker_index) && isscalar(tracker_index) && round(tracker_index)==tracker_index && 1<=tracker_index && tracker_index<=tracker_count ,
        % all is well
      else
        error('APT:invalidPropertyValue', 'Invalid tracker index') ;
      end
      
      % If a custom top-down tracker, ask if we want to keep it or make a new one.
      previousTracker = tAll{tracker_index};
      if isa(previousTracker,'DeepTrackerTopDownCustom')
        do_use_previous = ask_if_should_use_previous_custom_top_down_tracker(previousTracker) ;
      else
        do_use_previous = [] ;  % value will be ignored
      end  % if isa(tAll{iTrk},'DeepTrackerTopDownCustom')
      
      % Finally, call the model method to set the tracker
      obj.labeler_.trackSetCurrentTracker(tracker_index, do_use_previous);      
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
        methodName=[controlName '_actuated'] ;
        if ismethod(obj,methodName) ,
          obj.(methodName)(source, event, varargin{:});
        end
      else
        type=get(source,'Type');
        if isequal(type,'uitable') ,
          if isfield(event,'EditData') || isprop(event,'EditData') ,  % in older Matlabs, event is a struct, in later, an object
            methodName=[controlName '_cell_edited'];
          else
            methodName=[controlName '_cell_selected'];
          end
          if ismethod(obj,methodName) ,
            obj.(methodName)(source, event, varargin{:});
          end
        elseif isequal(type,'uicontrol') || isequal(type,'uimenu') ,
          methodName=[controlName '_actuated'] ;
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
          set(handles.FigureToolBar,'Visible','off')
      
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
          
          lObj = handles.labelerObj;
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
      %      set(handles.menu_track_id,'Checked',handles.labelerObj.track_id,'Visible','on');
          else
            set(handles.h_ma_only,'Enable','off');
      %      set(handles.menu_track_id,'Visible','off');
          end
          if lObj.nLabelPointsAdd == 0,
            set(handles.h_addpoints_only,'Visible','off');
          else
            set(handles.h_addpoints_only,'Visible','on');
          end
      
        otherwise
          fprintf('Not implemented\n');
      end
    end  % function

    function updateTrackerInfoText(obj)
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
    
    function target_table_row_actuated(obj, source, event, row, rowdata)  %#ok<INUSD>
      % Does what needs doing when the target table row is selected.
      labeler = obj.labeler_ ;
      labeler.setMFT(rowdata.mov,rowdata.frm1,rowdata.iTgt) ;
    end  % function

    function target_table_update_button_actuated(obj, source, event)  %#ok<INUSD>
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

    function susp_frame_table_row_actuated(obj, source, event, row, rowdata)  %#ok<INUSD>
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
      obj.trackingMonitorVisualizer_.resultsReceived(sRes) ;
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
      handles = guidata(obj.mainFigure_) ;
      
      %handles = clearDepHandles(handles);
      obj.clearSatellites() ;
      
      handles = apt.initTblFrames(handles, labeler.maIsMA) ;
      
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
        figs(iView) = figure(...
          'CloseRequestFcn',@(s,e)cbkAuxFigCloseReq(s,e,obj),...
          'Color',figs(1).Color,...
          'Menubar','none',...
          'Toolbar','figure',...
          'UserData',struct('view',iView)...
          );
        axs(iView) = axes;
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
        axsOcc(iView) = axes('Parent',axparent,'Position',axpos,'Units',axunits,...
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
        handles = obj.cropInitImRects(handles);
      catch ME
        fprintf(2,'Crop Mode initialization error: %s\n',ME.message);
      end
      
      if isfield(handles,'allAxHiliteMgr') && ~isempty(handles.allAxHiliteMgr)
        % Explicit deletion not supposed to be nec
        delete(handles.allAxHiliteMgr);
      end
      handles.allAxHiliteMgr = AxesHighlightManager(axs);
      
      axis(handles.axes_occ,[0 labeler.nLabelPoints+1 0 2]);
      
      % The link destruction/recreation may not be necessary
      if isfield(handles,'hLinkPrevCurr') && isvalid(handles.hLinkPrevCurr)
        delete(handles.hLinkPrevCurr);
      end
      viewCfg = labeler.projPrefs.View;
      handles.newProjAxLimsSetInConfig = apt.hlpSetConfigOnViews(viewCfg,handles,...
        viewCfg(1).CenterOnTarget); % lObj.CenterOnTarget is not set yet
      AX_LINKPROPS = {'XLim' 'YLim' 'XDir' 'YDir'};
      handles.hLinkPrevCurr = ...
        linkprop([handles.axes_curr,handles.axes_prev],AX_LINKPROPS);
      
      arrayfun(@(x)colormap(x,gray),figs);
      obj.setGUIFigureNames_() ;
      obj.setMainAxesName_();
      
      arrayfun(@(x)zoom(x,'off'),handles.figs_all); % Cannot set KPF if zoom or pan is on
      arrayfun(@(x)pan(x,'off'),handles.figs_all);
      hTmp = findall(handles.figs_all,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',@(src,evt)cbkKPF(src,evt,labeler));
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
      guidata(obj.mainFigure_, handles) ;
    end  % function

    function menu_file_new_actuated(obj)
      % Create a new project
      lableler = obj.labeler_ ;
      lableler.setStatus('Starting New Project');
      if obj.raiseUnsavedChangesDialogIfNeeded() ,
        cfg = ProjectSetup(obj.mainFigure_);  % launches the project setup window
        if ~isempty(cfg)    
          lableler.setStatus('Configuring New Project') ;
          lableler.initFromConfig(cfg);
          lableler.projNew(cfg.ProjectName);
          lableler.setStatus('Adding Movies') ;
          if ~isempty(controller.movieManagerController_) && isvalid(controller.movieManagerController_) ,
            controller.movieManagerController_.setVisible(true);
          else
            error('LabelerController:menu_file_new_actuated', 'Please create or load a project.') ;
          end
        end  
      end
      labeler.clearStatus();
    end  % function

    function updateMainFigureName(obj)    
      labeler = obj.labeler_ ;
      maxlength = 80;
      if isempty(labeler.projectfile),
        projname = [labeler.projname,' (unsaved)'];
      elseif numel(labeler.projectfile) <= maxlength,
        projname = labeler.projectfile;
      else
        [~,projname,ext] = fileparts(labeler.projectfile);
        projname = [projname,ext];
      end
      obj.mainFigure_.Name = sprintf('APT - Project %s',projname) ;
    end  % function

    function didChangeProjectName(obj)
      labeler = obj.labeler_ ;
      str = sprintf('Project $PROJECTNAME created (unsaved) at %s',datestr(now(),16));
      labeler.setRawStatusStringWhenClear_(str) ;
      obj = labeler.controller_ ;
      obj.updateMainFigureName() ;
    end  % function

    function cbkProjFSInfoChanged(obj)
      labeler = obj.labeler_ ;
      info = labeler.projFSInfo ;
      if ~isempty(info)
        str = sprintf('Project $PROJECTNAME %s at %s',info.action,datestr(info.timestamp,16)) ;
        labeler.setRawStatusStringWhenClear_(str) ;
      end
      obj.updateMainFigureName() ;
    end  % function

    function cbkMovieInvertChanged(obj)
      labeler = obj.labeler_ ;
      handles = guidata(obj.mainFigure_) ;
      figs = handles.figs_all ;
      obj.setGUIFigureNames_() ;
      obj.setMainAxesName_() ;
      movInvert = labeler.movieInvert ;
      viewNames = labeler.viewNames ;
      for i=1:labeler.nview
        name = viewNames{i};
        if isempty(name)
          name = '';
        else
          name = sprintf('View: %s',name);
        end
        if movInvert(i)
          name = [name ' (movie inverted)']; %#ok<AGROW>
        end
        figs(i).Name = name;
      end      
    end  % function

    function setGUIFigureNames_(obj)
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

    function setMainAxesName_(obj)
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

    function menu_file_shortcuts_actuated(obj)
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

    function handles = cropInitImRects(obj, handles)
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

  end  % methods  
end  % classdef
