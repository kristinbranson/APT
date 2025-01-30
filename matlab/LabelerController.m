classdef LabelerController < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    mainFigure_  % the GH to the main figure
    listeners_
    satellites_ = gobjects(1,0)  % handles of dialogs, figures, etc that will get deleted when this object is deleted
    waitbarFigure_ = gobjects(1,0)  % a GH to a waitbar() figure, or empty
    %waitbarListeners_ = event.listener.empty(1,0)
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
      deleteValidHandles(obj.satellites_) ;
      deleteValidHandles(obj.waitbarFigure_) ;
      main_figure = obj.mainFigure_ ;
      if ~isempty(main_figure) && isvalid(main_figure)
        handles = guidata(main_figure) ;
        deleteValidHandles(handles.depHandles);
        handles.depHandles = [] ;
        if isfield(handles,'movieMgr') && ~isempty(handles.movieMgr) && isvalid(handles.movieMgr) ,
          delete(handles.movieMgr);
        end        
        handles.movieMgr = [] ;
        deleteValidHandles(main_figure) ;
        delete(obj.labeler_) ;  % We don't want the model to hang around
      end
    end
    
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
          set(handles.figure,'Pointer','watch');
        end
      else
        color = handles.idlestatuscolor;
        if isfield(handles,'figs_all') && any(ishandle(handles.figs_all)),
          set(handles.figs_all(ishandle(handles.figs_all)),'Pointer','arrow');
        else
          set(handles.figure,'Pointer','arrow');
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
      obj.satellites_(1,end+1) = h ;  % register dialog to we can delete when main window closes
    end

    function showGTResults(obj, source, event)  %#ok<INUSD> 
      % Event handler that gets called after the labeler finishes computing GT results.
      % Raises a dialog, and registers it as a 'satellite' window so we can delete
      % it when the main window closes.
      obj.createGTResultFigures_() ;
      h = msgbox('GT results available in Labeler property ''gtTblRes''.');
      obj.satellites_(1,end+1) = h ;  % register dialog to we can delete when main window closes
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
      obj.satellites_(1,end+1) = fig_1 ;
      plotPercentileHist(allims,prcs,allpos,prc_vals,fig_1,txtOffset)

      % Err by landmark
      fig_2 = figure('Name','GT err by landmark');
      obj.satellites_(1,end+1) = fig_2 ;
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
      obj.satellites_(1,end+1) = fig_3 ;
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
        % this means the control actuated was a 'faux' control
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
  end  % methods
end  % classdef
