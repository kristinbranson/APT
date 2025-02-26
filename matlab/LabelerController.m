classdef LabelerController < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    mainFigure_  % the GH to the main figure
    listeners_
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
      labeler = Labeler('isgui', true, 'isInDebugMode', isInDebugMode,  'isInAwsDebugMode', isInAwsDebugMode) ;  
        % Create the labeler, tell it there will be a GUI attached
      obj.labeler_ = labeler ;
      obj.mainFigure_ = LabelerGUI(labeler, obj) ;
      obj.labeler_.registerController(obj) ;  % hack
      obj.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      obj.isInYodaMode_ = isInYodaMode ;  
        % If in yoda mode, we don't wrap GUI-event function calls in a try..catch.
        % Useful for debugging.
      obj.listeners_ = cell(1,0) ;
      obj.listeners_{end+1} = ...
        addlistener(labeler, 'updateDoesNeedSave', @(source,event)(obj.updateDoesNeedSave(source, event))) ;      
      obj.listeners_{end+1} = ...
        addlistener(labeler, 'updateStatus', @(source,event)(obj.updateStatus(source, event))) ;      
      obj.listeners_{end+1} = ...
        addlistener(labeler, 'didSetTrx', @(source,event)(obj.didSetTrx(source, event))) ;      
      obj.listeners_{end+1} = ...
        addlistener(labeler, 'updateTrxSetShowTrue', @(source,event)(obj.updateTrxSetShowTrue(source, event))) ;      
      obj.listeners_{end+1} = ...
        addlistener(labeler, 'updateTrxSetShowFalse', @(source,event)(obj.updateTrxSetShowFalse(source, event))) ;      
      % Do this once listeners are set up
      obj.labeler_.handleCreationTimeAdditionalArguments_(varargin{:}) ;
    end

    function delete(obj)
      % Having the figure without a controller would be bad, so we make sure to
      % delete the figure (and subfigures) in our destructor.
      % We also delete the model.
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
      tm = obj.get_track_mode_();
      obj.labeler_.track(tm, varargin{:}) ;
    end

    function mftset = get_track_mode_(obj)
      % This is designed to do the same thing as LabelerGUI::getTrackMode().
      % The two methods should likely be consolidated at some point.  Private by
      % convention
      pumTrack = findobj(obj.mainFigure_, 'Tag', 'pumTrack') ;
      idx = pumTrack.Value ;
      % Note, .TrackingMenuNoTrx==.TrackingMenuTrx(1:K), so we can just index
      % .TrackingMenuTrx.
      mfts = MFTSetEnum.TrackingMenuTrx;
      mftset = mfts(idx);      
    end

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
          sfn = LabelerGUI('get_local_fn','menu_file_saveas_Callback');
          sfn(source, event, guidata(source))
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
      
      %labeler.trackSetAutoParams();
      
      fprintf('Training started at %s...\n',datestr(now));
      oc1 = onCleanup(@()(labeler.clearStatus()));
      wbObj = WaitBarWithCancel('Training');
      oc2 = onCleanup(@()delete(wbObj));
      centerOnParentFigure(wbObj.hWB,obj.mainFigure_);
      labeler.trackRetrain(...
        'retrainArgs',{'wbObj',wbObj}, ...
        'do_just_generate_db', do_just_generate_db, ...
        'do_call_apt_interface_dot_py', do_call_apt_interface_dot_py) ;
      if wbObj.isCancel
        msg = wbObj.cancelMessage('Training canceled');
        msgbox(msg,'Train');
      end
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
  end  % public methods block

  methods  % private by convention methods block
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
      ec2 = backend.awsec2 ;
      if ~ec2.isConfigured || canConfigure >= 2,
        if canConfigure,
          [tfsucc,keyName,pemFile] = ...
            specifySSHKeyUIStc(ec2.keyName,ec2.pem);
          if tfsucc ,
            % For changing things in the model, we go through the top-level model object
            labeler.setAwsPemFileAndKeyName(pemFile, keyName) ;
          end
          if ~tfsucc && ~ec2.isConfigured,
            reason = 'AWS EC2 instance is not configured.';
            error(reason) ;
          end
        else
          reason = 'AWS EC2 instance is not configured.';
          error(reason) ;
        end
      end
      if forceSelect || ~ec2.isSpecified,
        if ec2.isSpecified ,
          instanceID = ec2.instanceID;
        else
          instanceID = '';
        end
        if canLaunch,
          qstr = 'Launch a new instance or attach to an existing instance?';
          if ~ec2.isSpecified,
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
              tf = ec2.launchInstance();
              if ~tf
                reason = 'Could not launch AWS EC2 instance.';
                error(reason) ;
              end
              break
            case 'Attach to Existing',
              [tfsucc,instanceIDs,instanceTypes] = ec2.listInstances();
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
              if ~isempty(ec2.instanceID),
                v = find(strcmp(instanceIDs,ec2.instanceID),1);
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
        labeler.setAwsInstanceId(instanceID, instanceType) ;
        %ec2.setInstanceID(instanceID,instanceType);
      end
    end  % function selectAwsInstance_()
  end  % private by-convention methods block

  methods
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
    
  end  % public methods block
  
end
