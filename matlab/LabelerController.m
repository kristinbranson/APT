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
      [isInDebugMode, isInYodaMode] = ...
        myparse_nocheck(varargin, ...
                        'isInDebugMode',false, ...
                        'isInYodaMode', false) ;
      labeler = Labeler('isgui', true, 'isInDebugMode', isInDebugMode) ;  % Create the labeler, tell it there will be a GUI attached
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

    function result = doProjectAndMovieExist_(obj)
      % Returns true iff a project exists and a movie is open.
      % If no project exists, returns false.
      % If a project exists but no movie is open, throws up a dialog box indictating
      % this, then returns false.

      labeler = obj.labeler_ ;
      if labeler.hasProject ,
        if labeler.hasMovie ,
          result = true ;
        else
          msgbox('There is no movie open.');
          result = false ;
        end
      else
        result = false ;
      end
    end

    function pbTrack_actuated(obj, source, event)
      obj.track_core_(source, event) ;
    end

    function menu_start_tracking_but_dont_call_apt_interface_dot_py_actuated(obj, source, event)
      obj.track_core_(source, event, 'do_call_apt_interface_dot_py', false) ;
    end
    
    function track_core_(obj, source, event, varargin)  %#ok<INUSD> 
      % Process keyword args
      [do_call_apt_interface_dot_py] = ...
        myparse(varargin, ...
                'do_call_apt_interface_dot_py', true) ;
      
      if ~obj.doProjectAndMovieExist_() ,
        return
      end
      labeler = obj.labeler_ ;
      labeler.setStatus('Tracking...');
      tm = obj.get_track_mode_();
      tblMFT = tm.getMFTable(labeler,'istrack',true);
      if isempty(tblMFT),
        msgbox('All frames tracked.','Track');
        labeler.clearStatus() ;
        return;
      end
      [tfCanTrack,reason] = labeler.trackCanTrack(tblMFT);
      if ~tfCanTrack,
        errordlg(['Error tracking: ',reason],'Error tracking');
        labeler.clearStatus();
        return;
      end
      fprintf('Tracking started at %s...\n',datestr(now()));
      wbObj = WaitBarWithCancel('Tracking');
      centerOnParentFigure(wbObj.hWB,obj.mainFigure_);
      oc = onCleanup(@()delete(wbObj));
      if labeler.maIsMA
        labeler.track(tblMFT,'wbObj',wbObj,'track_type','detect', 'do_call_apt_interface_dot_py', do_call_apt_interface_dot_py);
      else
        labeler.track(tblMFT,'wbObj',wbObj);
      end
      if wbObj.isCancel
        msg = wbObj.cancelMessage('Tracking canceled');
        msgbox(msg,'Track');
      end
      labeler.clearStatus();
    end

    function mftset = get_track_mode_(obj)
      % This is designed to do the same thing as LabelerGUI::getTrackMode().
      % The two methods should likely be consolidated at some point.  Private by
      % convention

      %handles = guidata(obj.mainFigure_) ;      
      %idx = handles.pumTrack.Value;
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
      
      if ~obj.doProjectAndMovieExist_() ,
        return
      end
      labeler = obj.labeler_ ;
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

    function menu_quit_but_dont_delete_temp_folder_actuated(obj, source, event)
      obj.labeler_.projTempDirDontClearOnDestructor = true ;
      obj.quitRequested() ;
    end  % method    
  end  % public methods block

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
          if isequal(exception.identifier,'APT:invalidPropertyValue') ,
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
