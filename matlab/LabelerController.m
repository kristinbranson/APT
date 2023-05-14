classdef LabelerController < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    mainFigure_  % the GH to the main figure
    listeners_
  end
  properties  % private/protected by convention
    tvTrx_  % scalar TrackingVisualizerTrx
  end

  methods
    function self = LabelerController(varargin)
      labeler = Labeler('isgui', true) ;  % Create the labeler, tell it  there's a GUI attached
      self.labeler_ = labeler ;
      self.mainFigure_ = LabelerGUI(labeler, self) ;
      self.labeler_.register_controller_(self) ;  % hack
      self.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      self.labeler_.handle_creation_time_additional_arguments(varargin{:}) ;
      self.listeners_ = cell(1,0) ;
      self.listeners_{end+1} = ...
        addlistener(labeler, 'updateDoesNeedSave', @(source,event)(self.updateDoesNeedSave(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'updateStatus', @(source,event)(self.updateStatus(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'didSetTrx', @(source,event)(self.didSetTrx(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'updateTrxSetShowTrue', @(source,event)(self.updateTrxSetShowTrue(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'updateTrxSetShowFalse', @(source,event)(self.updateTrxSetShowFalse(source, event))) ;      
    end

    function delete(self)
      % Having the figure without a controller would be bad, so we make sure to
      % delete the figure (and subfigures) in our destructor.
      % We also delete the model.
      main_figure = self.mainFigure_ ;
      if ~isempty(main_figure) && isvalid(main_figure)
        handles = guidata(main_figure) ;
        deleteValidHandles(handles.depHandles);
        handles.depHandles = [] ;
        if isfield(handles,'movieMgr') && ~isempty(handles.movieMgr) && isvalid(handles.movieMgr) ,
          delete(handles.movieMgr);
        end        
        handles.movieMgr = [] ;
        deleteValidHandles(main_figure) ;
        delete(self.labeler_) ;  % We don't want the model to hang around
      end
    end
    
    function updateDoesNeedSave(self, ~, ~)      
      labeler = self.labeler_ ;
      doesNeedSave = labeler.doesNeedSave ;
      handles = guidata(self.mainFigure_) ;
      hTx = handles.txUnsavedChanges ;
      if doesNeedSave
        set(hTx,'Visible','on');
      else
        set(hTx,'Visible','off');
      end
    end

    function updateStatus(self, ~, ~)
      % Update the status text box to reflect the current model state.
      labeler = self.labeler_ ;
      handles = guidata(self.mainFigure_) ;
      is_busy = labeler.is_status_busy ;
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
      raw_status_string = labeler.raw_status_string;
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

    function didSetTrx(self, ~, ~)
      trx = self.labeler_.trx ;
      self.tvTrx_.init(true, numel(trx)) ;
    end

    function quitRequested(self)
      is_ok_to_quit = self.raiseUnsavedChangesDialogIfNeeded() ;
      if is_ok_to_quit ,
        delete(self) ;
      end      
    end

    function is_ok_to_proceed = raiseUnsavedChangesDialogIfNeeded(self)
      labeler = self.labeler_ ;
      
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

    function updateTrxSetShowTrue(self, ~, ~)
      % Update .hTrx, .hTraj based on .trx, .showTrx*, .currFrame
      labeler = self.labeler_ ;
      if ~labeler.hasTrx,
        return
      end           
      tfShow = labeler.which_trx_are_showing() ;      
      tv = self.tvTrx_ ;
      tv.setShow(tfShow);
      tv.updateTrx(tfShow);
    end
    
    function updateTrxSetShowFalse(self, ~, ~)
      % Update .hTrx, .hTraj based on .trx, .showTrx*, .currFrame
      labeler = self.labeler_ ;
      if ~labeler.hasTrx,
        return
      end            
      tfShow = labeler.which_trx_are_showing() ;      
      tv = self.tvTrx_ ;
      tv.updateTrx(tfShow);
    end
    
  end
end
