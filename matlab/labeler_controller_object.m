classdef labeler_controller_object < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    main_figure_  % the GH to the main figure
    listeners_
  end
  properties  % private/protected by convention
    tvTrx_  % scalar TrackingVisualizerTrx
  end

  methods
    function self = labeler_controller_object(varargin)
      labeler = Labeler('isgui', true) ;  % Create the labeler, tell it  there's a GUI attached
      self.labeler_ = labeler ;
      self.main_figure_ = LabelerGUI(labeler, self) ;
      self.labeler_.register_controller_(self) ;  % hack
      self.tvTrx_ = TrackingVisualizerTrx(labeler) ;
      self.labeler_.handle_creation_time_additional_arguments(varargin{:}) ;
      self.listeners_ = cell(1,0) ;
      self.listeners_{end+1} = ...
        addlistener(labeler, 'update_does_need_save', @(source,event)(self.update_does_need_save(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'update_status', @(source,event)(self.update_status(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'did_set_trx', @(source,event)(self.did_set_trx(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'update_trx_set_show_true', @(source,event)(self.update_trx_set_show_true(source, event))) ;      
      self.listeners_{end+1} = ...
        addlistener(labeler, 'update_trx_set_show_false', @(source,event)(self.update_trx_set_show_false(source, event))) ;      
    end

    function delete(self)
      % Having the figure without a controller would be bad, so we make sure to
      % delete the figure (and subfigures) in our destructor.
      % We also delete the model.
      main_figure = self.main_figure_ ;
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
    
    function update_does_need_save(self, ~, ~)      
      labeler = self.labeler_ ;
      does_need_save = labeler.does_need_save ;
      handles = guidata(self.main_figure_) ;
      hTx = handles.txUnsavedChanges ;
      if does_need_save
        set(hTx,'Visible','on');
      else
        set(hTx,'Visible','off');
      end
    end

    function update_status(self, ~, ~)
      % Update the status text box to reflect the current model state.
      labeler = self.labeler_ ;
      handles = guidata(self.main_figure_) ;
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
      self.update_status_text_() ;
      drawnow('limitrate', 'nocallbacks');
    end

    function update_status_text_(self)
      % Actually update the String in the status text box.  Use the shorter status
      % string from the labeler if the normal one is too long for the text box.
      labeler = self.labeler_ ;
      handles = guidata(self.main_figure_) ;      
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
    end

    function did_set_trx(self, ~, ~)
      trx = self.labeler_.trx ;
      self.tvTrx_.init(true, numel(trx)) ;
    end

    function quit_requested(self)
      is_ok_to_quit = self.raise_unsaved_changes_dialog_if_needed() ;
      if is_ok_to_quit ,
        delete(self) ;
      end      
    end

    function is_ok_to_proceed = raise_unsaved_changes_dialog_if_needed(self)
      labeler = self.labeler_ ;
      
      if ~verLessThan('matlab','9.6') && batchStartupOptionUsed
        return
      end

      OPTION_SAVE = 'Save first';
      OPTION_PROC = 'Proceed without saving';
      OPTION_CANC = 'Cancel';
      if labeler.does_need_save ,
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

    function update_trx_set_show_true(self, ~, ~)
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
    
    function update_trx_set_show_false(self, ~, ~)
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
