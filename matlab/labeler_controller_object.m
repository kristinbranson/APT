classdef labeler_controller_object < handle
  properties  % private/protected by convention
    labeler_  % the controlled Labeler object
    main_figure_  % the GH to the main figure
    listeners_
  end

  methods
    function self = labeler_controller_object(varargin)
      labeler = Labeler('isgui', true) ;  % Create the labeler, tell it  there's a GUI attached
      self.labeler_ = labeler ;
      self.main_figure_ = LabelerGUI(labeler, self) ;
      self.labeler_.register_figure(self.main_figure_) ;  % hack
      self.labeler_.handle_creation_time_additional_arguments(varargin{:}) ;
      self.listeners_ = cell(1,0) ;
      self.listeners_{end+1} = ...
        addlistener(labeler, 'update_does_need_save', @(source,event)(self.update_does_need_save(source, event))) ;      
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
      if does_need_save,
        info = labeler.projFSInfo ;
        why = labeler.why_does_need_save ;
        if isempty(info) ,
          display_string = sprintf('%s since $PROJECTNAME saved.', why) ;
        else
          display_string = sprintf('%s since $PROJECTNAME %s at %s', why, info.action,datestr(info.timestamp,16)) ;
        end
        is_busy = false ;
        SetStatus = LabelerGUI('get_local_fn', 'SetStatus') ;
        feval(SetStatus, handles, display_string, is_busy) ;  
          % This causes the status sting in the figure to be set, and the cursor to be
          % changed to reflect the busy status.
        % Really think display_string and is_busy belong in the model, 
        % but we'll leave that for another day.  -- ALT, 2023-05-08
      end
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
  end
end
