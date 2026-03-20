classdef UncertainFramesController < handle
% Owns the figure and listbox for displaying uncertain frames.

  properties (Access=private, Transient)  % private by convention
    labelerController_  % parent controller
    labeler_  % Labeler
    model_  % UncertainFramesModel
    figure_  % figure handle
    listbox_  % uicontrol listbox handle
    titleText_  % uicontrol text showing movie name / status
  end

  properties (Dependent, Hidden)
    hasValidFigure  % checks figure_ handle validity
  end

  methods
    function obj = UncertainFramesController(model, labelerController, labeler)
      % Create an UncertainFramesController with its figure and listbox.
      obj.model_ = model ;
      obj.labelerController_ = labelerController ;
      obj.labeler_ = labeler ;

      figurePosition = [200 200 400 500] ;
      obj.figure_ = figure(...
        'Name', 'Uncertain Frames', ...
        'NumberTitle', 'off', ...
        'MenuBar', 'none', ...
        'ToolBar', 'none', ...
        'Position', figurePosition, ...
        'Tag', 'uncertain_frames_figure', ...
        'Visible', 'off', ...
        'CloseRequestFcn', @(src, evt)(obj.hideRequested())) ;

      % titlePosition = [10 figurePosition(4)-30 figurePosition(3)-20 20] ;
      obj.titleText_ = uicontrol(...
        'Parent', obj.figure_, ...
        'Style', 'text', ...
        'String', 'Computing...', ...
        'HorizontalAlignment', 'left', ...
        'Tag', 'uncertain_frames_title') ;

      % listboxPosition = [10 10 figurePosition(3)-20 figurePosition(4)-50] ;
      obj.listbox_ = uicontrol(...
        'Parent', obj.figure_, ...
        'Style', 'listbox', ...
        'String', {}, ...
        'Tag', 'uncertain_frames_listbox', ...
        'Callback', @(src, evt)(labelerController.controlActuated('uncertain_frames_listbox', src, evt))) ;

      % Set up resize behavior
      obj.figure_.SizeChangedFcn = @(src, evt)(obj.resizeFigure()) ;

      % Resize to lay out properly
      obj.resizeFigure() ;
    end  % function

    function result = get.hasValidFigure(obj)
      % Return whether the figure handle is valid.
      result = ~isempty(obj.figure_) && ishghandle(obj.figure_) ;
    end  % function

    function update(obj)
      % Sync the listbox and title to the model state.
      model = obj.model_ ;
      isVisible = model.isVisible ;
      if ~isVisible
        obj.figure_.Visible = 'off' ;
        % No need to update if not visible
        return
      end
      if model.isValid
        strings = model.listboxString ;
        nEntries = numel(strings) ;
        movieName = obj.labeler_.moviename ;
        obj.titleText_.String = sprintf('%s  (%d entries)', movieName, nEntries) ;
        obj.listbox_.String = strings ;
        obj.listbox_.Value = max(1, min(obj.listbox_.Value, nEntries)) ;
        obj.listbox_.Enable = 'on' ;
      else
        obj.titleText_.String = 'No tracking data for current movie' ;
        obj.listbox_.String = {} ;
        obj.listbox_.Value = 1 ;
        obj.listbox_.Enable = 'off' ;
      end
      % Make visible at end to reduced flickering
      obj.figure_.Visible = 'on' ;
    end  % function

    function delete(obj)
      % Delete the figure.
      if obj.hasValidFigure
        delete(obj.figure_) ;
      end
    end  % function
  end  % methods

  methods  
    function hideRequested(obj)
      % Handle figure close request by hiding instead of deleting.
      obj.model_.isVisible = false ;
    end  % function

    function resizeFigure(obj)
      % Adjust child positions when figure is resized.
      if ~obj.hasValidFigure
        return
      end
      figPos = obj.figure_.Position ;
      figWidth = figPos(3) ;
      figHeight = figPos(4) ;
      obj.titleText_.Position = [10 figHeight-30 figWidth-20 20] ;
      obj.listbox_.Position = [10 10 figWidth-20 figHeight-50] ;
    end  % function
  end  % methods
end  % classdef
