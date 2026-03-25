classdef UncertainFramesController < handle
  % Owns the figure and listbox for displaying uncertain frames.

  properties (Access=private, Transient)  % private by convention
    labelerController_  % parent controller
    labeler_  % Labeler
    model_  % UncertainFramesModel
    figure_  % figure handle
    checkbox_  % uicontrol checkbox for confidence-is-lack-thereof
    thresholdLabel_  % uicontrol text label for threshold
    thresholdEdit_  % uicontrol edit box for threshold
    listbox_  % uicontrol listbox handle
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

      obj.checkbox_ = uicontrol(...
        'Parent', obj.figure_, ...
        'Style', 'checkbox', ...
        'String', 'Confidence is lack thereof', ...
        'Value', 0, ...
        'Tag', 'uncertain_frames_confidence_lack_thereof_checkbox') ;

      obj.thresholdLabel_ = uicontrol(...
        'Parent', obj.figure_, ...
        'Style', 'text', ...
        'String', 'Threshold:', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'uncertain_frames_threshold_label') ;

      obj.thresholdEdit_ = uicontrol(...
        'Parent', obj.figure_, ...
        'Style', 'edit', ...
        'String', '1', ...
        'Tag', 'uncertain_frames_threshold_edit') ;

      obj.listbox_ = uicontrol(...
        'Parent', obj.figure_, ...
        'Style', 'listbox', ...
        'String', {}, ...
        'Tag', 'uncertain_frames_listbox') ;

      % Set up callbacks using tags to determine the method name
      visit_children(obj.figure_, @set_standard_callback_if_none_bang, labelerController) ;

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
      obj.checkbox_.Value = model.isConfidenceLackThereof ;
      obj.thresholdEdit_.String = sprintf('%g', model.confidenceThreshold) ;
      if model.isLaden
        strings = model.listboxString ;
        nEntries = numel(strings) ;
        obj.listbox_.String = strings ;
        obj.listbox_.Value = max(1, min(obj.listbox_.Value, nEntries)) ;
        obj.listbox_.Enable = 'on' ;
      else
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

    function uncertain_frames_confidence_lack_thereof_checkbox_actuated_(obj, src)
      % Handle checkbox toggle for confidence-is-lack-thereof.
      obj.model_.isConfidenceLackThereof = logical(src.Value) ;
    end  % function

    function uncertain_frames_threshold_edit_actuated_(obj, src)
      % Handle threshold edit box change.
      newValue = str2double(src.String) ;
      if isfinite(newValue) && newValue >= 0
        obj.model_.confidenceThreshold = newValue ;
      else
        src.String = sprintf('%g', obj.model_.confidenceThreshold) ;
      end
    end  % function

    function resizeFigure(obj)
      % Adjust child positions when figure is resized.
      if ~obj.hasValidFigure
        return
      end
      figPos = obj.figure_.Position ;
      figWidth = figPos(3) ;
      figHeight = figPos(4) ;
      pad = 10 ;
      checkboxHeight = 20 ;
      thresholdRowHeight = 22 ;
      gap = 5 ;
      labelWidth = 70 ;
      editWidth = 80 ;
      editGap = 5 ;

      obj.checkbox_.Position = ...
        [pad, figHeight - pad - checkboxHeight, figWidth - 2*pad, checkboxHeight] ;
      thresholdRowTop = figHeight - pad - checkboxHeight - gap ;
      obj.thresholdLabel_.Position = ...
        [pad, thresholdRowTop - thresholdRowHeight, labelWidth, thresholdRowHeight] ;
      obj.thresholdEdit_.Position = ...
        [pad + labelWidth + editGap, thresholdRowTop - thresholdRowHeight, editWidth, thresholdRowHeight] ;
      listboxTop = thresholdRowTop - thresholdRowHeight - gap ;
      obj.listbox_.Position = [pad, pad, figWidth - 2*pad, listboxTop - pad] ;
    end  % function
  end  % methods
end  % classdef
