classdef UncertainFramesController < handle
  % Owns the figure and listbox for displaying uncertain frames.

  properties (Access=private, Transient)  % private by convention
    labelerController_  % parent controller
    labeler_  % Labeler
    model_  % UncertainFramesModel
    figure_  % uifigure handle
    thresholdLabel_  % uilabel for threshold
    thresholdEdit_  % uieditfield for threshold
    thresholdHint_  % uilabel showing min/max when listbox is empty
    listbox_  % uilistbox handle
  end

  properties (Dependent, Access=private)
    hasValidFigure_  % checks figure_ handle validity
  end

  methods
    function obj = UncertainFramesController(model, labelerController, labeler)
      % Create an UncertainFramesController.  The figure is not created
      % until it is first made visible.
      obj.model_ = model ;
      obj.labelerController_ = labelerController ;
      obj.labeler_ = labeler ;
    end  % function

    function result = get.hasValidFigure_(obj)
      % Return whether the figure handle is valid.
      result = ~isempty(obj.figure_) && isvalid(obj.figure_) ;
    end  % function

    function update(obj)
      % Sync the listbox and title to the model state.
      model = obj.model_ ;
      isVisible = model.isVisible ;
      if ~isVisible
        if obj.hasValidFigure_
          obj.figure_.Visible = 'off' ;
        end
        % No need to update if not visible
        return
      end
      if ~obj.hasValidFigure_
        obj.createFigure_() ;
      end
      wasFigureVisible = strcmp(obj.figure_.Visible, 'on') ;
      obj.thresholdEdit_.Value = sprintf('%g', model.quantileConfidenceThreshold) ;
      absoluteThreshold = model.absoluteConfidenceThreshold ;
      if isfinite(absoluteThreshold)
        obj.thresholdHint_.Text = sprintf('(%g)', absoluteThreshold) ;
        obj.thresholdHint_.Visible = 'on' ;
      else
        obj.thresholdHint_.Visible = 'off' ;
      end
      if model.isLaden
        strings = model.displayStringFromBoutIndex ;
        entryCount = numel(strings) ;
        obj.listbox_.Items = strings ;
        % uilistbox has no ValueIndex property on this MATLAB version; derive
        %/set the index by matching against Items instead.
        if isempty(obj.listbox_.Value)
          previousIndex = 1 ;
        else
          previousIndex = find(strcmp(obj.listbox_.Items, obj.listbox_.Value), 1) ;
          if isempty(previousIndex)
            previousIndex = 1 ;
          end
        end
        obj.listbox_.Value = obj.listbox_.Items{max(1, min(previousIndex, entryCount))} ;
        obj.listbox_.Enable = 'on' ;
      else
        obj.listbox_.Items = {} ;
        obj.listbox_.Enable = 'off' ;
      end
      % Make visible at end to reduce flickering
      obj.figure_.Visible = 'on' ;
      if ~wasFigureVisible
        % Block until the uifigure is laid out so the caller's busy
        % cursor stays up until the window is ready for input.
        waitForFigureToSync(obj.figure_) ;
      end
    end  % function

    function delete(obj)
      % Delete the figure.
      if obj.hasValidFigure_
        delete(obj.figure_) ;
      end
    end  % function
  end  % methods

  methods
    function hideRequested(obj)
      % Handle figure close request by hiding instead of deleting.
      obj.model_.isVisible = false ;
    end  % function

    function uncertain_frames_threshold_edit_actuated_(obj, src)
      % Handle threshold edit box change.
      newValue = str2double(src.Value) ;
      obj.model_.quantileConfidenceThreshold = newValue ;
    end  % function
  end  % methods

  methods (Access=private)
    function createFigure_(obj)
      % Create the uifigure and all child controls.
      figurePosition = [200 200 400 500] ;
      obj.figure_ = uifigure(...
        'Name', 'Uncertain Frames', ...
        'Position', figurePosition, ...
        'Tag', 'uncertain_frames_figure', ...
        'Visible', 'off', ...
        'CloseRequestFcn', @(src, evt)(obj.hideRequested())) ;

      labelerController = obj.labelerController_ ;

      gridLayout = uigridlayout(obj.figure_, [2, 1]) ;
      gridLayout.RowHeight = {22, '1x'} ;
      gridLayout.ColumnWidth = {'1x'} ;

      thresholdRow = uigridlayout(gridLayout, [1, 3]) ;
      thresholdRow.RowHeight = {'1x'} ;
      thresholdRow.ColumnWidth = {64, 80, '1x'} ;
      thresholdRow.Padding = [0, 0, 0, 0] ;

      obj.thresholdLabel_ = uilabel(thresholdRow, ...
        'Text', 'Threshold:', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'uncertain_frames_threshold_label') ;

      obj.thresholdEdit_ = uieditfield(thresholdRow, ...
        'text', ...
        'Value', '1', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'uncertain_frames_threshold_edit', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('uncertain_frames_threshold_edit', src, evt))) ;

      obj.thresholdHint_ = uilabel(thresholdRow, ...
        'Text', '', ...
        'FontAngle', 'italic', ...
        'HorizontalAlignment', 'left', ...
        'Visible', 'off', ...
        'Tag', 'uncertain_frames_threshold_hint') ;

      obj.listbox_ = uilistbox(gridLayout, ...
        'Items', {}, ...
        'Tag', 'uncertain_frames_listbox', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('uncertain_frames_listbox', src, evt))) ;

      % Center on the main APT figure
      mainFigurePosition = obj.labelerController_.mainFigurePixelPosition() ;
      centerOnOtherFigureGivenPositionBang(obj.figure_, mainFigurePosition) ;
    end  % function
  end  % methods
end  % classdef
