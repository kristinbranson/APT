classdef CompareTrackersController < handle
  % Owns the figure, dropdowns, threshold edit, and listbox for the
  % Compare Trackers... window.

  properties (Access=private, Transient)  % private by convention
    labelerController_  % parent controller
    labeler_  % Labeler
    model_  % CompareTrackersModel
    figure_  % uifigure handle
    referenceDropdown_  % uidropdown for the reference tracker
    testDropdown_  % uidropdown for the test tracker
    thresholdLabel_  % uilabel for threshold
    thresholdEdit_  % uieditfield for threshold
    thresholdHint_  % uilabel showing the absolute distance threshold
    listbox_  % uilistbox handle
  end

  properties (Dependent, Access=private)
    hasValidFigure_
  end

  methods
    function obj = CompareTrackersController(model, labelerController, labeler)
      % Create a CompareTrackersController.  The figure is not created
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
      % Sync the figure and its widgets to the model state.
      model = obj.model_ ;
      isVisible = model.isVisible ;
      if ~isVisible
        if obj.hasValidFigure_
          obj.figure_.Visible = 'off' ;
        end
        return
      end
      if ~obj.hasValidFigure_
        obj.createFigure_() ;
      end
      wasFigureVisible = strcmp(obj.figure_.Visible, 'on') ;

      % Refresh dropdown items and selection from the current
      % trackerHistory, in case trackers were added since the last
      % update.  The reference tracker is always the current tracker
      % (trackerHistory index 1), so its dropdown is always disabled and
      % just displays that tracker.
      [items, itemsData] = obj.trackerDropdownItems_() ;
      obj.referenceDropdown_.Items = items ;
      obj.referenceDropdown_.ItemsData = itemsData ;
      if ~isempty(itemsData)
        obj.referenceDropdown_.Value = clampDropdownValue_(model.referenceTrackerHistoryIndex, itemsData) ;
      end
      obj.referenceDropdown_.Enable = 'off' ;

      % The test dropdown lists every tracker except the current
      % (reference) tracker, unless the current tracker is itself
      % selected as the test tracker -- in which case include it so the
      % selection is displayable (and flagged pink).
      referenceTrackerHistoryIndex = model.referenceTrackerHistoryIndex ;
      isCurrentTrackerSelectedAsTest = isequal(model.testTrackerHistoryIndex, referenceTrackerHistoryIndex) ;
      if isCurrentTrackerSelectedAsTest
        testItems = items ;
        testItemsData = itemsData ;
      else
        areTestCandidates = ~cellfun(@(d)(isequal(d, referenceTrackerHistoryIndex)), itemsData) ;
        testItems = items(areTestCandidates) ;
        testItemsData = itemsData(areTestCandidates) ;
      end
      if isempty(testItemsData)
        obj.testDropdown_.Items = {''} ;
        obj.testDropdown_.ItemsData = {} ;
        obj.testDropdown_.Enable = 'off' ;
      else
        obj.testDropdown_.Items = testItems ;
        obj.testDropdown_.ItemsData = testItemsData ;
        obj.testDropdown_.Enable = 'on' ;
        obj.testDropdown_.Value = clampDropdownValue_(model.testTrackerHistoryIndex, testItemsData) ;
      end

      % Indicate the reference (== current) tracker in the test dropdown,
      % since selecting it as the test tracker yields no comparison.  In
      % R2023a and later, per-item dropdown styling is available, so pink
      % the reference item when it is present in the list.  In older
      % releases, fall back to flagging the whole control pink when the
      % test selection coincides with the reference.
      if verLessThan('matlab', '9.14')  % R2023a
        obj.testDropdown_.BackgroundColor = ...
          fif(model.isTestTrackerChoiceValid, [1 1 1], [1 0.8 0.85]) ;
      else
        removeStyle(obj.testDropdown_) ;
        referenceItemPosition = ...
          find(cellfun(@(d)(isequal(d, referenceTrackerHistoryIndex)), testItemsData), 1) ;
        if ~isempty(referenceItemPosition)
          pinkStyle = uistyle('BackgroundColor', [1 0.8 0.85]) ;
          addStyle(obj.testDropdown_, pinkStyle, 'item', referenceItemPosition) ;
        end
      end

      obj.thresholdEdit_.Value = sprintf('%g', model.quantileThreshold) ;
      absoluteThreshold = model.absoluteDistanceThreshold ;
      if isfinite(absoluteThreshold)
        obj.thresholdHint_.Text = sprintf('(%.2f px)', absoluteThreshold) ;
        obj.thresholdHint_.Visible = 'on' ;
      else
        obj.thresholdHint_.Visible = 'off' ;
      end

      if model.isLaden
        strings = model.displayStringFromBoutIndex ;
        entryCount = numel(strings) ;
        obj.listbox_.Items = strings ;
        previousIndex = obj.listbox_.ValueIndex ;
        if isempty(previousIndex)
          previousIndex = 1 ;
        end
        obj.listbox_.ValueIndex = max(1, min(previousIndex, entryCount)) ;
        obj.listbox_.FontAngle = 'normal' ;
        obj.listbox_.Enable = 'on' ;
      elseif ~model.isTestTrackerChoiceValid
        obj.listbox_.Items = {'(Test tracker should differ from reference.)'} ;
        obj.listbox_.FontAngle = 'italic' ;
        obj.listbox_.Enable = 'off' ;
      else
        obj.listbox_.Items = {} ;
        obj.listbox_.FontAngle = 'normal' ;
        obj.listbox_.Enable = 'off' ;
      end

      obj.figure_.Visible = 'on' ;
      if ~wasFigureVisible
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

    function compare_trackers_threshold_edit_actuated_(obj, src)
      % Handle threshold edit box change.
      newValue = str2double(src.Value) ;
      obj.model_.quantileThreshold = newValue ;
    end  % function

    function compare_trackers_test_dropdown_actuated_(obj, src)
      % Handle test-tracker dropdown change.
      obj.model_.testTrackerHistoryIndex = src.Value ;
    end  % function
  end  % methods

  methods (Access=private)
    function createFigure_(obj)
      % Create the uifigure and all child controls.
      figurePosition = [200 200 480 520] ;
      obj.figure_ = uifigure(...
        'Name', 'Compare Trackers', ...
        'Position', figurePosition, ...
        'Tag', 'compare_trackers_figure', ...
        'Visible', 'off', ...
        'CloseRequestFcn', @(src, evt)(obj.hideRequested())) ;

      labelerController = obj.labelerController_ ;

      gridLayout = uigridlayout(obj.figure_, [4, 1]) ;
      gridLayout.RowHeight = {22, 22, 22, '1x'} ;
      gridLayout.ColumnWidth = {'1x'} ;

      % Row 1: reference tracker
      referenceRow = uigridlayout(gridLayout, [1, 2]) ;
      referenceRow.RowHeight = {'1x'} ;
      referenceRow.ColumnWidth = {80, '1x'} ;
      referenceRow.Padding = [0, 0, 0, 0] ;
      uilabel(referenceRow, ...
        'Text', 'Reference:', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'compare_trackers_reference_label') ;
      % The reference tracker is always the current tracker, so this
      % dropdown is disabled and has no value-changed callback.
      obj.referenceDropdown_ = uidropdown(referenceRow, ...
        'Items', {''}, ...
        'Enable', 'off', ...
        'Tag', 'compare_trackers_reference_dropdown') ;

      % Row 2: test tracker
      testRow = uigridlayout(gridLayout, [1, 2]) ;
      testRow.RowHeight = {'1x'} ;
      testRow.ColumnWidth = {80, '1x'} ;
      testRow.Padding = [0, 0, 0, 0] ;
      uilabel(testRow, ...
        'Text', 'Test:', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'compare_trackers_test_label') ;
      obj.testDropdown_ = uidropdown(testRow, ...
        'Items', {''}, ...
        'Tag', 'compare_trackers_test_dropdown', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('compare_trackers_test_dropdown', src, evt))) ;

      % Row 3: threshold
      thresholdRow = uigridlayout(gridLayout, [1, 3]) ;
      thresholdRow.RowHeight = {'1x'} ;
      thresholdRow.ColumnWidth = {80, 80, '1x'} ;
      thresholdRow.Padding = [0, 0, 0, 0] ;
      obj.thresholdLabel_ = uilabel(thresholdRow, ...
        'Text', 'Threshold:', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'compare_trackers_threshold_label') ;
      obj.thresholdEdit_ = uieditfield(thresholdRow, ...
        'text', ...
        'Value', '0.99', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'compare_trackers_threshold_edit', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('compare_trackers_threshold_edit', src, evt))) ;
      obj.thresholdHint_ = uilabel(thresholdRow, ...
        'Text', '', ...
        'FontAngle', 'italic', ...
        'HorizontalAlignment', 'left', ...
        'Visible', 'off', ...
        'Tag', 'compare_trackers_threshold_hint') ;

      % Row 4: listbox
      obj.listbox_ = uilistbox(gridLayout, ...
        'Items', {}, ...
        'Tag', 'compare_trackers_listbox', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('compare_trackers_listbox', src, evt))) ;

      mainFigurePosition = obj.labelerController_.mainFigurePixelPosition() ;
      centerOnOtherFigureGivenPositionBang(obj.figure_, mainFigurePosition) ;
    end  % function

    function [items, itemsData] = trackerDropdownItems_(obj)
      % Build the dropdown Items / ItemsData arrays from the labeler's
      % current trackerHistory.  Returns parallel cell arrays.
      trackers = obj.labeler_.trackerHistory ;
      trackerCount = numel(trackers) ;
      items = cell(trackerCount, 1) ;
      itemsData = cell(trackerCount, 1) ;
      for i = 1 : trackerCount
        tracker = trackers{i} ;
        algNamePretty = tracker.algorithmNamePretty ;
        rawTrnNameLbl = tracker.trnNameLbl ;
        trnNameLbl = fif(isempty(rawTrnNameLbl), 'untrained', rawTrnNameLbl) ;
        userTag = tracker.userTag ;
        if isempty(userTag)
          items{i} = sprintf('%s (%s)', algNamePretty, trnNameLbl) ;
        else
          items{i} = sprintf('%s (%s, %s)', algNamePretty, userTag, trnNameLbl) ;
        end
        itemsData{i} = i ;
      end
    end  % function
  end  % methods
end  % classdef



function clampedValue = clampDropdownValue_(desiredValue, itemsData)
% Return desiredValue if it is one of the items in itemsData; otherwise
% return the first item.  Used to keep dropdown selection valid when the
% trackerHistory shrinks or the model's selection drifts past the end.
isPresent = any(cellfun(@(d)(isequal(d, desiredValue)), itemsData)) ;
if isPresent
  clampedValue = desiredValue ;
else
  clampedValue = itemsData{1} ;
end
end  % function
