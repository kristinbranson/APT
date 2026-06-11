classdef CompareTrackersController < handle
  % Owns the figure, dropdowns, threshold edit, and listbox for the
  % Compare Trackers... window.

  properties (Access=private, Transient)  % private by convention
    labelerController_  % parent controller
    labeler_  % Labeler
    model_  % CompareTrackersModel
    figure_  % uifigure handle
    modeDropdown_  % uidropdown for the mode (CompareTrackersMode)
    referenceDropdown_  % uidropdown for the reference tracker
    testDropdown_  % uidropdown for the test tracker
    thresholdLabel_  % uilabel for threshold
    thresholdEdit_  % uieditfield for threshold
    thresholdHint_  % uilabel showing the absolute distance threshold
    listbox_  % uilistbox handle
    gridLayout_  % the figure's top-level uigridlayout
    previewAxes_  % uiaxes showing the selected bout's max-distance frame
    previewImage_  % image object in previewAxes_
    connectorLineGroup_
      % hggroup holding the ref-to-test connector lines, created between
      % the image and the scatters so the lines render below the markers
    connectorLines_ = gobjects(0, 1)
      % [landmarkCount x 1] line objects, each connecting a ref landmark
      % to the corresponding test landmark
    refScatter_  % scatter object for the reference tracker's pose
    testScatter_  % scatter object for the test tracker's pose
    unmatchedCentroidScatter_
      % scatter object marking the centroids of the unmatched reference
      % tracks (UnmatchedAnimalCount mode only)
    previewPlaceholderText_
      % text object shown centered in the preview axes when no bout is
      % selected
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

      obj.modeDropdown_.Value = model.mode ;

      % Refresh dropdown items and selection from the current
      % trackerHistory, in case trackers were added since the last
      % update.  The reference tracker is always the current tracker
      % (trackerHistory index 1), so its dropdown is always disabled and
      % just displays that tracker.
      [items, itemsData] = obj.trackerDropdownItems_() ;
      referenceTracker = model.referenceTracker ;
      obj.referenceDropdown_.Items = items ;
      obj.referenceDropdown_.ItemsData = itemsData ;
      if ~isempty(itemsData) && ~isempty(referenceTracker)
        obj.referenceDropdown_.Value = clampDropdownValue_(referenceTracker, itemsData) ;
      end
      obj.referenceDropdown_.Enable = 'off' ;

      % The test dropdown lists every tracker except the current
      % (reference) tracker, unless the current tracker is itself
      % selected as the test tracker -- in which case include it so the
      % selection is displayable (and flagged pink).
      testTracker = model.testTracker ;
      isCurrentTrackerSelectedAsTest = ...
        ~isempty(testTracker) && ~isempty(referenceTracker) && (testTracker == referenceTracker) ;
      if isCurrentTrackerSelectedAsTest
        testItems = items ;
        testItemsData = itemsData ;
      else
        areTestCandidates = ~cellfun(@(d)(d == referenceTracker), itemsData) ;
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
        obj.testDropdown_.Value = clampDropdownValue_(testTracker, testItemsData) ;
      end

      % Indicate the reference (== current) tracker in the test dropdown,
      % since selecting it as the test tracker yields no comparison.  In
      % R2023a and later, per-item dropdown styling is available, so pink
      % the reference item when it is present in the list.  In older
      % releases, fall back to flagging the whole control pink when the
      % test selection coincides with the reference.
      if verLessThan('matlab', '9.14')  %#ok<VERLESSMATLAB>  % R2023a
        obj.testDropdown_.BackgroundColor = ...
          fif(model.isTestTrackerChoiceValid, [1 1 1], [1 0.8 0.85]) ;
      else
        removeStyle(obj.testDropdown_) ;
        referenceItemPosition = ...
          find(cellfun(@(d)(d == referenceTracker), testItemsData), 1) ;
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
        obj.listbox_.Items = model.displayStringFromBoutIndex ;
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
      % The listbox selection mirrors the model's currentBoutIndexMaybe;
      % empty means no item selected.  This must come after assigning
      % Items, since that auto-selects the first item.
      boutIndexMaybe = model.currentBoutIndexMaybe ;
      if isempty(boutIndexMaybe)
        obj.listbox_.Value = {} ;
      else
        obj.listbox_.ValueIndex = boutIndexMaybe ;
      end

      obj.updatePreviewAxes_() ;

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
      % Handle test-tracker dropdown change.  src.Value is the selected
      % tracker handle.
      obj.model_.testTracker = src.Value ;
    end  % function

    function compare_trackers_mode_dropdown_actuated_(obj, src)
      % Handle mode dropdown change.  src.Value is the selected
      % CompareTrackersMode.
      obj.model_.mode = src.Value ;
    end  % function
  end  % methods

  methods (Access=private)
    function createFigure_(obj)
      % Create the uifigure and all child controls.
      figurePosition = [200 200 480 880] ;
      obj.figure_ = uifigure(...
        'Name', 'Compare Trackers', ...
        'Position', figurePosition, ...
        'Tag', 'compare_trackers_figure', ...
        'Visible', 'off', ...
        'CloseRequestFcn', @(src, evt)(obj.hideRequested()), ...
        'AutoResizeChildren', 'off', ...
        'SizeChangedFcn', @(src, evt)(obj.updatePreviewRowHeight_())) ;
          % AutoResizeChildren must be off for SizeChangedFcn to fire;
          % the grid layout handles resizing the children regardless.

      labelerController = obj.labelerController_ ;

      gridLayout = uigridlayout(obj.figure_, [6, 1]) ;
      gridLayout.RowHeight = {22, 22, 22, 22, '1x', 460} ;
      gridLayout.ColumnWidth = {'1x'} ;
      obj.gridLayout_ = gridLayout ;

      % Row 1: mode
      modeRow = uigridlayout(gridLayout, [1, 2]) ;
      modeRow.RowHeight = {'1x'} ;
      modeRow.ColumnWidth = {80, '1x'} ;
      modeRow.Padding = [0, 0, 0, 0] ;
      uilabel(modeRow, ...
        'Text', 'Mode:', ...
        'HorizontalAlignment', 'right', ...
        'Tag', 'compare_trackers_mode_label') ;
      modeValues = enumeration('CompareTrackersMode') ;
      modeItems = arrayfun(@(m)(char(m.prettyStr)), modeValues, 'UniformOutput', false) ;
      obj.modeDropdown_ = uidropdown(modeRow, ...
        'Items', modeItems, ...
        'ItemsData', modeValues, ...
        'Tag', 'compare_trackers_mode_dropdown', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('compare_trackers_mode_dropdown', src, evt))) ;

      % Row 2: reference tracker
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

      % Row 3: test tracker
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

      % Row 4: threshold
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

      % Row 5: listbox.  ClickedFcn is registered in addition to
      % ValueChangedFcn so that clicking the already-selected item (which
      % does not fire ValueChangedFcn) still navigates to the bout.
      obj.listbox_ = uilistbox(gridLayout, ...
        'Items', {}, ...
        'Tag', 'compare_trackers_listbox', ...
        'ValueChangedFcn', ...
          @(src, evt)(labelerController.controlActuated('compare_trackers_listbox', src, evt)), ...
        'ClickedFcn', ...
          @(src, evt)(labelerController.controlActuated('compare_trackers_listbox_clicked', src, evt))) ;

      % Row 6: preview axes, showing the selected bout's peak frame with
      % both trackers' poses overlaid.  Kept square (matching the listbox
      % width) by updatePreviewRowHeight_().
      obj.previewAxes_ = uiaxes(gridLayout, ...
        'Tag', 'compare_trackers_preview_axes') ;
      previewAxes = obj.previewAxes_ ;
      previewAxes.XTick = [] ;
      previewAxes.YTick = [] ;
      previewAxes.XColor = 'none' ;
      previewAxes.YColor = 'none' ;
      previewAxes.YDir = 'reverse' ;
      previewAxes.DataAspectRatio = [1 1 1] ;
      previewAxes.NextPlot = 'add' ;
      previewAxes.Color = 'k' ;
      previewAxes.Toolbar.Visible = 'off' ;
      disableDefaultInteractivity(previewAxes) ;
      obj.previewImage_ = image(previewAxes, ...
        'CData', zeros(0, 0), ...
        'Visible', 'off', ...
        'Tag', 'compare_trackers_preview_image') ;
      colormap(previewAxes, 'gray') ;
      obj.connectorLineGroup_ = hggroup('Parent', previewAxes, ...
                                        'Tag', 'compare_trackers_preview_connector_group') ;
      obj.connectorLines_ = gobjects(0, 1) ;
      obj.refScatter_ = scatter(previewAxes, nan, nan, ...
        'Visible', 'off', ...
        'Tag', 'compare_trackers_preview_ref_scatter') ;
      obj.testScatter_ = scatter(previewAxes, nan, nan, ...
        'Visible', 'off', ...
        'Tag', 'compare_trackers_preview_test_scatter') ;
      % Marks the unmatched reference tracks' centroids in
      % UnmatchedAnimalCount mode.  A single conspicuous marker, since a
      % centroid has no per-landmark color.
      obj.unmatchedCentroidScatter_ = scatter(previewAxes, nan, nan, ...
        'Marker', 'x', ...
        'MarkerEdgeColor', [1 0 0], ...
        'LineWidth', 2, ...
        'SizeData', 144, ...
        'Visible', 'off', ...
        'Tag', 'compare_trackers_preview_unmatched_centroid_scatter') ;
      % Normalized units keep the placeholder centered regardless of the
      % axes limits left over from the last-shown image.
      obj.previewPlaceholderText_ = text('Parent', previewAxes, ...
        'Units', 'normalized', ...
        'Position', [0.5, 0.5], ...
        'String', '(No bout selected)', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 14, ...
        'Color', [1, 1, 1], ...
        'Visible', 'on', ...
        'Tag', 'compare_trackers_preview_placeholder_text') ;

      obj.updatePreviewRowHeight_() ;
      mainFigurePosition = obj.labelerController_.mainFigurePixelPosition() ;
      centerOnOtherFigureGivenPositionBang(obj.figure_, mainFigurePosition) ;
    end  % function

    function updatePreviewRowHeight_(obj)
      % Keep the preview axes square by pinning its grid row height to
      % the grid's inner width.
      if ~obj.hasValidFigure_ || isempty(obj.gridLayout_) || ~isvalid(obj.gridLayout_)
        return
      end
      figureWidth = obj.figure_.Position(3) ;
      padding = obj.gridLayout_.Padding ;
      innerWidth = figureWidth - padding(1) - padding(3) ;
      rowHeight = obj.gridLayout_.RowHeight ;
      rowHeight{6} = max(innerWidth, 50) ;
      obj.gridLayout_.RowHeight = rowHeight ;
    end  % function

    function updatePreviewAxes_(obj)
      % Sync the preview axes to the model's currently-selected bout:
      % show the bout's max-distance frame, zoomed to an invisible
      % bounding box around both trackers' poses, with the poses
      % overlaid in the main window's predicted-landmark colors.
      preview = obj.model_.currentBoutPreviewMaybe() ;
      if isempty(preview)
        obj.previewImage_.Visible = 'off' ;
        obj.refScatter_.Visible = 'off' ;
        obj.testScatter_.Visible = 'off' ;
        obj.unmatchedCentroidScatter_.Visible = 'off' ;
        set(obj.connectorLines_, 'Visible', 'off') ;
        obj.previewPlaceholderText_.Visible = 'on' ;
        return
      end
      obj.previewPlaceholderText_.Visible = 'off' ;

      % Show the frame image.  Grayscale frames render through the axes'
      % gray colormap; CDataMapping is ignored for RGB frames.
      imageMatrix = preview.imageMatrix ;
      imageHeight = size(imageMatrix, 1) ;
      imageWidth = size(imageMatrix, 2) ;
      set(obj.previewImage_, ...
          'CData', imageMatrix, ...
          'CDataMapping', 'scaled', ...
          'XData', [1, imageWidth], ...
          'YData', [1, imageHeight], ...
          'Visible', 'on') ;

      % Overlay the poses, using the main window's predicted-landmark
      % colors and marker cosmetics.  The two trackers share landmark
      % colors in the main window, so the test pose gets a distinct
      % marker shape.
      labeler = obj.labeler_ ;
      pointColors = labeler.PredictPointColors() ;
      markerProps = labeler.predPointsPlotInfo.MarkerProps ;
      refMarker = markerProps.Marker ;
      testMarker = fif(strcmp(refMarker, 'o'), 'square', 'o') ;
      sizeData = markerProps.MarkerSize ^ 2 ;
      updatePoseScatterBang_(obj.refScatter_, preview.refPoseXy, pointColors, ...
                             refMarker, sizeData, markerProps.LineWidth) ;
      updatePoseScatterBang_(obj.testScatter_, preview.testPoseXy, pointColors, ...
                             testMarker, sizeData, markerProps.LineWidth) ;
      obj.updateConnectorLines_(preview.refPoseXy, preview.testPoseXy, pointColors) ;

      % Mark the unmatched reference tracks' centroids (UnmatchedAnimalCount
      % mode); empty in the other mode, in which case the markers are
      % hidden.
      unmatchedCentroidsXy = preview.unmatchedCentroidsXy ;
      if isempty(unmatchedCentroidsXy)
        obj.unmatchedCentroidScatter_.Visible = 'off' ;
      else
        set(obj.unmatchedCentroidScatter_, ...
            'XData', unmatchedCentroidsXy(:, 1)', ...
            'YData', unmatchedCentroidsXy(:, 2)', ...
            'Visible', 'on') ;
      end

      % Zoom to an invisible square bounding box around all the
      % landmarks of both poses.
      [xLimits, yLimits] = ...
        squareLimitsFromPoses_(preview.refPoseXy, preview.testPoseXy, imageWidth, imageHeight) ;
      obj.previewAxes_.XLim = xLimits ;
      obj.previewAxes_.YLim = yLimits ;
    end  % function

    function updateConnectorLines_(obj, refPoseXy, testPoseXy, pointColors)
      % Update the lines connecting each ref landmark to the
      % corresponding test landmark in the preview axes, one line per
      % landmark, colored like the landmark.  Lines whose endpoints are
      % not both finite are hidden.
      landmarkCount = size(pointColors, 1) ;
      isDrawable = ...
        size(refPoseXy, 1) == landmarkCount && size(testPoseXy, 1) == landmarkCount ;
      if ~isDrawable
        set(obj.connectorLines_, 'Visible', 'off') ;
        return
      end
      if numel(obj.connectorLines_) ~= landmarkCount
        delete(obj.connectorLines_) ;
        obj.connectorLines_ = gobjects(landmarkCount, 1) ;
        for landmarkIndex = 1 : landmarkCount
          obj.connectorLines_(landmarkIndex) = ...
            line('Parent', obj.connectorLineGroup_, ...
                 'XData', nan, ...
                 'YData', nan, ...
                 'LineWidth', 1, ...
                 'Visible', 'off', ...
                 'Tag', 'compare_trackers_preview_connector_line') ;
        end
      end
      for landmarkIndex = 1 : landmarkCount
        lineHandle = obj.connectorLines_(landmarkIndex) ;
        endpointXy = [refPoseXy(landmarkIndex, :) ; testPoseXy(landmarkIndex, :)] ;
        if all(isfinite(endpointXy(:)))
          set(lineHandle, ...
              'XData', endpointXy(:, 1), ...
              'YData', endpointXy(:, 2), ...
              'Color', pointColors(landmarkIndex, :), ...
              'Visible', 'on') ;
        else
          lineHandle.Visible = 'off' ;
        end
      end
    end  % function

    function [items, itemsData] = trackerDropdownItems_(obj)
      % Build the dropdown Items / ItemsData arrays from the labeler's
      % current trackerHistory.  Returns parallel cell arrays.  Each
      % ItemsData entry is the tracker handle itself, so selections are
      % tracked by identity rather than by position in trackerHistory.
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
        itemsData{i} = tracker ;
      end
    end  % function
  end  % methods
end  % classdef



function updatePoseScatterBang_(scatterHandle, poseXy, pointColors, marker, sizeData, lineWidth)
% Update one pose-overlay scatter object in place to show the given pose
% ([landmarkCount x 2], possibly empty) with per-landmark colors.
landmarkCount = size(poseXy, 1) ;
if landmarkCount == 0 || size(pointColors, 1) ~= landmarkCount
  scatterHandle.Visible = 'off' ;
  return
end
set(scatterHandle, ...
    'XData', poseXy(:, 1)', ...
    'YData', poseXy(:, 2)', ...
    'CData', pointColors, ...
    'Marker', marker, ...
    'SizeData', sizeData, ...
    'LineWidth', lineWidth, ...
    'Visible', 'on') ;
end  % function



function [xLimits, yLimits] = squareLimitsFromPoses_(refPoseXy, testPoseXy, imageWidth, imageHeight)
% Compute square axes limits bounding all the finite landmarks of both
% poses, with some margin, shifted/clipped to lie within the image.
% Falls back to the whole image when there are no finite landmarks.
allXy = [refPoseXy ; testPoseXy] ;
isRowFinite = all(isfinite(allXy), 2) ;
finiteXy = allXy(isRowFinite, :) ;
if isempty(finiteXy)
  xLimits = [0.5, imageWidth + 0.5] ;
  yLimits = [0.5, imageHeight + 0.5] ;
  return
end
minXy = min(finiteXy, [], 1) ;
maxXy = max(finiteXy, [], 1) ;
centerXy = (minXy + maxXy) / 2 ;
marginFactor = 2 ;
minimumHalfSpan = 10 ;
% Pad each dimension proportionally, then square up to the larger of the
% two padded half-spans.
paddedHalfSpanXy = (maxXy - minXy) / 2 * marginFactor ;
halfSpan = max(max(paddedHalfSpanXy), minimumHalfSpan) ;
xLimits = shiftIntervalIntoRange_(centerXy(1) + [-1, 1] * halfSpan, [0.5, imageWidth + 0.5]) ;
yLimits = shiftIntervalIntoRange_(centerXy(2) + [-1, 1] * halfSpan, [0.5, imageHeight + 0.5]) ;
end  % function



function shifted = shiftIntervalIntoRange_(interval, range)
% Shift the given interval to lie within range if possible, preserving
% its width; if it is wider than range, return range itself.
width = diff(interval) ;
if width >= diff(range)
  shifted = range ;
  return
end
if interval(1) < range(1)
  shifted = [range(1), range(1) + width] ;
elseif interval(2) > range(2)
  shifted = [range(2) - width, range(2)] ;
else
  shifted = interval ;
end
end  % function



function clampedValue = clampDropdownValue_(desiredTracker, itemsData)
% Return desiredTracker if it is one of the trackers in itemsData;
% otherwise return the first item.  Used to keep the dropdown selection
% valid when the trackerHistory changes.  Compared by identity, so a
% backup copy of a tracker does not count as a match.
isPresent = ~isempty(desiredTracker) && any(cellfun(@(d)(d == desiredTracker), itemsData)) ;
if isPresent
  clampedValue = desiredTracker ;
else
  clampedValue = itemsData{1} ;
end
end  % function
