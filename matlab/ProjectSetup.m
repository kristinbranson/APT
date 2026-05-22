classdef ProjectSetup < matlab.apps.AppBase

  % Widget properties (emitted by App Designer Migration Tool)
  properties (Access = public, Transient)
    fig                           matlab.ui.Figure
    labelHasBodyTrackingDetails   matlab.ui.control.Label
    labelMultipleAnimalsDetails   matlab.ui.control.Label
    labelNumberOfViewsDetails     matlab.ui.control.Label
    labelNumberOfKeypointsDetails matlab.ui.control.Label
    labelMultipleAnimals          matlab.ui.control.Label
    cbMA                          matlab.ui.control.CheckBox
    labelHasBodyTracking          matlab.ui.control.Label
    cbHasTrx                      matlab.ui.control.CheckBox
    pbCopySettingsFrom            matlab.ui.control.Button
    pbCancel                      matlab.ui.control.Button
    pbCreateProject               matlab.ui.control.Button
    etNumberOfViews               matlab.ui.control.EditField
    etNumberOfPoints              matlab.ui.control.EditField
    etProjectName                 matlab.ui.control.EditField
    labelNumberOfViews            matlab.ui.control.Label
    labelNumberOfKeypoints        matlab.ui.control.Label
    labelProjectName              matlab.ui.control.Label
  end

  % Non-widget state (private in spirit; underscore suffix marks the intent)
  properties (Access = private, Transient)
    viewCount_  = 1
    pointCount_ = 1
    propsPane_  = []
    mirror_     = []
    output_     = []
  end

  properties (Dependent)
    output
  end

  methods
    function result = get.output(app)
      % Getter for output: the project-config struct chosen by the user,
      % or [] if the dialog was cancelled.
      result = app.output_ ;
    end  % function
  end  % methods

  methods (Access = public)
    function app = ProjectSetup(varargin)
      % Build and show the ProjectSetup dialog; block until the user closes it.
      app.createComponents() ;
      app.registerApp(app.fig) ;
      app.opening_(varargin{:}) ;
    end  % function

    function delete(app)
      % Delete the underlying figure when the app is deleted.
      delete(app.fig) ;
    end  % function

    % function advModeCollapse_(app)
    %   % Collapse the dialog so the advanced-properties panel is hidden.
    %   posMid = app.landmarkMid.Position ;
    %   posMid = posMid(1) + posMid(3)/2 ;
    %   pos = app.fig.Position ;
    %   pos(3) = posMid ;
    %   app.fig.Position = pos ;
    %   app.advancedOn_ = false ;
    %   % app.pbAdvanced.Text = 'Advanced >' ;
    % end  % function

    % function advModeExpand_(app)
    %   % % Expand the dialog so the advanced-properties panel is visible.
    %   % posRight = app.landmarkRight.Position ;
    %   % posRight = posRight(1) + posRight(3) ;
    %   % pos = app.fig.Position ;
    %   % pos(3) = posRight ;
    %   % app.fig.Position = pos ;
    %   % app.advancedOn_ = true ;
    %   % % app.pbAdvanced.Text = '< Basic' ;
    % end  % function

    % function advModeToggle_(app)
    %   % Toggle the dialog between expanded and collapsed advanced mode.
    %   if app.advancedOn_
    %     app.advModeCollapse_() ;
    %   else
    %     % app.advModeExpand_() ;
    %   end
    % end  % function

    function advTableRefresh_(app, sMirror)
      % Refresh the advanced-properties table to match the current view and
      % point counts.  If sMirror is supplied, it replaces the current mirror;
      % otherwise the existing mirror is reused.
      if ~exist('sMirror', 'var')
        sMirror = app.mirror_ ;
      end
      sMirror = Labeler.hlpAugmentOrTruncNameField(sMirror, 'ViewNames', 'view', app.viewCount_) ;
      sMirror = Labeler.hlpAugmentOrTruncNameField(sMirror, 'LabelPointNames', 'point', app.pointCount_) ;
      sMirror = Labeler.hlpAugmentOrTruncStructField(sMirror, 'View', app.viewCount_) ;
      app.mirror_ = sMirror ;
      if ~isempty(app.propsPane_) && ishandle(app.propsPane_)
        delete(app.propsPane_) ;
        app.propsPane_ = [] ;
      end
      app.propsPane_ = [] ;  % newPropertiesGUI(app.pnlAdvanced, sMirror) when re-enabled
    end  % function

    function cfg = genCurrentConfig_(app)
      % Generate a config struct from the current UI state.
      cfg = app.mirror_ ;

      assert(numel(fieldnames(cfg.ViewNames)) == app.viewCount_) ;
      assert(numel(fieldnames(cfg.LabelPointNames)) == app.pointCount_) ;
      cfg.NumViews = app.viewCount_ ;
      cfg.NumLabelPoints = app.pointCount_ ;
      cfg.ViewNames = struct2cell(cfg.ViewNames) ;
      cfg.LabelPointNames = struct2cell(cfg.LabelPointNames) ;
      cfg.Trx.HasTrx = app.cbHasTrx.Value ;
      cfg.MultiAnimal = app.cbMA.Value ;
      isMultiAnimal = cfg.MultiAnimal && ~cfg.Trx.HasTrx ;
      if isMultiAnimal
        cfg.LabelMode = LabelMode.MULTIANIMAL ;
      else
        cfg.LabelMode = LabelMode.SEQUENTIAL ;
      end
      cfg.Track.Enable = true ;
      % propertiesGUI treats props with empty values as strings even if they
      % are subsequently filled with numbers
      fieldsToDoublify = {'Gamma', 'FigurePos', 'AxisLim', 'InvertMovie', 'AxFontSize', 'ShowAxTicks', 'ShowGrid'} ;
      for i = 1:numel(cfg.View)
        cfg.View(i) = structLeavesStr2Double(cfg.View(i), fieldsToDoublify) ;
      end
    end  % function

    function setCurrentConfig_(app, cfg)
      % Set the given config struct on the controls and on internal state.
      app.viewCount_ = cfg.NumViews ;
      app.pointCount_ = cfg.NumLabelPoints ;
      app.etNumberOfViews.Value = num2str(app.viewCount_) ;
      app.etNumberOfPoints.Value = num2str(app.pointCount_) ;
      app.cbHasTrx.Value = cfg.Trx.HasTrx ;
      app.cbMA.Value = cfg.MultiAnimal ;
      sMirror = Labeler.cfg2mirror(cfg) ;
      app.advTableRefresh_(sMirror) ;
    end  % function
  end  % methods (Access = public)

  methods (Access = private)
    function opening_(app, varargin)
      % Initialize dialog state.  Returns once the figure is shown and
      % populated; the caller is responsible for blocking (e.g. via
      % uiwait(app.fig)) and for reading app.output afterwards.
      %
      %   app = ProjectSetup() ;
      %   app = ProjectSetup(hParentFig) ;  % centered on hParentFig
      movegui(app.fig, 'onscreen') ;

      if numel(varargin) >= 1
        hParentFig = varargin{1} ;
        if ~ishandle(hParentFig)
          error('ProjectSetup:arg', 'Expected argument to be a figure handle.') ;
        end
        centerOnParentFigure(app.fig, hParentFig) ;
      end

      cfg = Labeler.cfgGetLastProjectConfigNoView() ;
      app.setCurrentConfig_(cfg) ;
      % app.advModeCollapse_() ;
    end  % function

    % Value-changed handler for the keypoint-count edit field.
    function etNumberOfPoints_Callback(app, ~)
      val = str2double(app.etNumberOfPoints.Value) ;
      if floor(val) == val && val >= 1
        app.pointCount_ = val ;
      else
        app.etNumberOfPoints.Value = num2str(app.pointCount_) ;
      end
      app.advTableRefresh_() ;
    end  % function

    % Value-changed handler for the view-count edit field.
    function etNumberOfViews_Callback(app, ~)
      val = str2double(app.etNumberOfViews.Value) ;
      if floor(val) == val && val >= 1
        app.viewCount_ = val ;
      else
        app.etNumberOfViews.Value = num2str(app.viewCount_) ;
      end
      switch app.viewCount_
        case 1
          app.cbHasTrx.Enable = 'on' ;
          app.cbMA.Enable = 'on' ;
        otherwise
          app.cbHasTrx.Value = false ;
          app.cbMA.Value = false ;
          app.cbHasTrx.Enable = 'off' ;
          app.cbMA.Enable = 'off' ;
      end
      app.advTableRefresh_() ;
    end  % function

    % Value-changed handler for the project-name edit field.
    function etProjectName_Callback(app, ~)
      name = app.etProjectName.Value ;
      if ~all(isstrprop(name, 'alphanum'))
        % This unfortunately invalidates _ also.  Checking for it seems more
        % work than worth.  MK 20220913
        warndlg('Name should have only alphanumeric characters') ;
        app.etProjectName.Value = '' ;
      end
    end  % function

    % Close-request function for the main figure.
    function figure1_CloseRequestFcn(app, ~)
      if isequal(get(app.fig, 'waitstatus'), 'waiting')
        % The dialog is still in uiwait; release it.
        uiresume(app.fig) ;
      else
        delete(app.fig) ;
      end
    end  % function

    % Button-pushed function for the Advanced/Basic toggle.
    function pbAdvanced_Callback(app, ~)
      app.advModeToggle_() ;
    end  % function

    % Button-pushed function for the Cancel button.
    function pbCancel_Callback(app, ~)
      app.output_ = [] ;
      close(app.fig) ;
    end  % function

    % Button-pushed function for the Copy Settings From... button.
    function pbCopySettingsFrom_Callback(app, ~)
      lastLblFile = RC.getprop('lastLblFile') ;
      if isempty(lastLblFile)
        lastLblFile = pwd ;
      end
      [fname, pth] = uigetfile('*.lbl', 'Select project file', lastLblFile) ;
      if isequal(fname, 0)
        return
      end
      lbl = loadLbl(fullfile(pth, fname)) ;
      lbl = Labeler.lblModernize(lbl) ;
      cfg = lbl.cfg ;
      app.setCurrentConfig_(cfg) ;
    end  % function

    % Button-pushed function for the Create Project button.
    function pbCreateProject_Callback(app, ~)
      cfg = app.genCurrentConfig_() ;
      cfg.ProjectName = app.etProjectName.Value ;
      app.output_ = cfg ;
      close(app.fig) ;
    end  % function
  end  % methods (Access = private)

  % Component initialization
  methods (Access = private)

    % Create UIFigure and components
    function createComponents(app)
      % Create the figure and all its child controls, laid out via
      % nested uigridlayouts: an outer 7-row column, where each
      % "Number of X" / question section is itself a 2-row column
      % (label-and-control row, then details label).

      % Layout constants
      figWidth = 450 ;
      marginWidth = 25 ;
      bgColor = [0 0.243 0.365] ;  % close to prussian blue
      labelColor = [0 1 1] ;  % cyan
      fieldFontColor = [0 250 209]/255 ;  % close to turqoise
      fieldBgColor = [0 0 0] ;  % black
      bigFontSize = 20 ;
      smallFontSize = 16 ;
      projectNameFieldWidth = 250 ;
      numberFieldWidth = 150 ;
      createProjectButtonWidth = 200 ;
      cancelButtonWidth = 150 ;
      copySettingsButtonWidth = 180 ;
      buttonHeight = 32 ;
      withinSectionRowSpacing = 8 ;  % gap between a section's label-and-control row and its details label
      outerGridRowSpacing = 12 ;  % gap between outerGrid rows
      copySettingsOverlapAmount = 6 ;  % how far the Copy Settings button overlaps into the Has Body Tracking details, matches the original GUIDE layout

      % Figure height: derived so the column of content fits with no slack.
      % Magic number 620 was the design height when the 4 sections had
      % RowSpacing = 2; each added pixel of within-section spacing adds 4
      % pixels of vertical content.
      figHeight = 620 + 4 * (withinSectionRowSpacing - 2) ;

      % Create the figure
      app.fig = uifigure('Visible', 'off') ;
      app.fig.Color = bgColor ;
      app.fig.Position = [100 100 figWidth figHeight] ;
      app.fig.Name = 'Project Setup' ;
      app.fig.Resize = 'off' ;
      app.fig.CloseRequestFcn = app.createCallbackFcn(@figure1_CloseRequestFcn, true) ;
      app.fig.HandleVisibility = 'callback' ;
      app.fig.Tag = 'figure1' ;

      % Outer column: 7 rows.  Rows 1-5 are content sections (sized 'fit').
      % Row 6 is reserved empty space for the Copy Settings button, which is
      % added as a direct child of the figure (not the grid) so it can
      % overlap the Has Body Tracking details area above.  Row 7 holds the
      % bottom buttons.
      outerGrid = uigridlayout(app.fig, [7 1]) ;
      outerGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', buttonHeight, buttonHeight} ;
      outerGrid.ColumnWidth = {'1x'} ;
      outerGrid.Padding = [marginWidth marginWidth marginWidth marginWidth] ;
      outerGrid.RowSpacing = outerGridRowSpacing ;
      outerGrid.BackgroundColor = bgColor ;

      % Row 1: Project Name -----------------------------------------
      projectNameRow = uigridlayout(outerGrid, [1 2]) ;
      projectNameRow.ColumnWidth = {'1x', projectNameFieldWidth} ;
      projectNameRow.RowHeight = {'fit'} ;
      projectNameRow.Padding = [0 0 0 0] ;
      projectNameRow.ColumnSpacing = 10 ;
      projectNameRow.BackgroundColor = bgColor ;

      app.labelProjectName = uilabel(projectNameRow) ;
      app.labelProjectName.Text = 'Project Name' ;
      app.labelProjectName.FontSize = bigFontSize ;
      app.labelProjectName.FontColor = labelColor ;
      app.labelProjectName.BackgroundColor = bgColor ;

      app.etProjectName = uieditfield(projectNameRow, 'text') ;
      app.etProjectName.ValueChangedFcn = app.createCallbackFcn(@etProjectName_Callback, true) ;
      app.etProjectName.FontSize = bigFontSize ;
      app.etProjectName.FontColor = fieldFontColor ;
      app.etProjectName.BackgroundColor = fieldBgColor ;

      % Row 2: Number of Keypoints ---------------------------------
      keypointsSection = uigridlayout(outerGrid, [2 1]) ;
      keypointsSection.RowHeight = {'fit', 'fit'} ;
      keypointsSection.ColumnWidth = {'1x'} ;
      keypointsSection.Padding = [0 0 0 0] ;
      keypointsSection.RowSpacing = withinSectionRowSpacing ;
      keypointsSection.BackgroundColor = bgColor ;

      keypointsRow = uigridlayout(keypointsSection, [1 2]) ;
      keypointsRow.ColumnWidth = {'1x', numberFieldWidth} ;
      keypointsRow.RowHeight = {'fit'} ;
      keypointsRow.Padding = [0 0 0 0] ;
      keypointsRow.ColumnSpacing = 10 ;
      keypointsRow.BackgroundColor = bgColor ;

      app.labelNumberOfKeypoints = uilabel(keypointsRow) ;
      app.labelNumberOfKeypoints.Text = 'Number of Keypoints' ;
      app.labelNumberOfKeypoints.FontSize = bigFontSize ;
      app.labelNumberOfKeypoints.FontColor = labelColor ;
      app.labelNumberOfKeypoints.BackgroundColor = bgColor ;

      app.etNumberOfPoints = uieditfield(keypointsRow, 'text') ;
      app.etNumberOfPoints.ValueChangedFcn = app.createCallbackFcn(@etNumberOfPoints_Callback, true) ;
      app.etNumberOfPoints.HorizontalAlignment = 'right' ;
      app.etNumberOfPoints.FontSize = bigFontSize ;
      app.etNumberOfPoints.FontColor = fieldFontColor ;
      app.etNumberOfPoints.BackgroundColor = fieldBgColor ;
      app.etNumberOfPoints.Value = '12' ;

      app.labelNumberOfKeypointsDetails = uilabel(keypointsSection) ;
      app.labelNumberOfKeypointsDetails.Text = 'Number of keypoints to label for each animal' ;
      app.labelNumberOfKeypointsDetails.WordWrap = 'on' ;
      app.labelNumberOfKeypointsDetails.VerticalAlignment = 'top' ;
      app.labelNumberOfKeypointsDetails.FontSize = smallFontSize ;
      app.labelNumberOfKeypointsDetails.FontColor = labelColor ;
      app.labelNumberOfKeypointsDetails.BackgroundColor = bgColor ;

      % Row 3: Number of Views -------------------------------------
      viewsSection = uigridlayout(outerGrid, [2 1]) ;
      viewsSection.RowHeight = {'fit', 'fit'} ;
      viewsSection.ColumnWidth = {'1x'} ;
      viewsSection.Padding = [0 0 0 0] ;
      viewsSection.RowSpacing = withinSectionRowSpacing ;
      viewsSection.BackgroundColor = bgColor ;

      viewsRow = uigridlayout(viewsSection, [1 2]) ;
      viewsRow.ColumnWidth = {'1x', numberFieldWidth} ;
      viewsRow.RowHeight = {'fit'} ;
      viewsRow.Padding = [0 0 0 0] ;
      viewsRow.ColumnSpacing = 10 ;
      viewsRow.BackgroundColor = bgColor ;

      app.labelNumberOfViews = uilabel(viewsRow) ;
      app.labelNumberOfViews.Text = 'Number of Views' ;
      app.labelNumberOfViews.FontSize = bigFontSize ;
      app.labelNumberOfViews.FontColor = labelColor ;
      app.labelNumberOfViews.BackgroundColor = bgColor ;

      app.etNumberOfViews = uieditfield(viewsRow, 'text') ;
      app.etNumberOfViews.ValueChangedFcn = app.createCallbackFcn(@etNumberOfViews_Callback, true) ;
      app.etNumberOfViews.HorizontalAlignment = 'right' ;
      app.etNumberOfViews.FontSize = bigFontSize ;
      app.etNumberOfViews.FontColor = fieldFontColor ;
      app.etNumberOfViews.BackgroundColor = fieldBgColor ;
      app.etNumberOfViews.Value = '1' ;

      app.labelNumberOfViewsDetails = uilabel(viewsSection) ;
      app.labelNumberOfViewsDetails.Text = 'APT can do 3D labeling and tracking from multiple calibrated cameras. Enter 1 if animals were imaged from just one camera. Otherwise, enter the number of synced cameras recording the animals.' ;
      app.labelNumberOfViewsDetails.WordWrap = 'on' ;
      app.labelNumberOfViewsDetails.VerticalAlignment = 'top' ;
      app.labelNumberOfViewsDetails.FontSize = smallFontSize ;
      app.labelNumberOfViewsDetails.FontColor = labelColor ;
      app.labelNumberOfViewsDetails.BackgroundColor = bgColor ;

      % Row 4: Multiple Animals ------------------------------------
      multipleAnimalsSection = uigridlayout(outerGrid, [2 1]) ;
      multipleAnimalsSection.RowHeight = {'fit', 'fit'} ;
      multipleAnimalsSection.ColumnWidth = {'1x'} ;
      multipleAnimalsSection.Padding = [0 0 0 0] ;
      multipleAnimalsSection.RowSpacing = withinSectionRowSpacing ;
      multipleAnimalsSection.BackgroundColor = bgColor ;

      multipleAnimalsRow = uigridlayout(multipleAnimalsSection, [1 2]) ;
      multipleAnimalsRow.ColumnWidth = {'1x', numberFieldWidth} ;
      multipleAnimalsRow.RowHeight = {'fit'} ;
      multipleAnimalsRow.Padding = [0 0 0 0] ;
      multipleAnimalsRow.ColumnSpacing = 10 ;
      multipleAnimalsRow.BackgroundColor = bgColor ;

      app.labelMultipleAnimals = uilabel(multipleAnimalsRow) ;
      app.labelMultipleAnimals.Text = 'Multiple Animals?' ;
      app.labelMultipleAnimals.FontSize = bigFontSize ;
      app.labelMultipleAnimals.FontColor = labelColor ;
      app.labelMultipleAnimals.BackgroundColor = bgColor ;

      % Center cbMA horizontally within the numberFieldWidth-wide cell so
      % it aligns with the centered "12" / "1" in the number fields above.
      cbMACell = uigridlayout(multipleAnimalsRow, [1 3]) ;
      cbMACell.ColumnWidth = {'1x', 'fit', '1x'} ;
      cbMACell.RowHeight = {'fit'} ;
      cbMACell.Padding = [0 0 0 0] ;
      cbMACell.ColumnSpacing = 0 ;
      cbMACell.BackgroundColor = bgColor ;

      app.cbMA = uicheckbox(cbMACell) ;
      app.cbMA.Layout.Column = 2 ;
      app.cbMA.Text = '' ;
      app.cbMA.FontSize = bigFontSize ;
      app.cbMA.FontColor = labelColor ;

      app.labelMultipleAnimalsDetails = uilabel(multipleAnimalsSection) ;
      app.labelMultipleAnimalsDetails.Text = 'Check this box if there are multiple animals visible in any video frames. Otherwise, APT will assume there is just one animal visible per frame.' ;
      app.labelMultipleAnimalsDetails.WordWrap = 'on' ;
      app.labelMultipleAnimalsDetails.VerticalAlignment = 'top' ;
      app.labelMultipleAnimalsDetails.FontSize = smallFontSize ;
      app.labelMultipleAnimalsDetails.FontColor = labelColor ;
      app.labelMultipleAnimalsDetails.BackgroundColor = bgColor ;

      % Row 5: Has Body Tracking -----------------------------------
      hasBodyTrackingSection = uigridlayout(outerGrid, [2 1]) ;
      hasBodyTrackingSection.RowHeight = {'fit', 'fit'} ;
      hasBodyTrackingSection.ColumnWidth = {'1x'} ;
      hasBodyTrackingSection.Padding = [0 0 0 0] ;
      hasBodyTrackingSection.RowSpacing = withinSectionRowSpacing ;
      hasBodyTrackingSection.BackgroundColor = bgColor ;

      hasBodyTrackingRow = uigridlayout(hasBodyTrackingSection, [1 2]) ;
      hasBodyTrackingRow.ColumnWidth = {'1x', numberFieldWidth} ;
      hasBodyTrackingRow.RowHeight = {'fit'} ;
      hasBodyTrackingRow.Padding = [0 0 0 0] ;
      hasBodyTrackingRow.ColumnSpacing = 10 ;
      hasBodyTrackingRow.BackgroundColor = bgColor ;

      app.labelHasBodyTracking = uilabel(hasBodyTrackingRow) ;
      app.labelHasBodyTracking.Text = 'Has Body Tracking?' ;
      app.labelHasBodyTracking.FontSize = bigFontSize ;
      app.labelHasBodyTracking.FontColor = labelColor ;
      app.labelHasBodyTracking.BackgroundColor = bgColor ;

      cbHasTrxCell = uigridlayout(hasBodyTrackingRow, [1 3]) ;
      cbHasTrxCell.ColumnWidth = {'1x', 'fit', '1x'} ;
      cbHasTrxCell.RowHeight = {'fit'} ;
      cbHasTrxCell.Padding = [0 0 0 0] ;
      cbHasTrxCell.ColumnSpacing = 0 ;
      cbHasTrxCell.BackgroundColor = bgColor ;

      app.cbHasTrx = uicheckbox(cbHasTrxCell) ;
      app.cbHasTrx.Layout.Column = 2 ;
      app.cbHasTrx.Text = '' ;
      app.cbHasTrx.FontSize = 24 ;
      app.cbHasTrx.FontColor = labelColor ;

      app.labelHasBodyTrackingDetails = uilabel(hasBodyTrackingSection) ;
      app.labelHasBodyTrackingDetails.Text = 'APT can do pose tracking on top of body tracking from an algorithm like FlyTracker or Ctrax. Check this box if you have already tracked the centroids and orientations of your animals and want to base pose tracking on those trajectories. If so, a trajectory file is input with each video.' ;
      app.labelHasBodyTrackingDetails.WordWrap = 'on' ;
      app.labelHasBodyTrackingDetails.VerticalAlignment = 'top' ;
      app.labelHasBodyTrackingDetails.FontSize = smallFontSize ;
      app.labelHasBodyTrackingDetails.FontColor = labelColor ;
      app.labelHasBodyTrackingDetails.BackgroundColor = bgColor ;

      % Row 6 is reserved empty space; pbCopySettingsFrom is added below as
      % a direct child of the figure so it can overlap the Has Body Tracking
      % details area in row 5.
      emptyRow = uigridlayout(outerGrid, [1 1]) ;
      emptyRow.Padding = [0 0 0 0] ;
      emptyRow.BackgroundColor = bgColor ;

      % Row 7: Bottom buttons (Create Project + Cancel, centered, with
      % an explicit gap between the two buttons) -------------------
      betweenBottomButtonsWidth = 20 ;
      bottomButtonsRow = uigridlayout(outerGrid, [1 5]) ;
      bottomButtonsRow.ColumnWidth = {'1x', createProjectButtonWidth, betweenBottomButtonsWidth, cancelButtonWidth, '1x'} ;
      bottomButtonsRow.RowHeight = {'fit'} ;
      bottomButtonsRow.Padding = [0 0 0 0] ;
      bottomButtonsRow.ColumnSpacing = 0 ;
      bottomButtonsRow.BackgroundColor = bgColor ;

      app.pbCreateProject = uibutton(bottomButtonsRow, 'push') ;
      app.pbCreateProject.Layout.Column = 2 ;
      app.pbCreateProject.ButtonPushedFcn = app.createCallbackFcn(@pbCreateProject_Callback, true) ;
      app.pbCreateProject.BackgroundColor = fieldBgColor ;
      app.pbCreateProject.FontSize = bigFontSize ;
      app.pbCreateProject.FontColor = fieldFontColor ;
      app.pbCreateProject.Text = 'Create Project' ;

      app.pbCancel = uibutton(bottomButtonsRow, 'push') ;
      app.pbCancel.Layout.Column = 4 ;
      app.pbCancel.ButtonPushedFcn = app.createCallbackFcn(@pbCancel_Callback, true) ;
      app.pbCancel.BackgroundColor = fieldBgColor ;
      app.pbCancel.FontSize = bigFontSize ;
      app.pbCancel.FontColor = fieldFontColor ;
      app.pbCancel.Text = 'Cancel' ;

      % Copy Settings... button.  Created as a direct child of the figure
      % (not inside outerGrid) so it can overlap into the Has Body Tracking
      % details area above, matching the look of the original GUIDE layout.
      % Drawn on top of outerGrid because it is created later.
      app.pbCopySettingsFrom = uibutton(app.fig, 'push') ;
      app.pbCopySettingsFrom.ButtonPushedFcn = app.createCallbackFcn(@pbCopySettingsFrom_Callback, true) ;
      app.pbCopySettingsFrom.BackgroundColor = fieldBgColor ;
      app.pbCopySettingsFrom.FontSize = bigFontSize ;
      app.pbCopySettingsFrom.FontColor = labelColor ;
      app.pbCopySettingsFrom.Tooltip = 'Copy settings from an existing project' ;
      app.pbCopySettingsFrom.Text = 'Copy Settings...' ;
      % Position derived from the known outerGrid layout.  Bottom of the Has
      % Body Tracking details label, measured from the figure bottom, is:
      %   marginWidth + row 7 + spacing + row 6 + spacing
      detailsBottomY = marginWidth + buttonHeight + outerGridRowSpacing + buttonHeight + outerGridRowSpacing ;
      copySettingsButtonX = (figWidth - marginWidth) - copySettingsButtonWidth ;
      copySettingsButtonY = detailsBottomY + copySettingsOverlapAmount - buttonHeight ;
      app.pbCopySettingsFrom.Position = [copySettingsButtonX, copySettingsButtonY, copySettingsButtonWidth, buttonHeight] ;

      % Show the figure after all components are created
      app.fig.Visible = 'on';
    end  % function
  end  % methods (Access = private)
end  % classdef
