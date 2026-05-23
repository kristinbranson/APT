classdef ProjectSetup < matlab.apps.AppBase

  % Widget properties (emitted by App Designer Migration Tool)
  properties (Access = public, Transient)
    fig                                  matlab.ui.Figure
    project_name_label                   matlab.ui.control.Label
    project_name_edit                    matlab.ui.control.EditField
    number_of_keypoints_label            matlab.ui.control.Label
    number_of_keypoints_edit             matlab.ui.control.EditField
    number_of_keypoints_details_label    matlab.ui.control.Label
    number_of_views_label                matlab.ui.control.Label
    number_of_views_edit                 matlab.ui.control.EditField
    number_of_views_details_label        matlab.ui.control.Label
    multiple_animals_label               matlab.ui.control.Label
    multiple_animals_checkbox            matlab.ui.control.CheckBox
    multiple_animals_details_label       matlab.ui.control.Label
    has_body_tracking_label              matlab.ui.control.Label
    has_body_tracking_checkbox           matlab.ui.control.CheckBox
    has_body_tracking_details_label      matlab.ui.control.Label
    copy_settings_from_button            matlab.ui.control.Button
    create_project_button                matlab.ui.control.Button
    cancel_button                        matlab.ui.control.Button
  end

  % Non-widget state (private in spirit; underscore suffix marks the intent)
  properties (Access = private, Transient)
    cfg_
    output_
  end

  properties (Dependent)
    output
  end

  methods
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
    
    function result = get.output(app)
      % Getter for output: the project-config struct chosen by the user,
      % or [] if the dialog was cancelled.
      result = app.output_ ;
    end  % function
  end  % methods
  
  methods (Access = public)
    function result = generateFinalConfig_(app)
      % Generate a config struct from the current object state.  Takes
      % app.cfg_ as a base, brings its variable-length fields (ViewNames,
      % LabelPointNames, View) in line with the current view/point counts,
      % then overlays the values from the UI.
      storedCfg = app.cfg_ ;
      projectName = app.project_name_edit.Value ;
      keypointCount = str2double(app.number_of_keypoints_edit.Value) ;
      viewCount = str2double(app.number_of_views_edit.Value) ;
      hasBodyTracking = app.has_body_tracking_checkbox.Value ;
      multipleAnimals = app.multiple_animals_checkbox.Value ;
      result = ProjectSetup.patchCfg(storedCfg, projectName, keypointCount, viewCount, ...
                                     hasBodyTracking, multipleAnimals) ;
    end  % function

    function setCfg_(app, cfg)
      % Set the given config struct on the controls and on internal state.
      app.cfg_ = cfg ;
      app.number_of_views_edit.Value = num2str(cfg.NumViews) ;
      app.number_of_keypoints_edit.Value = num2str(cfg.NumLabelPoints) ;
      app.has_body_tracking_checkbox.Value = cfg.Trx.HasTrx ;
      app.multiple_animals_checkbox.Value = cfg.MultiAnimal ;
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
      app.setCfg_(cfg) ;
    end  % function

    % Value-changed handler for the keypoint-count edit field.
    function number_of_points_edit_Callback(app, event)
      rawValue = str2double(app.number_of_keypoints_edit.Value) ;
      if ~(floor(rawValue) == rawValue && rawValue >= 1)
        app.number_of_keypoints_edit.Value = event.PreviousValue ;
      end
    end  % function

    % Value-changed handler for the view-count edit field.
    function number_of_views_edit_Callback(app, event)
      rawValue = str2double(app.number_of_views_edit.Value) ;
      if ~(floor(rawValue) == rawValue && rawValue >= 1)
        app.number_of_views_edit.Value = event.PreviousValue ;
      end
      viewCount = str2double(app.number_of_views_edit.Value) ;
      switch viewCount
        case 1
          app.has_body_tracking_checkbox.Enable = 'on' ;
          app.multiple_animals_checkbox.Enable = 'on' ;
        otherwise
          app.has_body_tracking_checkbox.Value = false ;
          app.multiple_animals_checkbox.Value = false ;
          app.has_body_tracking_checkbox.Enable = 'off' ;
          app.multiple_animals_checkbox.Enable = 'off' ;
      end
    end  % function

    % Value-changed handler for the project-name edit field.
    function project_name_edit_Callback(app, event)
      name = app.project_name_edit.Value ;
      if ~all(isstrprop(name, 'alphanum'))
        % This unfortunately invalidates _ also.  Checking for it seems more
        % work than worth.  MK 20220913
        warndlg('Name should have only alphanumeric characters') ;
        app.project_name_edit.Value = event.PreviousValue  ;
      end
    end  % function

    % Close-request function for the main figure.
    function didRequestClose(app, ~)
      % if isequal(get(app.fig, 'waitstatus'), 'waiting')
      %   % The dialog is still in uiwait; release it.
      %   uiresume(app.fig) ;
      % else
      %   delete(app.fig) ;
      % end
      app.output_ = [] ;
      delete(app.fig) ;
    end  % function

    % Button-pushed function for the Cancel button.
    function cancel_button_Callback(app, ~)
      app.output_ = [] ;
      delete(app.fig) ;
    end  % function

    % Button-pushed function for the Copy Settings From... button.
    function copy_settings_from_button_Callback(app, ~)
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
      app.setCfg_(cfg) ;
    end  % function

    % Button-pushed function for the Create Project button.
    function create_project_button_Callback(app, ~)
      cfg = app.generateFinalConfig_() ;
      app.output_ = cfg ;
      delete(app.fig) ;
    end  % function

    % Create UIFigure and components
    function createComponents(app)
      % Create the figure and all its child controls, laid out via
      % nested uigridlayouts: an outer 7-row column, where each
      % "Number of X" / question section is itself a 2-row column
      % (label-and-control row, then details label).

      % Layout constants
      figWidth = 450 ;
      initialFigHeight = 800 ;  % oversized; shrunk to fit content at the end
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
      copySettingsOverlapAmount = 6 ;  % how far the Copy Settings button overlaps into the Has Body Tracking details

      % Create the figure
      app.fig = uifigure('Visible', 'off') ;
      app.fig.Color = bgColor ;
      app.fig.Position = [100 100 figWidth initialFigHeight] ;
      app.fig.Name = 'Project Setup' ;
      app.fig.Resize = 'off' ;
      app.fig.CloseRequestFcn = app.createCallbackFcn(@didRequestClose, true) ;
      app.fig.HandleVisibility = 'callback' ;
      app.fig.Tag = 'project_setup_fig' ;

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

      app.project_name_label = uilabel(projectNameRow) ;
      app.project_name_label.Text = 'Project Name' ;
      app.project_name_label.FontSize = bigFontSize ;
      app.project_name_label.FontColor = labelColor ;
      app.project_name_label.BackgroundColor = bgColor ;

      app.project_name_edit = uieditfield(projectNameRow, 'text') ;
      app.project_name_edit.ValueChangedFcn = app.createCallbackFcn(@project_name_edit_Callback, true) ;
      app.project_name_edit.FontSize = bigFontSize ;
      app.project_name_edit.FontColor = fieldFontColor ;
      app.project_name_edit.BackgroundColor = fieldBgColor ;

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

      app.number_of_keypoints_label = uilabel(keypointsRow) ;
      app.number_of_keypoints_label.Text = 'Number of Keypoints' ;
      app.number_of_keypoints_label.FontSize = bigFontSize ;
      app.number_of_keypoints_label.FontColor = labelColor ;
      app.number_of_keypoints_label.BackgroundColor = bgColor ;

      app.number_of_keypoints_edit = uieditfield(keypointsRow, 'text') ;
      app.number_of_keypoints_edit.ValueChangedFcn = app.createCallbackFcn(@number_of_points_edit_Callback, true) ;
      app.number_of_keypoints_edit.HorizontalAlignment = 'right' ;
      app.number_of_keypoints_edit.FontSize = bigFontSize ;
      app.number_of_keypoints_edit.FontColor = fieldFontColor ;
      app.number_of_keypoints_edit.BackgroundColor = fieldBgColor ;
      app.number_of_keypoints_edit.Value = '12' ;

      app.number_of_keypoints_details_label = uilabel(keypointsSection) ;
      app.number_of_keypoints_details_label.Text = 'Number of keypoints to label for each animal' ;
      app.number_of_keypoints_details_label.WordWrap = 'on' ;
      app.number_of_keypoints_details_label.VerticalAlignment = 'top' ;
      app.number_of_keypoints_details_label.FontSize = smallFontSize ;
      app.number_of_keypoints_details_label.FontColor = labelColor ;
      app.number_of_keypoints_details_label.BackgroundColor = bgColor ;

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

      app.number_of_views_label = uilabel(viewsRow) ;
      app.number_of_views_label.Text = 'Number of Views' ;
      app.number_of_views_label.FontSize = bigFontSize ;
      app.number_of_views_label.FontColor = labelColor ;
      app.number_of_views_label.BackgroundColor = bgColor ;

      app.number_of_views_edit = uieditfield(viewsRow, 'text') ;
      app.number_of_views_edit.ValueChangedFcn = app.createCallbackFcn(@number_of_views_edit_Callback, true) ;
      app.number_of_views_edit.HorizontalAlignment = 'right' ;
      app.number_of_views_edit.FontSize = bigFontSize ;
      app.number_of_views_edit.FontColor = fieldFontColor ;
      app.number_of_views_edit.BackgroundColor = fieldBgColor ;
      app.number_of_views_edit.Value = '1' ;

      app.number_of_views_details_label = uilabel(viewsSection) ;
      app.number_of_views_details_label.Text = 'APT can do 3D labeling and tracking from multiple calibrated cameras. Enter 1 if animals were imaged from just one camera. Otherwise, enter the number of synced cameras recording the animals.' ;
      app.number_of_views_details_label.WordWrap = 'on' ;
      app.number_of_views_details_label.VerticalAlignment = 'top' ;
      app.number_of_views_details_label.FontSize = smallFontSize ;
      app.number_of_views_details_label.FontColor = labelColor ;
      app.number_of_views_details_label.BackgroundColor = bgColor ;

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

      app.multiple_animals_label = uilabel(multipleAnimalsRow) ;
      app.multiple_animals_label.Text = 'Multiple Animals?' ;
      app.multiple_animals_label.FontSize = bigFontSize ;
      app.multiple_animals_label.FontColor = labelColor ;
      app.multiple_animals_label.BackgroundColor = bgColor ;

      % Center multiple_animals_checkbox horizontally within the numberFieldWidth-wide cell so
      % it aligns with the centered "12" / "1" in the number fields above.
      multiple_animals_checkboxCell = uigridlayout(multipleAnimalsRow, [1 3]) ;
      multiple_animals_checkboxCell.ColumnWidth = {'1x', 'fit', '1x'} ;
      multiple_animals_checkboxCell.RowHeight = {'fit'} ;
      multiple_animals_checkboxCell.Padding = [0 0 0 0] ;
      multiple_animals_checkboxCell.ColumnSpacing = 0 ;
      multiple_animals_checkboxCell.BackgroundColor = bgColor ;

      app.multiple_animals_checkbox = uicheckbox(multiple_animals_checkboxCell) ;
      app.multiple_animals_checkbox.Layout.Column = 2 ;
      app.multiple_animals_checkbox.Text = '' ;
      app.multiple_animals_checkbox.FontSize = bigFontSize ;
      app.multiple_animals_checkbox.FontColor = labelColor ;

      app.multiple_animals_details_label = uilabel(multipleAnimalsSection) ;
      app.multiple_animals_details_label.Text = 'Check this box if there are multiple animals visible in any video frames. Otherwise, APT will assume there is just one animal visible per frame.' ;
      app.multiple_animals_details_label.WordWrap = 'on' ;
      app.multiple_animals_details_label.VerticalAlignment = 'top' ;
      app.multiple_animals_details_label.FontSize = smallFontSize ;
      app.multiple_animals_details_label.FontColor = labelColor ;
      app.multiple_animals_details_label.BackgroundColor = bgColor ;

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

      app.has_body_tracking_label = uilabel(hasBodyTrackingRow) ;
      app.has_body_tracking_label.Text = 'Has Body Tracking?' ;
      app.has_body_tracking_label.FontSize = bigFontSize ;
      app.has_body_tracking_label.FontColor = labelColor ;
      app.has_body_tracking_label.BackgroundColor = bgColor ;

      has_body_tracking_checkboxCell = uigridlayout(hasBodyTrackingRow, [1 3]) ;
      has_body_tracking_checkboxCell.ColumnWidth = {'1x', 'fit', '1x'} ;
      has_body_tracking_checkboxCell.RowHeight = {'fit'} ;
      has_body_tracking_checkboxCell.Padding = [0 0 0 0] ;
      has_body_tracking_checkboxCell.ColumnSpacing = 0 ;
      has_body_tracking_checkboxCell.BackgroundColor = bgColor ;

      app.has_body_tracking_checkbox = uicheckbox(has_body_tracking_checkboxCell) ;
      app.has_body_tracking_checkbox.Layout.Column = 2 ;
      app.has_body_tracking_checkbox.Text = '' ;
      app.has_body_tracking_checkbox.FontSize = 24 ;
      app.has_body_tracking_checkbox.FontColor = labelColor ;

      app.has_body_tracking_details_label = uilabel(hasBodyTrackingSection) ;
      app.has_body_tracking_details_label.Text = 'APT can do pose tracking on top of body tracking from an algorithm like FlyTracker or Ctrax. Check this box if you have already tracked the centroids and orientations of your animals and want to base pose tracking on those trajectories. If so, a trajectory file is input with each video.' ;
      app.has_body_tracking_details_label.WordWrap = 'on' ;
      app.has_body_tracking_details_label.VerticalAlignment = 'top' ;
      app.has_body_tracking_details_label.FontSize = smallFontSize ;
      app.has_body_tracking_details_label.FontColor = labelColor ;
      app.has_body_tracking_details_label.BackgroundColor = bgColor ;

      % Row 6 is reserved empty space; copy_settings_from_button is added below as
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

      app.create_project_button = uibutton(bottomButtonsRow, 'push') ;
      app.create_project_button.Layout.Column = 2 ;
      app.create_project_button.ButtonPushedFcn = app.createCallbackFcn(@create_project_button_Callback, true) ;
      app.create_project_button.BackgroundColor = fieldBgColor ;
      app.create_project_button.FontSize = bigFontSize ;
      app.create_project_button.FontColor = fieldFontColor ;
      app.create_project_button.Text = 'Create Project' ;

      app.cancel_button = uibutton(bottomButtonsRow, 'push') ;
      app.cancel_button.Layout.Column = 4 ;
      app.cancel_button.ButtonPushedFcn = app.createCallbackFcn(@cancel_button_Callback, true) ;
      app.cancel_button.BackgroundColor = fieldBgColor ;
      app.cancel_button.FontSize = bigFontSize ;
      app.cancel_button.FontColor = fieldFontColor ;
      app.cancel_button.Text = 'Cancel' ;

      % Show the figure so uigridlayout resolves to real pixel positions,
      % then resize the figure to fit the natural content height.
      app.fig.Visible = 'on' ;
      shrinkFigureToFitContentBang(app.fig, outerGrid) ;

      % Copy Settings... button.  Created as a direct child of the figure
      % (not inside outerGrid) so it can overlap into the Has Body Tracking
      % details area above, matching the look of the original GUIDE layout.
      % Drawn on top of outerGrid because it is created later.
      app.copy_settings_from_button = uibutton(app.fig, 'push') ;
      app.copy_settings_from_button.ButtonPushedFcn = app.createCallbackFcn(@copy_settings_from_button_Callback, true) ;
      app.copy_settings_from_button.BackgroundColor = fieldBgColor ;
      app.copy_settings_from_button.FontSize = bigFontSize ;
      app.copy_settings_from_button.FontColor = labelColor ;
      app.copy_settings_from_button.Tooltip = 'Copy settings from an existing project' ;
      app.copy_settings_from_button.Text = 'Copy Settings...' ;
      % Position derived from the known outerGrid layout.  Bottom of the Has
      % Body Tracking details label, measured from the figure bottom, is:
      %   marginWidth + row 7 + spacing + row 6 + spacing
      detailsBottomY = marginWidth + buttonHeight + outerGridRowSpacing + buttonHeight + outerGridRowSpacing ;
      copySettingsButtonX = (figWidth - marginWidth) - copySettingsButtonWidth ;
      copySettingsButtonY = detailsBottomY + copySettingsOverlapAmount - buttonHeight ;
      app.copy_settings_from_button.Position = [copySettingsButtonX, copySettingsButtonY, copySettingsButtonWidth, buttonHeight] ;
    end  % function
  end  % methods (Access = private)

  methods(Static)
    function result = patchCfg(cfg, projectName, keypointCount, viewCount, hasBodyTracking, multipleAnimals)
      result = cfg ;
      result.NumViews = viewCount ;
      result.NumLabelPoints = keypointCount ;

      viewNameCount = numel(result.ViewNames) ;
      if viewNameCount > viewCount
        result.ViewNames = result.ViewNames(1:viewCount) ;
      elseif viewNameCount < viewCount
        result.ViewNames(viewNameCount+1:viewCount) = {''} ;
      end
      pointNameCount = numel(result.LabelPointNames) ;
      if pointNameCount > keypointCount
        result.LabelPointNames = result.LabelPointNames(1:keypointCount) ;
      elseif pointNameCount < keypointCount
        result.LabelPointNames(pointNameCount+1:keypointCount) = {''} ;
      end
      result.View = augmentOrTruncateVector(result.View(:), viewCount) ;

      result.Trx.HasTrx = hasBodyTracking ;
      result.MultiAnimal = multipleAnimals ;
      isMultiAnimal = result.MultiAnimal && ~result.Trx.HasTrx ;
      if isMultiAnimal
        result.LabelMode = LabelMode.MULTIANIMAL ;
      else
        result.LabelMode = LabelMode.SEQUENTIAL ;
      end
      result.Track.Enable = true ;
      % propertiesGUI used to leave View fields as empty strings even when
      % they were meant to be numeric; coerce them back here.
      fieldsToDoublify = {'Gamma', 'FigurePos', 'AxisLim', 'InvertMovie', 'AxFontSize', 'ShowAxTicks', 'ShowGrid'} ;
      for i = 1:numel(result.View)
        result.View(i) = structLeavesStr2Double(result.View(i), fieldsToDoublify) ;
      end
      result.ProjectName = projectName ;      
    end
  end
end  % classdef
