classdef ProjectSetup < handle
  % Widget properties
  properties (Access = private, Transient)
    fig_                                  matlab.ui.Figure
    project_name_label_                   matlab.ui.control.Label
    project_name_edit_                    matlab.ui.control.EditField
    number_of_keypoints_label_            matlab.ui.control.Label
    number_of_keypoints_edit_             matlab.ui.control.EditField
    number_of_keypoints_details_label_    matlab.ui.control.Label
    number_of_views_label_                matlab.ui.control.Label
    number_of_views_edit_                 matlab.ui.control.EditField
    number_of_views_details_label_        matlab.ui.control.Label
    multiple_animals_label_               matlab.ui.control.Label
    multiple_animals_checkbox_            matlab.ui.control.CheckBox
    multiple_animals_details_label_       matlab.ui.control.Label
    has_body_tracking_label_              matlab.ui.control.Label
    has_body_tracking_checkbox_           matlab.ui.control.CheckBox
    has_body_tracking_details_label_      matlab.ui.control.Label
    copy_settings_from_button_            matlab.ui.control.Button
    create_project_button_                matlab.ui.control.Button
    cancel_button_                        matlab.ui.control.Button
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
    function obj = ProjectSetup(varargin)
      % Build and show the ProjectSetup dialog.  Returns once the figure
      % is shown and populated; the caller is responsible for blocking
      % (e.g. via uiwait(obj.fig_)) and for reading obj.output afterwards.
      %
      %   obj = ProjectSetup() ;
      %   obj = ProjectSetup(hParentFig) ;  % centered on hParentFig
      if numel(varargin) >= 1
        hParentFig = varargin{1} ;
        if ~ishandle(hParentFig)
          error('ProjectSetup:arg', 'Expected argument to be a figure handle.') ;
        end
      else
        hParentFig = [] ;
      end
      obj.createAndLayoutComponents_(hParentFig) ;
      cfg = Labeler.cfgGetLastProjectConfigNoView() ;
      obj.setCfg_(cfg) ;
      waitForFigureToSync(obj.fig_) ;  % block until the figure is actually visible
    end  % function

    function delete(obj)
      % Delete the underlying figure when the obj is deleted.
      delete(obj.fig_) ;
    end  % function
    
    function result = get.output(obj)
      % Getter for output: the project-config struct chosen by the user,
      % or [] if the dialog was cancelled.
      result = obj.output_ ;
    end  % function

    function uiwait(obj)
      % Block the caller until the dialog is closed.
      uiwait(obj.fig_) ;
    end  % function
  end  % methods
  
  methods (Access = private)
    function result = generateFinalConfig_(obj)
      % Generate a config struct from the current object state.  Takes
      % obj.cfg_ as a base, brings its variable-length fields (ViewNames,
      % LabelPointNames, View) in line with the current view/point counts,
      % then overlays the values from the UI.
      storedCfg = obj.cfg_ ;
      projectName = obj.project_name_edit_.Value ;
      keypointCount = str2double(obj.number_of_keypoints_edit_.Value) ;
      viewCount = str2double(obj.number_of_views_edit_.Value) ;
      hasBodyTracking = obj.has_body_tracking_checkbox_.Value ;
      multipleAnimals = obj.multiple_animals_checkbox_.Value ;
      result = ProjectSetup.patchCfg(storedCfg, projectName, keypointCount, viewCount, ...
                                     hasBodyTracking, multipleAnimals) ;
    end  % function

    function setCfg_(obj, cfg)
      % Set the given config struct on the controls and on internal state.
      obj.cfg_ = cfg ;
      obj.number_of_views_edit_.Value = num2str(cfg.NumViews) ;
      obj.number_of_keypoints_edit_.Value = num2str(cfg.NumLabelPoints) ;
      obj.has_body_tracking_checkbox_.Value = cfg.Trx.HasTrx ;
      obj.multiple_animals_checkbox_.Value = cfg.MultiAnimal ;
    end  % function

    function number_of_points_edit_actuated_(obj, source, event)  %#ok<INUSD>
      % Value-changed handler for the keypoint-count edit field.
      rawValue = str2double(obj.number_of_keypoints_edit_.Value) ;
      if ~(floor(rawValue) == rawValue && rawValue >= 1)
        obj.number_of_keypoints_edit_.Value = event.PreviousValue ;
      end
    end  % function

    function number_of_views_edit_actuated_(obj, source, event)  %#ok<INUSD>
      % Value-changed handler for the view-count edit field.
      rawValue = str2double(obj.number_of_views_edit_.Value) ;
      if ~(floor(rawValue) == rawValue && rawValue >= 1)
        obj.number_of_views_edit_.Value = event.PreviousValue ;
      end
      viewCount = str2double(obj.number_of_views_edit_.Value) ;
      switch viewCount
        case 1
          obj.has_body_tracking_checkbox_.Enable = 'on' ;
          obj.multiple_animals_checkbox_.Enable = 'on' ;
        otherwise
          obj.has_body_tracking_checkbox_.Value = false ;
          obj.multiple_animals_checkbox_.Value = false ;
          obj.has_body_tracking_checkbox_.Enable = 'off' ;
          obj.multiple_animals_checkbox_.Enable = 'off' ;
      end
    end  % function

    function project_name_edit_actuated_(obj, source, event)  %#ok<INUSD>
      % Value-changed handler for the project-name edit field.
      name = obj.project_name_edit_.Value ;
      if ~all(isstrprop(name, 'alphanum'))
        % This unfortunately invalidates _ also.  Checking for it seems more
        % work than worth.  MK 20220913
        warndlg('Name should have only alphanumeric characters') ;
        obj.project_name_edit_.Value = event.PreviousValue  ;
      end
    end  % function

    function didRequestClose_(obj, source, event)  %#ok<INUSD>
      % Close-request function for the main figure.
      obj.output_ = [] ;
      delete(obj.fig_) ;
    end  % function

    function cancel_button_actuated_(obj, source, event)  %#ok<INUSD>
      % Button-pushed function for the Cancel button.
      obj.output_ = [] ;
      delete(obj.fig_) ;
    end  % function

    function copy_settings_from_button_actuated_(obj, source, event)  %#ok<INUSD>
      % Button-pushed function for the Copy Settings From... button.
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
      obj.setCfg_(cfg) ;
    end  % function

    function create_project_button_actuated_(obj, source, event)  %#ok<INUSD>
      % Button-pushed function for the Create Project button.
      cfg = obj.generateFinalConfig_() ;
      obj.output_ = cfg ;
      delete(obj.fig_) ;
    end  % function

    function createAndLayoutComponents_(obj, hParentFig)
      % Create UIFigure and components.
      % Create the figure and all its child controls, laid out via
      % nested uigridlayouts: an outer 7-row column, where each
      % "Number of X" / question section is itself a 2-row column
      % (label-and-control row, then details label).

      % Layout constants
      figWidth = 450 ;
      initialFigHeight = 700 ;  % oversized; shrunk to fit content at the end
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

      % Create the figure
      obj.fig_ = uifigure('Visible', 'off') ;
      obj.fig_.Color = bgColor ;
      obj.fig_.Position = [100 100 figWidth initialFigHeight] ;
      obj.fig_.Name = 'Project Setup' ;
      obj.fig_.Resize = 'off' ;
      obj.fig_.CloseRequestFcn = @(src, evt) obj.didRequestClose_(src, evt) ;
      obj.fig_.HandleVisibility = 'callback' ;
      obj.fig_.Tag = 'project_setup_fig' ;
      if ~isempty(hParentFig)
        centerOnParentFigure(obj.fig_, hParentFig) ;
      end

      % Outer column: 8 rows.  Rows 1-5 are content sections (sized 'fit').
      % Row 6 holds the right-aligned Copy Settings button.  Row 7 is an
      % empty spacer.  Row 8 holds the bottom buttons.
      outerGrid = uigridlayout(obj.fig_, [8 1]) ;
      outerGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', buttonHeight, buttonHeight, buttonHeight} ;
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

      obj.project_name_label_ = uilabel(projectNameRow) ;
      obj.project_name_label_.Tag = 'project_name_label_' ;
      obj.project_name_label_.Text = 'Project Name' ;
      obj.project_name_label_.FontSize = bigFontSize ;
      obj.project_name_label_.FontColor = labelColor ;
      obj.project_name_label_.BackgroundColor = bgColor ;

      obj.project_name_edit_ = uieditfield(projectNameRow, 'text') ;
      obj.project_name_edit_.Tag = 'project_name_edit_' ;
      obj.project_name_edit_.ValueChangedFcn = @(src, evt) obj.project_name_edit_actuated_(src, evt) ;
      obj.project_name_edit_.FontSize = bigFontSize ;
      obj.project_name_edit_.FontColor = fieldFontColor ;
      obj.project_name_edit_.BackgroundColor = fieldBgColor ;

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

      obj.number_of_keypoints_label_ = uilabel(keypointsRow) ;
      obj.number_of_keypoints_label_.Tag = 'number_of_keypoints_label_' ;
      obj.number_of_keypoints_label_.Text = 'Number of Keypoints' ;
      obj.number_of_keypoints_label_.FontSize = bigFontSize ;
      obj.number_of_keypoints_label_.FontColor = labelColor ;
      obj.number_of_keypoints_label_.BackgroundColor = bgColor ;

      obj.number_of_keypoints_edit_ = uieditfield(keypointsRow, 'text') ;
      obj.number_of_keypoints_edit_.Tag = 'number_of_keypoints_edit_' ;
      obj.number_of_keypoints_edit_.ValueChangedFcn = @(src, evt) obj.number_of_points_edit_actuated_(src, evt) ;
      obj.number_of_keypoints_edit_.HorizontalAlignment = 'right' ;
      obj.number_of_keypoints_edit_.FontSize = bigFontSize ;
      obj.number_of_keypoints_edit_.FontColor = fieldFontColor ;
      obj.number_of_keypoints_edit_.BackgroundColor = fieldBgColor ;
      obj.number_of_keypoints_edit_.Value = '12' ;

      obj.number_of_keypoints_details_label_ = uilabel(keypointsSection) ;
      obj.number_of_keypoints_details_label_.Tag = 'number_of_keypoints_details_label_' ;
      obj.number_of_keypoints_details_label_.Text = 'Number of keypoints to label for each animal' ;
      obj.number_of_keypoints_details_label_.WordWrap = 'on' ;
      obj.number_of_keypoints_details_label_.VerticalAlignment = 'top' ;
      obj.number_of_keypoints_details_label_.FontSize = smallFontSize ;
      obj.number_of_keypoints_details_label_.FontColor = labelColor ;
      obj.number_of_keypoints_details_label_.BackgroundColor = bgColor ;

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

      obj.number_of_views_label_ = uilabel(viewsRow) ;
      obj.number_of_views_label_.Tag = 'number_of_views_label_' ;
      obj.number_of_views_label_.Text = 'Number of Views' ;
      obj.number_of_views_label_.FontSize = bigFontSize ;
      obj.number_of_views_label_.FontColor = labelColor ;
      obj.number_of_views_label_.BackgroundColor = bgColor ;

      obj.number_of_views_edit_ = uieditfield(viewsRow, 'text') ;
      obj.number_of_views_edit_.Tag = 'number_of_views_edit_' ;
      obj.number_of_views_edit_.ValueChangedFcn = @(src, evt) obj.number_of_views_edit_actuated_(src, evt) ;
      obj.number_of_views_edit_.HorizontalAlignment = 'right' ;
      obj.number_of_views_edit_.FontSize = bigFontSize ;
      obj.number_of_views_edit_.FontColor = fieldFontColor ;
      obj.number_of_views_edit_.BackgroundColor = fieldBgColor ;
      obj.number_of_views_edit_.Value = '1' ;

      obj.number_of_views_details_label_ = uilabel(viewsSection) ;
      obj.number_of_views_details_label_.Tag = 'number_of_views_details_label_' ;
      obj.number_of_views_details_label_.Text = 'APT can do 3D labeling and tracking from multiple calibrated cameras. Enter 1 if animals were imaged from just one camera. Otherwise, enter the number of synced cameras recording the animals.' ;
      obj.number_of_views_details_label_.WordWrap = 'on' ;
      obj.number_of_views_details_label_.VerticalAlignment = 'top' ;
      obj.number_of_views_details_label_.FontSize = smallFontSize ;
      obj.number_of_views_details_label_.FontColor = labelColor ;
      obj.number_of_views_details_label_.BackgroundColor = bgColor ;

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

      obj.multiple_animals_label_ = uilabel(multipleAnimalsRow) ;
      obj.multiple_animals_label_.Tag = 'multiple_animals_label_' ;
      obj.multiple_animals_label_.Text = 'Multiple Animals?' ;
      obj.multiple_animals_label_.FontSize = bigFontSize ;
      obj.multiple_animals_label_.FontColor = labelColor ;
      obj.multiple_animals_label_.BackgroundColor = bgColor ;

      % Center multiple_animals_checkbox_ horizontally within the numberFieldWidth-wide cell so
      % it aligns with the centered "12" / "1" in the number fields above.
      multiple_animals_checkbox_Cell = uigridlayout(multipleAnimalsRow, [1 3]) ;
      multiple_animals_checkbox_Cell.ColumnWidth = {'1x', 'fit', '1x'} ;
      multiple_animals_checkbox_Cell.RowHeight = {'fit'} ;
      multiple_animals_checkbox_Cell.Padding = [0 0 0 8] ;  % top-pad to vertically align glyph with the adjacent label text
      multiple_animals_checkbox_Cell.ColumnSpacing = 0 ;
      multiple_animals_checkbox_Cell.BackgroundColor = bgColor ;

      obj.multiple_animals_checkbox_ = uicheckbox(multiple_animals_checkbox_Cell) ;
      obj.multiple_animals_checkbox_.Tag = 'multiple_animals_checkbox_' ;
      obj.multiple_animals_checkbox_.Layout.Column = 2 ;
      obj.multiple_animals_checkbox_.Text = '' ;
      obj.multiple_animals_checkbox_.FontSize = bigFontSize ;
      obj.multiple_animals_checkbox_.FontColor = labelColor ;

      obj.multiple_animals_details_label_ = uilabel(multipleAnimalsSection) ;
      obj.multiple_animals_details_label_.Tag = 'multiple_animals_details_label_' ;
      obj.multiple_animals_details_label_.Text = 'Check this box if there are multiple animals visible in any video frames. Otherwise, APT will assume there is just one animal visible per frame.' ;
      obj.multiple_animals_details_label_.WordWrap = 'on' ;
      obj.multiple_animals_details_label_.VerticalAlignment = 'top' ;
      obj.multiple_animals_details_label_.FontSize = smallFontSize ;
      obj.multiple_animals_details_label_.FontColor = labelColor ;
      obj.multiple_animals_details_label_.BackgroundColor = bgColor ;

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

      obj.has_body_tracking_label_ = uilabel(hasBodyTrackingRow) ;
      obj.has_body_tracking_label_.Tag = 'has_body_tracking_label_' ;
      obj.has_body_tracking_label_.Text = 'Has Body Tracking?' ;
      obj.has_body_tracking_label_.FontSize = bigFontSize ;
      obj.has_body_tracking_label_.FontColor = labelColor ;
      obj.has_body_tracking_label_.BackgroundColor = bgColor ;

      has_body_tracking_checkbox_Cell = uigridlayout(hasBodyTrackingRow, [1 3]) ;
      has_body_tracking_checkbox_Cell.ColumnWidth = {'1x', 'fit', '1x'} ;
      has_body_tracking_checkbox_Cell.RowHeight = {'fit'} ;
      has_body_tracking_checkbox_Cell.Padding = [0 0 0 8] ;  % top-pad to vertically align glyph with the adjacent label text
      has_body_tracking_checkbox_Cell.ColumnSpacing = 0 ;
      has_body_tracking_checkbox_Cell.BackgroundColor = bgColor ;

      obj.has_body_tracking_checkbox_ = uicheckbox(has_body_tracking_checkbox_Cell) ;
      obj.has_body_tracking_checkbox_.Tag = 'has_body_tracking_checkbox_' ;
      obj.has_body_tracking_checkbox_.Layout.Column = 2 ;
      obj.has_body_tracking_checkbox_.Text = '' ;
      obj.has_body_tracking_checkbox_.FontSize = 24 ;
      obj.has_body_tracking_checkbox_.FontColor = labelColor ;

      obj.has_body_tracking_details_label_ = uilabel(hasBodyTrackingSection) ;
      obj.has_body_tracking_details_label_.Tag = 'has_body_tracking_details_label_' ;
      obj.has_body_tracking_details_label_.Text = 'APT can do pose tracking on top of body tracking from an algorithm like FlyTracker or Ctrax. Check this box if you have already tracked the centroids and orientations of your animals and want to base pose tracking on those trajectories. If so, a trajectory file is input with each video.' ;
      obj.has_body_tracking_details_label_.WordWrap = 'on' ;
      obj.has_body_tracking_details_label_.VerticalAlignment = 'top' ;
      obj.has_body_tracking_details_label_.FontSize = smallFontSize ;
      obj.has_body_tracking_details_label_.FontColor = labelColor ;
      obj.has_body_tracking_details_label_.BackgroundColor = bgColor ;

      % Row 6: the Copy Settings... button, right-aligned in its own grid
      % row.  (It used to float over the grid with a hard-coded overlap into
      % the Has Body Tracking details area, which broke whenever font metrics
      % wrapped that text differently.)
      copySettingsRow = uigridlayout(outerGrid, [1 2]) ;
      copySettingsRow.ColumnWidth = {'1x', copySettingsButtonWidth} ;
      copySettingsRow.RowHeight = {buttonHeight} ;
      copySettingsRow.Padding = [0 0 0 0] ;
      copySettingsRow.ColumnSpacing = 0 ;
      copySettingsRow.BackgroundColor = bgColor ;

      obj.copy_settings_from_button_ = uibutton(copySettingsRow, 'push') ;
      obj.copy_settings_from_button_.Tag = 'copy_settings_from_button_' ;
      obj.copy_settings_from_button_.Layout.Column = 2 ;
      obj.copy_settings_from_button_.ButtonPushedFcn = @(src, evt) obj.copy_settings_from_button_actuated_(src, evt) ;
      obj.copy_settings_from_button_.BackgroundColor = fieldBgColor ;
      obj.copy_settings_from_button_.FontSize = bigFontSize ;
      obj.copy_settings_from_button_.FontColor = labelColor ;
      obj.copy_settings_from_button_.Tooltip = 'Copy settings from an existing project' ;
      obj.copy_settings_from_button_.Text = 'Copy Settings...' ;

      % Row 7: empty spacer above the bottom buttons.  A child grid rather
      % than a bare row: shrinkFigureToFitContentBang sizes rows by their
      % children's rendered heights, so a childless row would count as zero
      % and the figure would come out a row short.
      spacerRow = uigridlayout(outerGrid, [1 1]) ;
      spacerRow.Padding = [0 0 0 0] ;
      spacerRow.BackgroundColor = bgColor ;

      % Row 8: Bottom buttons (Create Project + Cancel, centered, with
      % an explicit gap between the two buttons) -------------------
      betweenBottomButtonsWidth = 20 ;
      bottomButtonsRow = uigridlayout(outerGrid, [1 5]) ;
      bottomButtonsRow.ColumnWidth = {'1x', createProjectButtonWidth, betweenBottomButtonsWidth, cancelButtonWidth, '1x'} ;
      bottomButtonsRow.RowHeight = {'fit'} ;
      bottomButtonsRow.Padding = [0 0 0 0] ;
      bottomButtonsRow.ColumnSpacing = 0 ;
      bottomButtonsRow.BackgroundColor = bgColor ;

      obj.create_project_button_ = uibutton(bottomButtonsRow, 'push') ;
      obj.create_project_button_.Tag = 'create_project_button_' ;
      obj.create_project_button_.Layout.Column = 2 ;
      obj.create_project_button_.ButtonPushedFcn = @(src, evt) obj.create_project_button_actuated_(src, evt) ;
      obj.create_project_button_.BackgroundColor = fieldBgColor ;
      obj.create_project_button_.FontSize = bigFontSize ;
      obj.create_project_button_.FontColor = fieldFontColor ;
      obj.create_project_button_.Text = 'Create Project' ;

      obj.cancel_button_ = uibutton(bottomButtonsRow, 'push') ;
      obj.cancel_button_.Tag = 'cancel_button_' ;
      obj.cancel_button_.Layout.Column = 4 ;
      obj.cancel_button_.ButtonPushedFcn = @(src, evt) obj.cancel_button_actuated_(src, evt) ;
      obj.cancel_button_.BackgroundColor = fieldBgColor ;
      obj.cancel_button_.FontSize = bigFontSize ;
      obj.cancel_button_.FontColor = fieldFontColor ;
      obj.cancel_button_.Text = 'Cancel' ;

      % Show the figure so uigridlayout resolves to real pixel positions,
      % then resize the figure to fit the natural content height.
      obj.fig_.Visible = 'on' ;
      shrinkFigureToFitContentBang(obj.fig_, outerGrid) ;

      % Do a final centering
      if ~isempty(hParentFig)
        centerOnParentFigure(obj.fig_, hParentFig) ;
      end
    end  % function
  end  % methods (Access = private)

  methods (Static)
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
      result.ViewNames = result.ViewNames(:) ;  % Force column so downstream shape assumptions hold
      pointNameCount = numel(result.LabelPointNames) ;
      if pointNameCount > keypointCount
        result.LabelPointNames = result.LabelPointNames(1:keypointCount) ;
      elseif pointNameCount < keypointCount
        result.LabelPointNames(pointNameCount+1:keypointCount) = {''} ;
      end
      result.LabelPointNames = result.LabelPointNames(:) ;  % Force column; initFromConfig_ uses size(...,1) as the set count
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
      result.ProjectName = projectName ;
    end
  end
end  % classdef
