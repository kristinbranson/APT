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

  methods (Access = public)pbAdvanced
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
      % Create all the controls.

      % Some layout constants
      figWidth = 450 ;
      figHeight = 620 ;
      mainColumnMarginWidth = 25 ;
      mainColumnWidth = figWidth - 2*mainColumnMarginWidth ;
      bigLabelHeight = 24 ;
      pbCancelRightX = 399 ;  % right edge of the Cancel button; right-align Copy Settings to here
      bottomMarginHeight = 20 ;
      buttonHeight = 32 ;

      % Create figure1 and hide until all components are created
      app.fig = uifigure('Visible', 'off');
      app.fig.AutoResizeChildren = 'off';
      app.fig.Resize = 'off';
      app.fig.Color = [0 0.243 0.365];
      app.fig.Position = [100 100 figWidth figHeight];
      app.fig.Name = 'Project Setup';
      app.fig.CloseRequestFcn = app.createCallbackFcn(@figure1_CloseRequestFcn, true);
      app.fig.HandleVisibility = 'callback';
      app.fig.Tag = 'figure1';

      % Create labelProjectName
      app.labelProjectName = uilabel(app.fig);
      app.labelProjectName.Tag = 'text2';
      app.labelProjectName.BackgroundColor = [0 0.243 0.365];
      app.labelProjectName.VerticalAlignment = 'top';
      app.labelProjectName.WordWrap = 'on';
      app.labelProjectName.FontSize = 20;
      app.labelProjectName.FontColor = [0 1 1];
      app.labelProjectName.Position = [mainColumnMarginWidth 559 137 bigLabelHeight];
      app.labelProjectName.Text = 'Project Name';

      % Create etProjectName
      app.etProjectName = uieditfield(app.fig, 'text');
      app.etProjectName.ValueChangedFcn = app.createCallbackFcn(@etProjectName_Callback, true);
      app.etProjectName.Tag = 'etProjectName';
      app.etProjectName.FontSize = 20;
      app.etProjectName.FontColor = [0 0.980392156862745 0.819607843137255];
      app.etProjectName.BackgroundColor = [0 0 0];
      app.etProjectName.Position = [186 555 250 32];

      % Create labelNumberOfKeypoints
      app.labelNumberOfKeypoints = uilabel(app.fig);
      app.labelNumberOfKeypoints.Tag = 'labelNumberOfKeypoints';
      app.labelNumberOfKeypoints.BackgroundColor = [0 0.243 0.365];
      app.labelNumberOfKeypoints.VerticalAlignment = 'top';
      app.labelNumberOfKeypoints.WordWrap = 'on';
      app.labelNumberOfKeypoints.FontSize = 20;
      app.labelNumberOfKeypoints.FontColor = [0 1 1];
      app.labelNumberOfKeypoints.Position = [mainColumnMarginWidth 512 220 bigLabelHeight];
      app.labelNumberOfKeypoints.Text = 'Number of Keypoints';

      % Create etNumberOfPoints
      app.etNumberOfPoints = uieditfield(app.fig, 'text');
      app.etNumberOfPoints.ValueChangedFcn = app.createCallbackFcn(@etNumberOfPoints_Callback, true);
      app.etNumberOfPoints.Tag = 'etNumberOfPoints';
      app.etNumberOfPoints.HorizontalAlignment = 'center';
      app.etNumberOfPoints.FontSize = 20;
      app.etNumberOfPoints.FontColor = [0 0.980392156862745 0.819607843137255];
      app.etNumberOfPoints.BackgroundColor = [0 0 0];
      app.etNumberOfPoints.Position = [282 512 150 32];
      app.etNumberOfPoints.Value = '12';

      % Create labelNumberOfKeypointsDetails
      app.labelNumberOfKeypointsDetails = uilabel(app.fig);
      app.labelNumberOfKeypointsDetails.Tag = 'text_nkeypoints_description';
      app.labelNumberOfKeypointsDetails.BackgroundColor = [0 0.243 0.365];
      app.labelNumberOfKeypointsDetails.VerticalAlignment = 'top';
      app.labelNumberOfKeypointsDetails.WordWrap = 'on';
      app.labelNumberOfKeypointsDetails.FontSize = 16;
      app.labelNumberOfKeypointsDetails.FontColor = [0 1 1];
      app.labelNumberOfKeypointsDetails.Position = [27 488 400 24];
      app.labelNumberOfKeypointsDetails.Text = 'Number of keypoints to label for each animal';

      % Create labelNumberOfViews
      app.labelNumberOfViews = uilabel(app.fig);
      app.labelNumberOfViews.Tag = 'labelNumberOfViews';
      app.labelNumberOfViews.BackgroundColor = [0 0.243 0.365];
      app.labelNumberOfViews.VerticalAlignment = 'top';
      app.labelNumberOfViews.WordWrap = 'on';
      app.labelNumberOfViews.FontSize = 20;
      app.labelNumberOfViews.FontColor = [0 1 1];
      app.labelNumberOfViews.Position = [mainColumnMarginWidth 450 179 bigLabelHeight];
      app.labelNumberOfViews.Text = 'Number of Views';

      % Create etNumberOfViews
      app.etNumberOfViews = uieditfield(app.fig, 'text');
      app.etNumberOfViews.ValueChangedFcn = app.createCallbackFcn(@etNumberOfViews_Callback, true);
      app.etNumberOfViews.Tag = 'etNumberOfViews';
      app.etNumberOfViews.HorizontalAlignment = 'center';
      app.etNumberOfViews.FontSize = 20;
      app.etNumberOfViews.FontColor = [0 0.980392156862745 0.819607843137255];
      app.etNumberOfViews.BackgroundColor = [0 0 0];
      app.etNumberOfViews.Position = [282 452 150 32];
      app.etNumberOfViews.Value = '1';

      % Create labelNumberOfViewsDetails
      app.labelNumberOfViewsDetails = uilabel(app.fig);
      app.labelNumberOfViewsDetails.Tag = 'text_nviews_description';
      app.labelNumberOfViewsDetails.BackgroundColor = [0 0.243 0.365];
      app.labelNumberOfViewsDetails.VerticalAlignment = 'top';
      app.labelNumberOfViewsDetails.WordWrap = 'on';
      app.labelNumberOfViewsDetails.FontSize = 16;
      app.labelNumberOfViewsDetails.FontColor = [0 1 1];
      app.labelNumberOfViewsDetails.Position = [mainColumnMarginWidth 363 mainColumnWidth 85];
      app.labelNumberOfViewsDetails.Text = 'APT can do 3D labeling and tracking from multiple calibrated cameras. Enter 1 if animals were imaged from just one camera. Otherwise, enter the number of synced cameras recording the animals. ';

      % Create labelMultipleAnimals
      app.labelMultipleAnimals = uilabel(app.fig);
      app.labelMultipleAnimals.Tag = 'labelMultipleAnimals';
      app.labelMultipleAnimals.BackgroundColor = [0 0.243 0.365];
      app.labelMultipleAnimals.VerticalAlignment = 'top';
      app.labelMultipleAnimals.WordWrap = 'on';
      app.labelMultipleAnimals.FontSize = 20;
      app.labelMultipleAnimals.FontColor = [0 1 1];
      app.labelMultipleAnimals.Position = [mainColumnMarginWidth 336 202 bigLabelHeight];
      app.labelMultipleAnimals.Text = 'Multiple Animals?';

      % Create cbMA
      app.cbMA = uicheckbox(app.fig);
      app.cbMA.Tag = 'cbMA';
      app.cbMA.Text = '';
      app.cbMA.FontSize = 20;
      app.cbMA.FontColor = [0 1 1];
      app.cbMA.Position = [324 338 14 22];

      % Create labelMultipleAnimalsDetails
      app.labelMultipleAnimalsDetails = uilabel(app.fig);
      app.labelMultipleAnimalsDetails.Tag = 'text_multianimal_description';
      app.labelMultipleAnimalsDetails.BackgroundColor = [0 0.243 0.365];
      app.labelMultipleAnimalsDetails.VerticalAlignment = 'top';
      app.labelMultipleAnimalsDetails.WordWrap = 'on';
      app.labelMultipleAnimalsDetails.FontSize = 16;
      app.labelMultipleAnimalsDetails.FontColor = [0 1 1];
      app.labelMultipleAnimalsDetails.Position = [mainColumnMarginWidth 265 mainColumnWidth 67];
      app.labelMultipleAnimalsDetails.Text = 'Check this box if there are multiple animals visible in any video frames. Otherwise, APT will assume there is just one animal visible per frame.';

      % Create labelHasBodyTracking
      app.labelHasBodyTracking = uilabel(app.fig);
      app.labelHasBodyTracking.Tag = 'text_multitarget';
      app.labelHasBodyTracking.BackgroundColor = [0 0.243 0.365];
      app.labelHasBodyTracking.VerticalAlignment = 'top';
      app.labelHasBodyTracking.WordWrap = 'on';
      app.labelHasBodyTracking.FontSize = 20;
      app.labelHasBodyTracking.FontColor = [0 1 1];
      app.labelHasBodyTracking.Position = [mainColumnMarginWidth 235 206 bigLabelHeight];
      app.labelHasBodyTracking.Text = 'Has Body Tracking?';

      % Create cbHasTrx
      app.cbHasTrx = uicheckbox(app.fig);
      app.cbHasTrx.Tag = 'cbHasTrx';
      app.cbHasTrx.Text = '';
      app.cbHasTrx.FontSize = 24;
      app.cbHasTrx.FontColor = [0 1 1];
      app.cbHasTrx.Position = [323 237 24 24];

      % Create labelHasBodyTrackingDetails
      app.labelHasBodyTrackingDetails = uilabel(app.fig);
      app.labelHasBodyTrackingDetails.Tag = 'labelHasBodyTrackingDetails';
      app.labelHasBodyTrackingDetails.BackgroundColor = [0 0.243 0.365];
      app.labelHasBodyTrackingDetails.VerticalAlignment = 'top';
      app.labelHasBodyTrackingDetails.WordWrap = 'on';
      app.labelHasBodyTrackingDetails.FontSize = 16;
      app.labelHasBodyTrackingDetails.FontColor = [0 1 1];
      app.labelHasBodyTrackingDetails.Position = [mainColumnMarginWidth 110 mainColumnWidth 122];
      app.labelHasBodyTrackingDetails.Text = 'APT can do pose tracking on top of body tracking from an algorithm like FlyTracker or Ctrax. Check this box if you have already tracked the centroids and orientations of your animals and want to base pose tracking on those trajectories. If so, a trajectory file is input with each video.';

      % Create pbCopySettingsFrom
      % Create this after labelHasBodyTrackingDetails so it draws on top of
      % that label (which it overlaps slightly).
      app.pbCopySettingsFrom = uibutton(app.fig, 'push');
      app.pbCopySettingsFrom.ButtonPushedFcn = app.createCallbackFcn(@pbCopySettingsFrom_Callback, true);
      app.pbCopySettingsFrom.Tag = 'pbCopySettingsFrom';
      app.pbCopySettingsFrom.BackgroundColor = [0 0 0];
      app.pbCopySettingsFrom.FontSize = 20;
      app.pbCopySettingsFrom.FontColor = [0 1 1];
      app.pbCopySettingsFrom.Tooltip = 'Copy settings from an existing project';
      pbCopySettingsWidth = 180 ;
      pbCopySettingsRightX = figWidth - mainColumnMarginWidth ;
      pbCopySettingsX = pbCopySettingsRightX - pbCopySettingsWidth ;
      app.pbCopySettingsFrom.Position = [pbCopySettingsX 92 pbCopySettingsWidth buttonHeight];
      app.pbCopySettingsFrom.Text = 'Copy Settings...';

      % Bottom buttons have fixed widths and a fixed distance between, then that
      % group is centered in the figure.
      createProjectButtonWidth = 200 ;
      buttonSpacerWidth = 20 ;
      cancelButtonWidth = 150 ;
      bottomButtonGroupWidth = createProjectButtonWidth + buttonSpacerWidth + cancelButtonWidth ;
      bottomButtonGroupX = (figWidth-bottomButtonGroupWidth)/2 ;
      createProjectButtonX = bottomButtonGroupX ;
      cancelButtonX = createProjectButtonX + createProjectButtonWidth + buttonSpacerWidth ;      

      % Create pbCreateProject
      app.pbCreateProject = uibutton(app.fig, 'push');
      app.pbCreateProject.ButtonPushedFcn = app.createCallbackFcn(@pbCreateProject_Callback, true);
      app.pbCreateProject.Tag = 'pbCreateProject';
      app.pbCreateProject.BackgroundColor = [0 0 0];
      app.pbCreateProject.FontSize = 20;
      app.pbCreateProject.FontColor = [0 0.980392156862745 0.819607843137255];
      app.pbCreateProject.Position = [createProjectButtonX bottomMarginHeight createProjectButtonWidth buttonHeight];
      app.pbCreateProject.Text = 'Create Project';

      % Create pbCancel
      app.pbCancel = uibutton(app.fig, 'push');
      app.pbCancel.ButtonPushedFcn = app.createCallbackFcn(@pbCancel_Callback, true);
      app.pbCancel.Tag = 'pbCancel';
      app.pbCancel.BackgroundColor = [0 0 0];
      app.pbCancel.FontSize = 20;
      app.pbCancel.FontColor = [0 0.980392156862745 0.819607843137255];
      app.pbCancel.Position = [cancelButtonX bottomMarginHeight cancelButtonWidth buttonHeight];
      app.pbCancel.Text = 'Cancel';
      
      % Show the figure after all components are created
      app.fig.Visible = 'on';
    end  % function
  end  % methods (Access = private)
end  % classdef
