classdef ProjectSetup < matlab.apps.AppBase

  % Widget properties (emitted by App Designer Migration Tool)
  properties (Access = public)
    figure1                       matlab.ui.Figure
    text16                        matlab.ui.control.Label
    text_multianimal_description  matlab.ui.control.Label
    text_nviews_description       matlab.ui.control.Label
    text_nkeypoints_description   matlab.ui.control.Label
    landmarkMid                   matlab.ui.control.Label
    text12                        matlab.ui.control.Label
    cbMA                          matlab.ui.control.CheckBox
    text_multitarget              matlab.ui.control.Label
    cbHasTrx                      matlab.ui.control.CheckBox
    pbCopySettingsFrom            matlab.ui.control.Button
    landmarkRight                 matlab.ui.control.Label
    text8                         matlab.ui.control.Label
    pnlAdvanced                   matlab.ui.container.Panel
    pbCancel                      matlab.ui.control.Button
    pbCreateProject               matlab.ui.control.Button
    pbAdvanced                    matlab.ui.control.Button
    etNumberOfViews               matlab.ui.control.EditField
    etNumberOfPoints              matlab.ui.control.EditField
    etProjectName                 matlab.ui.control.EditField
    text4                         matlab.ui.control.Label
    text3                         matlab.ui.control.Label
    text2                         matlab.ui.control.Label
  end

  % Non-widget state (private in spirit; underscore suffix marks the intent)
  properties (Access = public, Transient)
    viewCount_  = 1
    pointCount_ = 1
    advancedOn_ = false
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
      createComponents(app) ;
      registerApp(app, app.figure1) ;
      opening_(app, varargin{:}) ;
      if nargout == 0
        clear app
      end
    end  % function

    function delete(app)
      % Delete the underlying figure when the app is deleted.
      delete(app.figure1) ;
    end  % function

    function advModeCollapse_(app)
      % Collapse the dialog so the advanced-properties panel is hidden.
      h1 = findall(app.figure1, '-property', 'Units') ;
      set(h1, 'Units', 'pixels') ;
      posMid = app.landmarkMid.Position ;
      posMid = posMid(1) + posMid(3)/2 ;
      pos = app.figure1.Position ;
      pos(3) = posMid ;
      app.figure1.Position = pos ;
      set(h1, 'Units', 'normalized') ;
      app.advancedOn_ = false ;
      app.pbAdvanced.Text = 'Advanced >' ;
    end  % function

    function advModeExpand_(app)
      % Expand the dialog so the advanced-properties panel is visible.
      h1 = findall(app.figure1, '-property', 'Units') ;
      set(h1, 'Units', 'pixels') ;
      posRight = app.landmarkRight.Position ;
      posRight = posRight(1) + posRight(3) ;
      pos = app.figure1.Position ;
      pos(3) = posRight ;
      app.figure1.Position = pos ;
      set(h1, 'Units', 'normalized') ;
      app.advancedOn_ = true ;
      app.pbAdvanced.Text = '< Basic' ;
    end  % function

    function advModeToggle_(app)
      % Toggle the dialog between expanded and collapsed advanced mode.
      if app.advancedOn_
        app.advModeCollapse_() ;
      else
        app.advModeExpand_() ;
      end
    end  % function

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
      % Initialize dialog state and block until the user closes it.
      %
      % Modal dialog.  Generates a project configuration struct.
      %
      %   app = ProjectSetup() ;
      %   app = ProjectSetup(hParentFig) ;  % centered on hParentFig
      movegui(app.figure1, 'onscreen') ;
      set(findall(app.figure1, '-property', 'Units'), 'Units', 'normalized') ;

      if numel(varargin) >= 1
        hParentFig = varargin{1} ;
        if ~ishandle(hParentFig)
          error('ProjectSetup:arg', 'Expected argument to be a figure handle.') ;
        end
        centerOnParentFigure(app.figure1, hParentFig) ;
      end

      cfg = Labeler.cfgGetLastProjectConfigNoView() ;
      app.setCurrentConfig_(cfg) ;
      app.advModeCollapse_() ;

      uiwait(app.figure1) ;
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
      if isequal(get(app.figure1, 'waitstatus'), 'waiting')
        % The dialog is still in uiwait; release it.
        uiresume(app.figure1) ;
      else
        delete(app.figure1) ;
      end
    end  % function

    % Button-pushed function for the Advanced/Basic toggle.
    function pbAdvanced_Callback(app, ~)
      app.advModeToggle_() ;
    end  % function

    % Button-pushed function for the Cancel button.
    function pbCancel_Callback(app, ~)
      app.output_ = [] ;
      close(app.figure1) ;
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
      close(app.figure1) ;
    end  % function
  end  % methods (Access = private)

  % Component initialization
  methods (Access = private)

    % Create UIFigure and components
    function createComponents(app)

      % Create figure1 and hide until all components are created
      app.figure1 = uifigure('Visible', 'off');
      app.figure1.Color = [0 0.243 0.365];
      app.figure1.Position = [680 889 854 621];
      app.figure1.Name = 'Project Setup';
      app.figure1.CloseRequestFcn = createCallbackFcn(app, @figure1_CloseRequestFcn, true);
      app.figure1.HandleVisibility = 'callback';
      app.figure1.Tag = 'figure1';

      % Create text2
      app.text2 = uilabel(app.figure1);
      app.text2.Tag = 'text2';
      app.text2.BackgroundColor = [0 0.243 0.365];
      app.text2.VerticalAlignment = 'top';
      app.text2.WordWrap = 'on';
      app.text2.FontSize = 20;
      app.text2.FontColor = [0 1 1];
      app.text2.Position = [25 559 137 24];
      app.text2.Text = 'Project Name';

      % Create text3
      app.text3 = uilabel(app.figure1);
      app.text3.Tag = 'text3';
      app.text3.BackgroundColor = [0 0.243 0.365];
      app.text3.VerticalAlignment = 'top';
      app.text3.WordWrap = 'on';
      app.text3.FontSize = 20;
      app.text3.FontColor = [0 1 1];
      app.text3.Position = [25 512 220 24];
      app.text3.Text = 'Number of Keypoints';

      % Create text4
      app.text4 = uilabel(app.figure1);
      app.text4.Tag = 'text4';
      app.text4.BackgroundColor = [0 0.243 0.365];
      app.text4.VerticalAlignment = 'top';
      app.text4.WordWrap = 'on';
      app.text4.FontSize = 20;
      app.text4.FontColor = [0 1 1];
      app.text4.Position = [25 450 179 24];
      app.text4.Text = 'Number of Views';

      % Create etProjectName
      app.etProjectName = uieditfield(app.figure1, 'text');
      app.etProjectName.ValueChangedFcn = createCallbackFcn(app, @etProjectName_Callback, true);
      app.etProjectName.Tag = 'etProjectName';
      app.etProjectName.FontSize = 20;
      app.etProjectName.FontColor = [0 0.980392156862745 0.819607843137255];
      app.etProjectName.BackgroundColor = [0 0 0];
      app.etProjectName.Position = [186 555 250 32];

      % Create etNumberOfPoints
      app.etNumberOfPoints = uieditfield(app.figure1, 'text');
      app.etNumberOfPoints.ValueChangedFcn = createCallbackFcn(app, @etNumberOfPoints_Callback, true);
      app.etNumberOfPoints.Tag = 'etNumberOfPoints';
      app.etNumberOfPoints.HorizontalAlignment = 'center';
      app.etNumberOfPoints.FontSize = 20;
      app.etNumberOfPoints.FontColor = [0 0.980392156862745 0.819607843137255];
      app.etNumberOfPoints.BackgroundColor = [0 0 0];
      app.etNumberOfPoints.Position = [282 511 150 32];
      app.etNumberOfPoints.Value = '12';

      % Create etNumberOfViews
      app.etNumberOfViews = uieditfield(app.figure1, 'text');
      app.etNumberOfViews.ValueChangedFcn = createCallbackFcn(app, @etNumberOfViews_Callback, true);
      app.etNumberOfViews.Tag = 'etNumberOfViews';
      app.etNumberOfViews.HorizontalAlignment = 'center';
      app.etNumberOfViews.FontSize = 20;
      app.etNumberOfViews.FontColor = [0 0.980392156862745 0.819607843137255];
      app.etNumberOfViews.BackgroundColor = [0 0 0];
      app.etNumberOfViews.Position = [282 452 150 32];
      app.etNumberOfViews.Value = '1';

      % Create pbAdvanced
      app.pbAdvanced = uibutton(app.figure1, 'push');
      app.pbAdvanced.ButtonPushedFcn = createCallbackFcn(app, @pbAdvanced_Callback, true);
      app.pbAdvanced.Tag = 'pbAdvanced';
      app.pbAdvanced.BackgroundColor = [0 0 0];
      app.pbAdvanced.FontSize = 20;
      app.pbAdvanced.FontColor = [0 1 1];
      app.pbAdvanced.Tooltip = 'Show advanced options';
      app.pbAdvanced.Position = [250 64 150 32];
      app.pbAdvanced.Text = 'Advanced >';

      % Create pbCreateProject
      app.pbCreateProject = uibutton(app.figure1, 'push');
      app.pbCreateProject.ButtonPushedFcn = createCallbackFcn(app, @pbCreateProject_Callback, true);
      app.pbCreateProject.Tag = 'pbCreateProject';
      app.pbCreateProject.BackgroundColor = [0 0 0];
      app.pbCreateProject.FontSize = 20;
      app.pbCreateProject.FontColor = [0 0.980392156862745 0.819607843137255];
      app.pbCreateProject.Position = [44 21 200 32];
      app.pbCreateProject.Text = 'Create Project';

      % Create pbCancel
      app.pbCancel = uibutton(app.figure1, 'push');
      app.pbCancel.ButtonPushedFcn = createCallbackFcn(app, @pbCancel_Callback, true);
      app.pbCancel.Tag = 'pbCancel';
      app.pbCancel.BackgroundColor = [0 0 0];
      app.pbCancel.FontSize = 20;
      app.pbCancel.FontColor = [0 0.980392156862745 0.819607843137255];
      app.pbCancel.Position = [249 21 150 32];
      app.pbCancel.Text = 'Cancel';

      % Create pnlAdvanced
      app.pnlAdvanced = uipanel(app.figure1);
      app.pnlAdvanced.ForegroundColor = [0 1 1];
      app.pnlAdvanced.BorderType = 'none';
      app.pnlAdvanced.BackgroundColor = [0 0.243 0.365];
      app.pnlAdvanced.Tag = 'pnlAdvanced';
      app.pnlAdvanced.FontSize = 13.3333333333333;
      app.pnlAdvanced.Position = [457 20 377 551];

      % Create text8
      app.text8 = uilabel(app.figure1);
      app.text8.Tag = 'text8';
      app.text8.BackgroundColor = [0 0.243 0.365];
      app.text8.VerticalAlignment = 'top';
      app.text8.WordWrap = 'on';
      app.text8.FontSize = 20;
      app.text8.FontAngle = 'italic';
      app.text8.FontColor = [0 1 1];
      app.text8.Position = [458 564 217 24];
      app.text8.Text = 'Advanced Properties';

      % Create landmarkRight
      app.landmarkRight = uilabel(app.figure1);
      app.landmarkRight.Tag = 'landmarkRight';
      app.landmarkRight.HorizontalAlignment = 'center';
      app.landmarkRight.VerticalAlignment = 'top';
      app.landmarkRight.WordWrap = 'on';
      app.landmarkRight.FontSize = 10.6666666666667;
      app.landmarkRight.Visible = 'off';
      app.landmarkRight.Position = [825 594 8 8];
      app.landmarkRight.Text = '';

      % Create pbCopySettingsFrom
      app.pbCopySettingsFrom = uibutton(app.figure1, 'push');
      app.pbCopySettingsFrom.ButtonPushedFcn = createCallbackFcn(app, @pbCopySettingsFrom_Callback, true);
      app.pbCopySettingsFrom.Tag = 'pbCopySettingsFrom';
      app.pbCopySettingsFrom.BackgroundColor = [0 0 0];
      app.pbCopySettingsFrom.FontSize = 20;
      app.pbCopySettingsFrom.FontColor = [0 1 1];
      app.pbCopySettingsFrom.Tooltip = 'Copy settings from an existing project';
      app.pbCopySettingsFrom.Position = [45 63 200 32];
      app.pbCopySettingsFrom.Text = 'Copy Settings...';

      % Create cbHasTrx
      app.cbHasTrx = uicheckbox(app.figure1);
      app.cbHasTrx.Tag = 'cbHasTrx';
      app.cbHasTrx.Text = '';
      app.cbHasTrx.FontSize = 24;
      app.cbHasTrx.FontColor = [0 1 1];
      app.cbHasTrx.Position = [323 237 24 24];

      % Create text_multitarget
      app.text_multitarget = uilabel(app.figure1);
      app.text_multitarget.Tag = 'text_multitarget';
      app.text_multitarget.BackgroundColor = [0 0.243 0.365];
      app.text_multitarget.VerticalAlignment = 'top';
      app.text_multitarget.WordWrap = 'on';
      app.text_multitarget.FontSize = 20;
      app.text_multitarget.FontColor = [0 1 1];
      app.text_multitarget.Position = [25 235 206 24];
      app.text_multitarget.Text = 'Has Body Tracking?';

      % Create cbMA
      app.cbMA = uicheckbox(app.figure1);
      app.cbMA.Tag = 'cbMA';
      app.cbMA.Text = '';
      app.cbMA.FontSize = 20;
      app.cbMA.FontColor = [0 1 1];
      app.cbMA.Position = [324 338 14 22];

      % Create text12
      app.text12 = uilabel(app.figure1);
      app.text12.Tag = 'text12';
      app.text12.BackgroundColor = [0 0.243 0.365];
      app.text12.VerticalAlignment = 'top';
      app.text12.WordWrap = 'on';
      app.text12.FontSize = 20;
      app.text12.FontColor = [0 1 1];
      app.text12.Position = [25 336 202 24];
      app.text12.Text = 'Multiple Animals?';

      % Create landmarkMid
      app.landmarkMid = uilabel(app.figure1);
      app.landmarkMid.Tag = 'landmarkMid';
      app.landmarkMid.HorizontalAlignment = 'center';
      app.landmarkMid.VerticalAlignment = 'top';
      app.landmarkMid.WordWrap = 'on';
      app.landmarkMid.FontSize = 10.6666666666667;
      app.landmarkMid.Visible = 'off';
      app.landmarkMid.Position = [446 439 13 14];
      app.landmarkMid.Text = '';

      % Create text_nkeypoints_description
      app.text_nkeypoints_description = uilabel(app.figure1);
      app.text_nkeypoints_description.Tag = 'text_nkeypoints_description';
      app.text_nkeypoints_description.BackgroundColor = [0 0.243 0.365];
      app.text_nkeypoints_description.VerticalAlignment = 'top';
      app.text_nkeypoints_description.WordWrap = 'on';
      app.text_nkeypoints_description.FontSize = 16;
      app.text_nkeypoints_description.FontColor = [0 1 1];
      app.text_nkeypoints_description.Position = [27 488 400 24];
      app.text_nkeypoints_description.Text = 'Number of keypoints to label for each animal';

      % Create text_nviews_description
      app.text_nviews_description = uilabel(app.figure1);
      app.text_nviews_description.Tag = 'text_nviews_description';
      app.text_nviews_description.BackgroundColor = [0 0.243 0.365];
      app.text_nviews_description.VerticalAlignment = 'top';
      app.text_nviews_description.WordWrap = 'on';
      app.text_nviews_description.FontSize = 16;
      app.text_nviews_description.FontColor = [0 1 1];
      app.text_nviews_description.Position = [26 363 400 85];
      app.text_nviews_description.Text = 'APT can do 3D labeling and tracking from multiple calibrated cameras. Enter 1 if animals were imaged from just one camera. Otherwise, enter the number of synced cameras recording the animals. ';

      % Create text_multianimal_description
      app.text_multianimal_description = uilabel(app.figure1);
      app.text_multianimal_description.Tag = 'text_multianimal_description';
      app.text_multianimal_description.BackgroundColor = [0 0.243 0.365];
      app.text_multianimal_description.VerticalAlignment = 'top';
      app.text_multianimal_description.WordWrap = 'on';
      app.text_multianimal_description.FontSize = 16;
      app.text_multianimal_description.FontColor = [0 1 1];
      app.text_multianimal_description.Position = [28 265 400 67];
      app.text_multianimal_description.Text = 'Check this box if there are multiple animals visible in any video frames. Otherwise, APT will assume there is just one animal visible per frame.';

      % Create text16
      app.text16 = uilabel(app.figure1);
      app.text16.Tag = 'text16';
      app.text16.BackgroundColor = [0 0.243 0.365];
      app.text16.VerticalAlignment = 'top';
      app.text16.WordWrap = 'on';
      app.text16.FontSize = 16;
      app.text16.FontColor = [0 1 1];
      app.text16.Position = [28 110 400 122];
      app.text16.Text = 'APT can do pose tracking on top of body tracking from an algorithm like FlyTracker or Ctrax. Check this box if you have already tracked the centroids and orientations of your animals and want to base pose tracking on those trajectories. If so, a trajectory file is input with each video.';

      % Show the figure after all components are created
      app.figure1.Visible = 'on';
    end  % function
  end  % methods (Access = private)
end  % classdef
