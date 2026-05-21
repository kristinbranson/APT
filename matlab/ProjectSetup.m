classdef ProjectSetup_App_exported < matlab.apps.AppBase

  % Properties that correspond to app components
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


  methods (Access = private)
    function handles = advModeCollapse(app, handles)
      h1 = findall(handles.figure1,'-property','Units');
      set(h1,'Units','pixels');
      posMid = handles.landmarkMid.Position;
      posMid = posMid(1)+posMid(3)/2;
      pos = handles.figure1.Position;
      pos(3) = posMid;
      handles.figure1.Position = pos;
      set(h1,'Units','normalized');
      handles.advancedOn = false;
      handles.pbAdvanced.String = 'Advanced >';
    end

    function handles = advModeExpand(app, handles)
      h1 = findall(handles.figure1,'-property','Units');
      set(h1,'Units','pixels');
      posRight = handles.landmarkRight.Position;
      posRight = posRight(1)+posRight(3);
      pos = handles.figure1.Position;
      pos(3) = posRight;
      handles.figure1.Position = pos;
      set(h1,'Units','normalized');
      handles.advancedOn = true;
      handles.pbAdvanced.String = '< Basic';
    end

    function handles = advModeToggle(app, handles)
      if handles.advancedOn
        handles = advModeCollapse(app, handles);
      else
        handles = advModeExpand(app, handles);
      end
    end

    function handles = advTableRefresh(app, handles, sMirror)
      tfRefresh = exist('sMirror','var')==0;
      if tfRefresh
        ad = getappdata(handles.figure1);
        sMirror = ad.mirror;
      end
      sMirror = Labeler.hlpAugmentOrTruncNameField(sMirror,'ViewNames','view',handles.nViews);
      sMirror = Labeler.hlpAugmentOrTruncNameField(sMirror,'LabelPointNames','point',handles.nPoints);
      sMirror = Labeler.hlpAugmentOrTruncStructField(sMirror,'View',handles.nViews);
      if ~isempty(handles.propsPane) && ishandle(handles.propsPane)
        delete(handles.propsPane);
        handles.propsPane = [];
      end

      handles.propsPane = [] ;
    end

    function cfg = genCurrentConfig(app, handles)
      % Generate config from the current UI state

      ad = getappdata(handles.figure1);
      cfg = ad.mirror;

      assert(numel(fieldnames(cfg.ViewNames))==handles.nViews);
      assert(numel(fieldnames(cfg.LabelPointNames))==handles.nPoints);
      cfg.NumViews = handles.nViews;
      cfg.NumLabelPoints = handles.nPoints;
      cfg.ViewNames = struct2cell(cfg.ViewNames);
      cfg.LabelPointNames = struct2cell(cfg.LabelPointNames);
      cfg.Trx.HasTrx = handles.cbHasTrx.Value;
      cfg.MultiAnimal = handles.cbMA.Value;
      isMA = cfg.MultiAnimal && ~cfg.Trx.HasTrx;
      if isMA
        cfg.LabelMode = LabelMode.MULTIANIMAL;
      else
        cfg.LabelMode = LabelMode.SEQUENTIAL;
      end
      % pumLM = handles.pumLabelingMode;
      % lmVal = pumLM.Value;
      % cfg.LabelMode = char(pumLM.UserData(lmVal));
      % pumTrk = handles.pumTracking;
      % tracker = pumTrk.String{pumTrk.Value};
      % cfg.Track.Enable = ~strcmpi(tracker,'none');
      cfg.Track.Enable = true;
      % cfg.Track.Type = tracker;
      % propertiesGUI treats props with empty vals as strings even if they are
      % subsequently filled with numbers
      FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
      for i=1:numel(cfg.View)
        cfg.View(i) = structLeavesStr2Double(cfg.View(i),FIELDS2DOUBLIFY);
      end
    end

    function handles = setCurrentConfig(app, handles, cfg)
      % Set given config on controls

      % we store these two props on handles in order to be able to revert;
      % data/model is split between i) primary UIcontrols and ii) adv panel
      handles.nViews = cfg.NumViews;
      handles.nPoints = cfg.NumLabelPoints;
      set(handles.etNumberOfViews,'string',num2str(handles.nViews));
      set(handles.etNumberOfPoints,'string',num2str(handles.nPoints));
      set(handles.cbHasTrx,'Value',cfg.Trx.HasTrx);
      set(handles.cbMA,'Value',cfg.MultiAnimal);


      % pumLM = handles.pumLabelingMode;
      % [tf,val] = ismember(cfg.LabelMode,arrayfun(@char,pumLM.UserData,'uni',0));
      % if ~tf
      %   % should never happen
      %   val = 1; % NONE
      % end
      % pumLM.Value = val;

      % pumTrk = handles.pumTracking;
      % if cfg.Track.Enable
      %   [tf,val] = ismember(cfg.Track.Type,pumTrk.String);
      %   if ~tf
      %     % unexpected but maybe not impossible due to path
      %     val = 1; % None
      %   end
      % else
      %   val = 1;
      % end
      % pumTrk.Value = val;

      sMirror = Labeler.cfg2mirror(cfg);
      handles = advTableRefresh(app, handles,sMirror);
    end

  end


  % Callbacks that handle component events
  methods (Access = private)

    % Code that executes after component creation
    function ProjectSetup_OpeningFcn(app, varargin)
      % --- Executes just before ProjectSetup is made visible.
      %
      % Modal dialog. Generates project configuration struct
      %
      % cfg = ProjectSetup();
      % cfg = ProjectSetup(hParentFig); % centered on hParentFig

      % Ensure that the app appears on screen when run
      movegui(app.figure1, 'onscreen');

      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app); %#ok<ASGLU>


      h1 = findall(handles.figure1,'-property','Units');
      set(h1,'Units','Normalized');
      set(handles.figure1,'MenuBar','None');

      if numel(varargin)>=1
        hParentFig = varargin{1};
        if ~ishandle(hParentFig)
          error('ProjectSetup:arg','Expected argument to be a figure handle.');
        end
        centerOnParentFigure(hObject,hParentFig);
      end

      handles.output = [];

      % init PUMs that depend only on codebase
      % lms = enumeration('LabelMode');
      % tfnone = lms==LabelMode.NONE;
      % lms(tfnone,:) = [];
      % lmStrs = arrayfun(@(x)x.prettyString,lms,'uni',0);
      % handles.pumLabelingMode.String = lmStrs;
      % handles.pumLabelingMode.UserData = lms;
      % trackers = LabelTracker.findAllSubclasses;
      % trackers = [{'None'};trackers];
      % handles.pumTracking.String = trackers;

      handles.propsPane = [];

      % init ui state
      cfg = Labeler.cfgGetLastProjectConfigNoView;
      handles = setCurrentConfig(app, handles,cfg);
      handles.propsPane.Position(4) = handles.propsPane.Position(3); % by default table is slightly bigger than panel for some reason
      handles = advModeCollapse(app, handles);

      guidata(hObject, handles);

      % UIWAIT makes ProjectSetup wait for user response (see UIRESUME)
      uiwait(handles.figure1);
    end

    % Value changed function: etNumberOfPoints
    function etNumberOfPoints_Callback(app, event)
      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      %fprintf('etNOP enter');
      val = str2double(hObject.String);
      if floor(val)==val && val>=1
        handles.nPoints = val;
      else
        hObject.String = handles.nPoints;
      end
      handles = advTableRefresh(app, handles);
      guidata(hObject,handles);
    end

    % Value changed function: etNumberOfViews
    function etNumberOfViews_Callback(app, event)
      %fprintf('etNOP end');

      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      val = str2double(hObject.String);
      if floor(val)==val && val>=1
        handles.nViews = val;
      else
        hObject.String = handles.nViews;
      end
      switch handles.nViews
        case 1
          handles.cbHasTrx.Enable = 'on';
          handles.cbMA.Enable = 'on';
        otherwise
          handles.cbHasTrx.Value = 0;
          handles.cbMA.Value = 0;
          handles.cbHasTrx.Enable = 'off';
          handles.cbMA.Enable = 'off';
      end
      handles = advTableRefresh(app, handles);
      guidata(hObject,handles);
    end

    % Value changed function: etProjectName
    function etProjectName_Callback(app, event)
      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      name = hObject.String;
      if ~all(isstrprop(name,'alphanum'))
        % This unfortunately invalidates _ also. Checking for it seems more work
        % than worth. MK 20220913
        warndlg('Name should have only alphanumberic characters');
        hObject.String = '';
      end
    end

    % Close request function: figure1
    function figure1_CloseRequestFcn(app, event)
      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      if isequal(get(hObject,'waitstatus'),'waiting')
        % The GUI is still in UIWAIT, us UIRESUME
        uiresume(hObject);
      else
        delete(hObject);
      end
    end

    % Button pushed function: pbAdvanced
    function pbAdvanced_Callback(app, event)
      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      handles = advModeToggle(app, handles);
      guidata(handles.figure1,handles);
    end

    % Button pushed function: pbCancel
    function pbCancel_Callback(app, event)
      %fprintf('pbCreate end');

      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      handles.output = [];
      guidata(handles.figure1,handles);
      close(handles.figure1);
    end

    % Button pushed function: pbCopySettingsFrom
    function pbCopySettingsFrom_Callback(app, event)
      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      lastLblFile = RC.getprop('lastLblFile');
      if isempty(lastLblFile)
        lastLblFile = pwd;
      end
      [fname,pth] = uigetfile('*.lbl','Select project file',lastLblFile);
      if isequal(fname,0)
        return;
      end
      lbl = loadLbl(fullfile(pth,fname));
      lbl = Labeler.lblModernize(lbl);
      cfg = lbl.cfg;
      handles = setCurrentConfig(app, handles,cfg);
      guidata(handles.figure1,handles);
    end

    % Button pushed function: pbCreateProject
    function pbCreateProject_Callback(app, event)
      % function pumLabelingMode_Callback(hObject, eventdata, handles)
      % function pumTracking_Callback(hObject, eventdata, handles)

      % Create GUIDE-style callback args - Added by Migration Tool
      [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>

      %fprintf('pbCreate start');
      cfg = genCurrentConfig(app, handles);
      cfg.ProjectName = handles.etProjectName.String;
      handles.output = cfg;
      guidata(handles.figure1,handles);
      close(handles.figure1);
    end
  end

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
    end
  end

  % App creation and deletion
  methods (Access = public)

    % Construct app
    function app = ProjectSetup_App_exported(varargin)

      % Create UIFigure and components
      createComponents(app)

      % Register the app with App Designer
      registerApp(app, app.figure1)

      % Execute the startup function
      runStartupFcn(app, @(app)ProjectSetup_OpeningFcn(app, varargin{:}))

      if nargout == 0
        clear app
      end
    end

    % Code that executes before app deletion
    function delete(app)

      % Delete UIFigure when app is deleted
      delete(app.figure1)
    end
  end
end