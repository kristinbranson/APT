classdef ParameterSetupModalController < handle
  % Controller for the modal dialog used to set training or tracking
  % parameters.  The dialog is modal, but the constructor returns as soon as
  % the dialog is up.  When the user clicks Apply, the dialog writes the new
  % parameters to the Labeler itself; on Cancel the edits are discarded.

  properties
    labelerController_  % the parent LabelerController that owns this dialog
    labeler_  % the Labeler object
    istrain_  % true for training parameters, false for tracking parameters
    tree_  % the parameter tree/specification being edited
    minNodeNumber_  % smallest node number in the tree
    maxNodeNumber_  % largest node number in the tree
    colors_  % colormap used to color nodes
    nodeNum2Color_  % function mapping a node number to a color
    important_level_  % the PropertyLevelsEnum value for "Important"
    levels_str_  % cellstr of level names
    levels_  % PropertyLevelsEnum array of the levels
    figure_  % the dialog figure
    gl_  % top-level grid layout
    gl_left_  % left-column grid layout
    tb_viz_curr_  % the currently-active viz toggle button, or []
    vizdata_  % struct of data used by parameter visualizations
    keypointParamState_  % current keypoint-pairs state
    tabgroup_params_  % tab group holding the parameter tabs (training only)
    tab_autotune_  % the auto-tune tab (training only)
    hauto_  % handles for the auto-tune tab contents (training only)
    tab_important_  % the "Important" tab (training only)
    htree_important_  % tree handles for the "Important" tab (training only)
    tabs_rest_  % array of the remaining category tabs (training only)
    htree_rest_  % cell array of tree handles for the remaining tabs (training only)
    htree_  % tree handles for the single tracking tab (tracking only)
    gl_buttons_  % grid layout holding the level dropdown and buttons
    popupmenu_level_  % the level-selection dropdown
    pb_apply_  % the Apply button
    pb_cancel_  % the Cancel button
    panel_right_  % right-column panel holding the visualization
    tile_viz_  % tiled layout holding the visualization
    vizid_  % id of the currently-shown visualization, or ''
    vizobj_  % the currently-shown visualization object, or []
    autoparams_  % containers.Map of automatically-computed parameters
  end

  methods
    function obj = ParameterSetupModalController(parent, labeler, varargin)
      % Construct and show the modal parameter-setup dialog.
      obj.labelerController_ = parent ;
      obj.labeler_ = labeler;
      [obj.istrain_] = ...
        myparse(varargin,'istrain',true);

      if obj.istrain_,
        sPrmCurrent = obj.labeler_.trackGetTrainingParams();
        % Start with default "new" parameter tree/specification
        obj.tree_ = APTParameters.defaultParamsTree() ;
        % Overlay our starting point
        obj.tree_.structapply(sPrmCurrent);
      else
        obj.tree_ = obj.labeler_.trackGetTrackParams();
      end

      obj.resetTreeVisible();
      APTParameters.addNumbers(obj.tree_);
      [obj.minNodeNumber_,obj.maxNodeNumber_] = APTParameters.numberRange(obj.tree_);
      obj.colors_ = jet(256)*.25+.75;
      colors = obj.colors_;
      minNodeNumber = obj.minNodeNumber_;
      maxNodeNumber = obj.maxNodeNumber_;
      obj.nodeNum2Color_ = @(x) colors(round(1 + (x-minNodeNumber)/(maxNodeNumber-minNodeNumber)*255),:);

      obj.important_level_ = PropertyLevelsEnum('Important');
      obj.levels_str_ = {'Beginner','Advanced','Developer'};
      obj.levels_ = PropertyLevelsEnum(obj.levels_str_);

      lastLevel = ParameterSetupModalController.lastLevel_();
      if isempty(lastLevel),
        lastLevel = PropertyLevelsEnum(min(obj.levels_));
        ParameterSetupModalController.lastLevel_(lastLevel);
      end

      if obj.istrain_,
        tistr = 'Training parameters';
      else
        tistr = 'Tracking parameters';
      end

      obj.figure_ = uifigure('Name',tistr,...
        'Units','pixels', ...
        'Position',[100,100,1000,600], ...
        'Resize', 'on', ...
        'Visible','on', ...%'CloseRequestFcn', @cancel_callback,...
        'WindowStyle','modal', ...
        'Tag','figure_ParameterSetup') ;

      obj.gl_ = uigridlayout(obj.figure_,[1,2],'ColumnWidth',{'1x','1x'});

      obj.gl_left_ = uigridlayout(obj.gl_,[2,1],'RowHeight',{'1x','fit'},'Padding',[0,0,0,0]);

      obj.tb_viz_curr_ = [];
      obj.vizdata_ = struct;
      obj.keypointParamState_ = obj.labeler_.getKeypointParams();

      if obj.istrain_,

        obj.tabgroup_params_ = uitabgroup('Parent',obj.gl_left_,'Tag','tabgroup_params');
        obj.tab_autotune_ = uitab('Parent',obj.tabgroup_params_,...
          'Title','Auto-tune','Scrollable','on','ForegroundColor',[1,0,1]);
        obj.hauto_ = obj.InitAutoTune();

        obj.tab_important_ = uitab('Parent',obj.tabgroup_params_,...
          'Title','Important','Scrollable','on','ForegroundColor',[1,0,0]);
        APTParameters.filterPropertiesByLevel(obj.tree_,obj.important_level_);
        obj.htree_important_ = obj.InitTree(obj.tree_,obj.tab_important_,'important',0);

        obj.resetTreeVisible();
        APTParameters.filterPropertiesByLevel(obj.tree_,lastLevel);
        fprintf('lastLevel = %s\n',lastLevel);
        obj.tabs_rest_ = gobjects(1,0);
        obj.htree_rest_ = {};
        obj.resetTabsRest();

      else

        APTParameters.filterPropertiesByLevel(obj.tree_,lastLevel);
        obj.htree_ = obj.InitTree(obj.tree_,obj.gl_left_,'track',0);

      end  % if

      obj.gl_buttons_ = uigridlayout(obj.gl_left_,[1,3],...
        'ColumnWidth',{'1x',80,80});
      obj.popupmenu_level_ = uidropdown('Parent',obj.gl_buttons_,...
        'Items',obj.levels_str_,'Value',lastLevel,...
        'ValueChangedFcn',@obj.cbkLevelChanged,'Tag','popupmenu_level');
      obj.pb_apply_ = uibutton(obj.gl_buttons_,'Text','Apply',...
        'ButtonPushedFcn',@obj.cbkApply,'Tag','pb_apply');
      obj.pb_cancel_ = uibutton(obj.gl_buttons_,'Text','Cancel',...
        'ButtonPushedFcn',@obj.cbkCancel,'Tag','pb_cancel');

      obj.panel_right_ = uipanel(obj.gl_,'Tag','panel_right');
      obj.tile_viz_ = tiledlayout(obj.panel_right_,'vertical','TileSpacing','tight','Padding','compact');

      obj.vizid_ = '';
      obj.vizobj_ = [];
      obj.clearParamViz();
      waitForFigureToSync(obj.figure_) ;  % block until the figure is actually visible
    end  % constructor

    function delete(obj)
      % Destructor: delete the dialog figure if it still exists.
      deleteValidGraphicsHandles(obj.figure_) ;
    end  % function

    function resetTreeVisible(obj)
      % Reset which tree nodes are visible for the current labeler state.
      APTParameters.setAllVisible(obj.tree_);
      APTParameters.filterPropertiesByCondition(obj.tree_,obj.labeler_);
      if obj.istrain_,
        % only show training parameters
        APTParameters.filterPropertiesByAffectsTraining(obj.tree_,true);
      end

      % currently everything seems to affect training
      %APTParameters.filterPropertiesByAffectsTraining(obj.tree_,obj.istrain_);
    end  % function

    function idxvisible = getChildrenIdxVisible(obj, tprm)  %#ok<INUSD>
      % Return the indices of the visible children of a tree node.
      isvisible = false(1,numel(tprm.Children));
      for i = 1:numel(tprm.Children),
        isvisible(i) = tprm.Children(i).Data.Visible;
      end
      idxvisible = find(isvisible);
    end  % function

    function deleteInvalidObjects(obj, node)  %#ok<INUSD>
      % Remove deleted UI handles from a tree node's UserData.
      node.Data.UserData(~ishandle(node.Data.UserData)) = [];
    end  % function

    function resetLevelTrack(obj)
      % Rebuild the tracking-parameter tree after a level change.
      delete(obj.htree_.Data.handles.gl1);
      obj.htree_ = obj.InitTree(obj.tree_,obj.gl_left_,'track',0);
      obj.htree_.Data.handles.gl1.Layout.Row = 1;
    end  % function

    function resetTabsRest(obj)
      % Rebuild the per-category parameter tabs after a level change.

      fprintf('nleaves = %d\n',numel(APTParameters.getVisibleLeaves(obj.tree_)));

      currtab = obj.tabgroup_params_.SelectedTab;
      currtab_tag = currtab.Tag;
      needset = ismember(currtab,obj.tabs_rest_);
      for tab = obj.tabs_rest_(:)',
        delete(tab);
      end
      obj.tree_.traverse(@obj.deleteInvalidObjects);
      if ~isempty(obj.tb_viz_curr_) && ~ishandle(obj.tb_viz_curr_),
        obj.clearParamViz()
      end

      idxvisible = obj.getChildrenIdxVisible(obj.tree_);
      nchil = numel(idxvisible);
      obj.tabs_rest_ = gobjects(1,nchil);
      obj.htree_rest_ = cell(1,nchil);
      for i = 1:numel(idxvisible),
        chil = obj.tree_.Children(idxvisible(i));
        tag = ['tab_',chil.Data.Field];
        obj.tabs_rest_(i) = uitab('Parent',obj.tabgroup_params_,...
          'Title',obj.cleanDisplayName(chil.Data.DispNameUse),'Scrollable','on',...
          'Tag',tag);
        obj.htree_rest_{i} = obj.InitTree(chil,obj.tabs_rest_(i),chil.Data.Field,0);
        if needset && strcmp(tag,currtab_tag),
          obj.tabgroup_params_.SelectedTab = obj.tabs_rest_(i);
          needset = false;
        end
      end
      if needset,
        obj.tabgroup_params_.SelectedTab = obj.tab_important_;
      end
    end  % function

    function htree = InitNode(obj, tprm, parent, tag, depth, htree)
      % Build the UI for an internal (non-leaf) parameter tree node.

      if depth == 0,
        color = [.94,.94,.94];
      else
        color = obj.nodeNum2Color_(tprm.Data.Index);
      end

      if depth == 0,
        padding = 10;
        scrollable = 'on';
      else
        padding = 0;
        scrollable = 'off';
      end

      n = numel(obj.getChildrenIdxVisible(tprm));
      if strcmp(tprm.Data.Field,'ROOT'),
        n1 = n;
      else
        n1 = n+1;
      end

      htree.Data.handles.gl1 = uigridlayout(parent,[n1,1],...
        'Tag',['gl_',tag],'Padding',padding+zeros(1,4),...
        'RowHeight',repmat({'fit'},1,n+2),'Scrollable',scrollable);

      if ~strcmp(tprm.Data.Field,'ROOT'),
        s = obj.getDisplayName(tprm.Data);
        s = [repmat('>',[1,depth]),' ',s];
        ti = sprintf('<b>%s</b>: %s',s,tprm.Data.Description);
        htree.Data.handles.title = uilabel('Parent',htree.Data.handles.gl1,...
          'Text',ti,'Interpreter','html','WordWrap','on',...
          'BackgroundColor',color);
      end

      nchildren = numel(tprm.Children);
      for i = 1:nchildren,
        htreecurr = obj.InitTree(tprm.Children(i),htree.Data.handles.gl1,tag,depth+1);
        if i == 1,
          htree.Children = htreecurr;
        else
          htree.Children(end+1) = htreecurr;
        end
      end

    end  % function

    function s = getDisplayName(obj, data)
      % Return the display name for a parameter, annotated with its stage.
      s = data.DispNameUse;
      s = obj.cleanDisplayName(s);
      stage = APTParameters.getStage(data.FullPath);
      if obj.labeler_.trackerIsTwoStage
        if strcmpi(stage,'first'),
          s = [s,' (detection stage)'];
        elseif strcmp(stage,'last'),
          s = [s,' (pose stage)'];
        end
      end
    end  % function

    function s = cleanDisplayName(obj, s)  %#ok<INUSD>
      % Insert spaces into a camelCase display name.
      s = regexprep(s,'([a-z])([A-Z])','$1 $2');
    end  % function

    function leafhandles = InitLeaf(obj, tprm, parent, varargin)
      % Build the UI for a leaf (single parameter) tree node.

      [tag,suggestedvalue,extradescr] = myparse(varargin,...
        'tag',tprm.Data.Field,'suggestedvalue',[],'extradescr','');

      leafhandles = struct;
      s = obj.getDisplayName(tprm.Data);
      padding = 0;
      color = obj.nodeNum2Color_(tprm.Data.Index);

      leafhandles.gl1 = uigridlayout(parent,[2,1],'RowHeight',{'fit','fit'},'Padding',padding+zeros(1,4),...
          'BackgroundColor',color);
      leafhandles.gl2 = uigridlayout(leafhandles.gl1,[1,3],'ColumnWidth',{'3x','2x','fit'},'Padding',[0,0,0,0],...
          'BackgroundColor',color);
      leafhandles.label = uilabel('Parent',leafhandles.gl2,'Text',s,'FontWeight','bold');
      if iscell(tprm.Data.Type),
        leafhandles.value = uidropdown('Parent',leafhandles.gl2,...
          'Items',tprm.Data.Type,'Value',tprm.Data.Value,...
          'ValueChangedFcn',@obj.cbkValueDropdown,...
          'Tag',tag);
      else
        switch tprm.Data.Type,
          case 'unsigned',
            leafhandles.value = uispinner(leafhandles.gl2,...
              'Limits',[0,inf],'Tag',tag,...
              'ValueChangedFcn',@obj.cbkValueSpinner,...
              'Value',tprm.Data.Value,'RoundFractionalValues','on',...
              'ValueDisplayFormat','%d');
          case 'signed',
            leafhandles.value = uispinner(leafhandles.gl2,...
              'Limits',[-inf,inf],'Tag',tag,...
              'ValueChangedFcn',@obj.cbkValueSpinner,...
              'Value',tprm.Data.Value,'RoundFractionalValues','on',...
              'ValueDisplayFormat','%d');
          case 'float',
            leafhandles.value = uieditfield(leafhandles.gl2,...
              'numeric','Limits',[-inf,inf],'Tag',tag,...
              'ValueChangedFcn',@obj.cbkValueEditField,...
              'Value',tprm.Data.Value);
          case 'prctile',
            leafhandles.value = uieditfield(leafhandles.gl2,...
              'numeric','Limits',[0,100],'Tag',tag,...
              'ValueChangedFcn',@obj.cbkValueEditField,...
              'Value',tprm.Data.Value);
          case 'string',
            leafhandles.value = uieditfield(leafhandles.gl2,...
              'text','Tag',tag,...
              'ValueChangedFcn',@obj.cbkValueEditField,...
              'Value',tprm.Data.Value);
          case 'boolean',
            leafhandles.value = uidropdown('Parent',leafhandles.gl2,...
              'Items',{'False','True'},'ItemsData',[false,true],'ValueIndex',double(tprm.Data.Value)+1,...
              'ValueChangedFcn',@obj.cbkValueBool,...
              'Tag',tag);
          otherwise
            error('Unknown parameter type %s',tprm.Data.Type);
        end
      end
      leafhandles.value.Enable = onIff(tprm.Data.isEditable);
      if ~isempty(tprm.Data.ParamViz),
        leafhandles.tb_viz = uibutton("state",'Parent',leafhandles.gl2,...
          'Text','Viz >>','ValueChangedFcn',@obj.cbkVizButton,...
          'UserData',tprm.Data,'tag',['tb_viz_',tag],...
          'Value',0);
      else
        leafhandles.tb_viz = gobjects(1,0);
      end

      descr = tprm.Data.Description;
      if ~isempty(suggestedvalue),
        ss = ['Suggested value: ',mat2str(suggestedvalue)];
        if ~isequal(suggestedvalue,tprm.Data.Value),
          ss = sprintf('<font color="red"><b>%s</b></font>',ss);
        end
        descr = [descr,'<br/>',ss];
      end
      if ~isempty(extradescr),
        descr = [descr,'<br/>',extradescr];
      end
      leafhandles.descr = uilabel('Parent',leafhandles.gl1,'Text',descr,...
        'WordWrap','on','Interpreter','html');
      leafhandles.value.UserData = struct('data',tprm.Data,'suggestedvalue',suggestedvalue,...
        'extradescr',extradescr,'hdescr',leafhandles.descr,'htb_viz',leafhandles.tb_viz);
      if isempty(tprm.Data.UserData),
        tprm.Data.UserData = leafhandles.value;
      else
        tprm.Data.UserData(end+1) = leafhandles.value;
      end
    end  % function

    function htree = InitTree(obj, tprm, parent, tag, depth)
      % Recursively build the UI for a parameter (sub)tree.

      htree = struct;
      htree.Data = struct;
      htree.Data.handles = struct;
      htree.Data.Field = tprm.Data.Field;
      htree.Children = [];
      if ~tprm.Data.Visible,
        return;
      end
      idxvisible = obj.getChildrenIdxVisible(tprm);
      tag = [tag,'_',tprm.Data.Field];

      if ~isempty(idxvisible),
        htree = obj.InitNode(tprm,parent,tag,depth,htree);
      else
        htree.Data.handles = obj.InitLeaf(tprm,parent,'tag',tag);
      end

    end  % function

    function hkp = InitKeypointParamsButton(obj, parent, color)
      % Build the "Keypoint pairs" button used for flip augmentation.

      if ~exist('color', 'var'),
        color = [];
      end

      descr = ['If you augment your training data by flipping horizontally or vertically, you ',...
        'MUST set pairs of corresponding keypoints. '];
      hkp = obj.InitButton(parent,...
        'tag','flippairs','titlestr','Keypoint pairs',...
        'descr',descr,'buttonlabel','Set','color',color,'Callback',@obj.cbkKeypointParams);

    end  % function

    function hauto = InitAutoTune(obj)
      % Build the contents of the auto-tune tab.

      hauto = struct;

      % automatically set the parameters based on labels.
      [autoparams,vizdataAutoparams] = apt.compute_auto_params(obj.labeler_);
      obj.autoparams_ = autoparams;
      obj.vizdata_.autoparams = vizdataAutoparams;
      kk = obj.autoparams_.keys();

      [horz_flip_prm,vert_flip_prm] = APTParameters.getDataAugmentationFlipParams(obj.tree_,false);

      align_trx_theta_prm = APTParameters.getAlignTrxTheta(obj.tree_,false);

      nfields = 1;
      if align_trx_theta_prm.Data.Visible,
        nfields = nfields + 1;
      end
      if horz_flip_prm.Data.Visible,
        nfields = nfields + 1;
      end
      if vert_flip_prm.Data.Visible,
        nfields = nfields + 1;
      end
      if horz_flip_prm.Data.Visible || vert_flip_prm.Data.Visible,
        nfields = nfields + 1;
      end
      for i = 1:numel(kk),
        nprm = obj.tree_.findnode(kk{i});
        if nprm.Data.Visible,
          nfields = nfields + 1;
        end
      end

      hauto.gl = uigridlayout(obj.tab_autotune_,[nfields+1,1],'RowHeight',repmat({'fit'},[1,nfields+1]),'Scrollable','on');
      parent = hauto.gl;
      hauto.auto = {};


      % todo: put these in compute_auto_parameters
      if align_trx_theta_prm.Data.Visible,
        % Using head-tail for the first stage
        align_trx_theta = align_trx_theta_prm.Data.Value;
        hauto.auto{end+1} = obj.InitLeaf(align_trx_theta_prm,parent,'suggestedvalue',true,...
          'extradescr','Aligning animals using head-tail direction will lead to better performance.');
      else
        align_trx_theta = false;
      end
      if align_trx_theta,
        suggestedvalue = struct('horz',true,'vert',false);
        extradescr = 'Head-tail alignment is true. Horizontal flipping and not vertical flipping is recommended as the animal is rotated to face up.';
      else
        suggestedvalue = struct('horz',{[]},'vert',{[]});
        extradescr = '';
      end
      if horz_flip_prm.Data.Visible,
        hauto.auto{end+1} = obj.InitLeaf(horz_flip_prm,parent,'suggestedvalue',suggestedvalue.horz,...
          'extradescr',extradescr);
        flipcolor = obj.nodeNum2Color_(horz_flip_prm.Data.Index);
      end
      if vert_flip_prm.Data.Visible,
        hauto.auto{end+1} = obj.InitLeaf(vert_flip_prm,parent,'suggestedvalue',suggestedvalue.vert,...
          'extradescr',extradescr);
        flipcolor = obj.nodeNum2Color_(vert_flip_prm.Data.Index);
      end

      if horz_flip_prm.Data.Visible || vert_flip_prm.Data.Visible,
        hauto.auto{end+1} = obj.InitKeypointParamsButton(parent,flipcolor);
      end

      for i = 1:numel(kk),
        k = kk{i};
        nprm = obj.tree_.findnode(k);
        if nprm.Data.Visible,
          hauto.auto{end+1} = obj.InitLeaf(nprm,parent,'suggestedvalue',obj.autoparams_(k));
        end
      end

      gl = uigridlayout(hauto.gl,[1,3],'Padding',[0,0,0,0]);
      hauto.pb_accept_suggestions = uibutton(gl,'Text','Accept suggestions',...
          'ButtonPushedFcn',@obj.cbkAcceptSuggestions,'Tag','pb_accept_suggestions');
      hauto.pb_accept_suggestions.Layout.Column = 2;

    end  % function

    function cbkStoreKeypointParams(obj, state)
      % Store the keypoint-pairs state edited by the keypoint viz.
      obj.keypointParamState_ = state;
    end  % function

    function cbkKeypointParams(obj, src, evt)  %#ok<INUSD>
      % Callback for the "Keypoint pairs" button: show the keypoint viz.
      obj.vizobj_ = ParameterVisualizationKeypointParams();
      obj.vizobj_.init(obj.tile_viz_,obj.labeler_,'',obj.tree_,...
        @obj.cbkStoreKeypointParams,obj.keypointParamState_,obj.labelerController_);
    end  % function

    function buttonhandles = InitButton(obj, parent, varargin)  %#ok<INUSD>
      % Build a labeled button with a title and description.

      [tag,titlestr,descr,buttonlabel,color,cbk] = myparse(varargin,...
        'tag','','titlestr','','descr','','buttonlabel','Button','color',[],'Callback','');

      buttonhandles = struct;
      padding = 0;

      buttonhandles.gl1 = uigridlayout(parent,[2,1],'RowHeight',{'fit','fit'},'Padding',padding+zeros(1,4));
      if ~isempty(color),
        buttonhandles.gl1.BackgroundColor = color;
      end
      ws = {'2x','1x'};
      ncurr = 2;
      buttonhandles.gl2 = uigridlayout(buttonhandles.gl1,[1,ncurr],'ColumnWidth',ws,'Padding',[0,0,0,0]);
      if ~isempty(color),
        buttonhandles.gl2.BackgroundColor = color;
      end
      buttonhandles.label = uilabel('Parent',buttonhandles.gl2,'Text',titlestr,'FontWeight','bold');
      buttonhandles.button = uibutton('Parent',buttonhandles.gl2,'Text',buttonlabel,'Tag',['pb_',tag],...
        'ButtonPushedFcn',cbk);
      buttonhandles.descr = uilabel('Parent',buttonhandles.gl1,'Text',descr,...
        'WordWrap','on','Interpreter','html');
    end  % function

    function cbkValueDropdown(obj, src, evt)  %#ok<INUSD>
      % Callback for a dropdown-valued parameter.
      ud = src.UserData;
      value = src.Value;
      obj.updateValue(ud,value,src);
    end  % function

    function cbkValueEditField(obj, src, evt)  %#ok<INUSD>
      % Callback for an edit-field-valued parameter.
      ud = src.UserData;
      value = src.Value;
      obj.updateValue(ud,value,src);
    end  % function

    function cbkValueSpinner(obj, src, evt)  %#ok<INUSD>
      % Callback for a spinner-valued parameter.
      ud = src.UserData;
      value = src.Value;
      obj.updateValue(ud,value,src);
    end  % function

    function cbkValueBool(obj, src, evt)  %#ok<INUSD>
      % Callback for a boolean-valued parameter.
      ud = src.UserData;
      value = src.ValueIndex==2;
      obj.updateValue(ud,value,src);
    end  % function

    function updateValue(obj, ud, value, src)
      % Apply an edited value to the tree and mirror it in the UI.
      fprintf('updating value of %s to %s\n',ud.data.FullPath,mat2str(value));
      if ~exist('src', 'var'),
        src = gobjects(1,0);
      end
      ud.data.Value = value;
      ud.data.UserData(~ishandle(ud.data.UserData)) = [];

      for hother = [src,ud.data.UserData],
        if ~isequal(hother,src),
          obj.setUIValue(hother,ud.data.Type,value);
        end
        descr = ud.data.Description;
        if ~isempty(ud.suggestedvalue),

          ss = ['Suggested value: ',mat2str(ud.suggestedvalue)];
          if ~isequal(ud.suggestedvalue,ud.data.Value),
            ss = sprintf('<font color="red"><b>%s</b></font>',ss);
          end
          descr = [descr,'<br/>',ss];  %#ok<AGROW>
        end
        if ~isempty(ud.extradescr),
          descr = [descr,'<br/>',ud.extradescr];  %#ok<AGROW>
        end
        ud.hdescr.Text = descr;
      end

      if ~isempty(ud.data.ParamViz) && ud.htb_viz.Value,
        fprintf('calling updateParamViz\n');
        assert(isequal(ud.data.ParamViz,obj.vizid_));
        obj.updateParamViz(ud.data);
      end

    end  % function

    function setUIValue(obj, h, type, value)  %#ok<INUSD>
      % Set a UI widget's value according to the parameter type.
      if iscell(type),
        h.Value = value;
      else
        switch type,
          case {'unsigned','signed','float','prctile','string'},
            h.Value = value;
          case 'boolean',
            h.ValueIndex = double(value)+1;
          otherwise
            error('Unknown parameter type %s',type);
        end
      end
    end  % function

    function cbkAcceptSuggestions(obj, src, evt)  %#ok<INUSD>
      % Callback for the "Accept suggestions" button in the auto-tune tab.

      for i = 1:numel(obj.hauto_.auto)
        h = obj.hauto_.auto{i};
        if ~isfield(h,'value'),
          continue;
        end
        ud = h.value.UserData;
        if ~isempty(ud.suggestedvalue),
          obj.setUIValue(h.value,ud.data.Type,ud.suggestedvalue)
          obj.updateValue(ud,ud.suggestedvalue);
        end
      end

    end  % function

    function cbkLevelChanged(obj, src, evt)  %#ok<INUSD>
      % Callback for the level dropdown: refilter and rebuild the tree(s).
      lastLevel = PropertyLevelsEnum(src.Value);
      ParameterSetupModalController.lastLevel_(lastLevel);
      obj.resetTreeVisible();
      APTParameters.filterPropertiesByLevel(obj.tree_,lastLevel);
      if obj.istrain_,
        obj.resetTabsRest();
      else
        obj.resetLevelTrack();
      end
    end  % function

    function cbkVizButton(obj, src, evt)  %#ok<INUSD>
      % Callback for a parameter's "Viz >>" toggle button.

      val = src.Value;

      data = src.UserData;
      fprintf('cbkVizButton: %s\n',data.ParamViz);

      if val == 1,
        obj.initParamViz(data);
        obj.tb_viz_curr_ = src;
      else
        obj.clearParamViz();

      end

    end  % function

    function clearParamViz(obj)
      % Clear the current parameter visualization.
      if ~isempty(obj.vizobj_),
        obj.vizobj_.clear();
      end
      if ishandle(obj.tile_viz_),
        delete(obj.tile_viz_.Children);
      else
        obj.tile_viz_ = tiledlayout(obj.panel_right_,'vertical','TileSpacing','compact','Padding','compact');
      end
      obj.vizid_ = '';
      obj.vizobj_ = [];
      if ~isempty(obj.tb_viz_curr_) && ishandle(obj.tb_viz_curr_),
        obj.tb_viz_curr_.Value = 0;
        obj.tb_viz_curr_ = [];
      end
    end  % function

    function initParamViz(obj, data)
      % Initialize the parameter visualization for a given parameter.
      vizid = data.ParamViz;
      obj.clearParamViz();
      obj.vizid_ = vizid;
      % we are going to ignore paramVizID -- I don't understand its function
      [vizclassname,paramVizID] = ParameterVisualization.parseParamVizSpec(vizid); %#ok<ASGLU>
      obj.vizobj_ = feval(vizclassname);
      obj.vizobj_.init(obj.tile_viz_,obj.labeler_,data.FullPath,obj.tree_,obj.vizdata_);
    end  % function

    function updateParamViz(obj, data)
      % Update the current parameter visualization for a changed parameter.
      vizid = data.ParamViz;
      fprintf('Updating %s in %s to %s, obj.vizid_ = %s\n',data.FullPath,vizid,mat2str(data.Value),obj.vizid_);
      if ~isequal(obj.vizid_,vizid),
        return;
      end
      obj.vizobj_.update();
    end  % function

    function cbkApply(obj, src, evt)  %#ok<INUSD>
      % Dismiss the dialog, then write the edited parameters to the labeler.
      obj.clearParamViz();
      sPrmNew = obj.tree_.structize();
      keypointParams = obj.keypointParamState_;
      close(obj.figure_);
      labeler = obj.labeler_;
      if obj.istrain_,
        labeler.trackSetTrainingParams(sPrmNew);
      else
        labeler.setTrackingParameters(sPrmNew);
      end
      labeler.setKeypointParams(keypointParams);
      labeler.setDoesNeedSave(true, 'Parameters changed');
    end  % function

    function cbkCancel(obj, src, evt)  %#ok<INUSD>
      % Dismiss the dialog, discarding any edits.
      close(obj.figure_);
    end  % function

  end  % methods

  methods (Static)
    function value = lastLevel_(newValue)
      % Get or set the level remembered across openings of the dialog.
      persistent lastLevel
      if exist('newValue', 'var'),
        lastLevel = newValue;
      end
      value = lastLevel;
    end  % function
  end  % methods

end  % classdef
