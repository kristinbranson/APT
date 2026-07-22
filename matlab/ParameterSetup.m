function output = ParameterSetup(varargin)

persistent lastLevel;

handles = struct;
output = [];
handles.output = output;
if (nargin>=1) && isa(varargin{1},'Labeler'),
  handles.labelerObj = varargin{1};
  [handles.hPar,handles.istrain] = myparse(varargin(2:end),'hPar',nan,'istrain',true);
else
  feval(varargin{:});
  return;
end

if handles.istrain,
  sPrmCurrent = handles.labelerObj.trackGetTrainingParams();
  % Start with default "new" parameter tree/specification
  handles.tree = APTParameters.defaultParamsTree() ;
  % Overlay our starting point
  handles.tree.structapply(sPrmCurrent);
else
  handles.tree = handles.labelerObj.trackGetTrackParams();
end

resetTreeVisible();
APTParameters.addNumbers(handles.tree);
[handles.minNodeNumber,handles.maxNodeNumber] = APTParameters.numberRange(handles.tree);
handles.colors = jet(256)*.25+.75;
handles.nodeNum2Color = @(x) handles.colors(round(1 + (x-handles.minNodeNumber)/(handles.maxNodeNumber-handles.minNodeNumber)*255),:);

handles.important_level = PropertyLevelsEnum('Important');
handles.levels_str = {'Beginner','Advanced','Developer'};
handles.levels = PropertyLevelsEnum(handles.levels_str);

if isempty(lastLevel),
  lastLevel = PropertyLevelsEnum(min(handles.levels));
end

if handles.istrain,
  tistr = 'Training parameters';
else
  tistr = 'Tracking parameters';
end

handles.figure = uifigure('Name',tistr,...
  'Units','pixels', ...
  'Position',[100,100,1000,600], ...
  'Resize', 'on', ...
  'Visible','on', ...%'CloseRequestFcn', @cancel_callback,...
  'Tag','figure_ParameterSetup') ;

handles.gl = uigridlayout(handles.figure,[1,2],'ColumnWidth',{'1x','1x'});

handles.gl_left = uigridlayout(handles.gl,[2,1],'RowHeight',{'1x','fit'},'Padding',[0,0,0,0]);

handles.tb_viz_curr = [];
handles.vizdata = struct;
handles.keypointParamState = handles.labelerObj.getKeypointParams();

if handles.istrain,

  handles.tabgroup_params = uitabgroup('Parent',handles.gl_left,'Tag','tabgroup_params');
  handles.tab_autotune = uitab('Parent',handles.tabgroup_params,...
    'Title','Auto-tune','Scrollable','on','ForegroundColor',[1,0,1]);
  handles.hauto = InitAutoTune();

  handles.tab_important = uitab('Parent',handles.tabgroup_params,...
    'Title','Important','Scrollable','on','ForegroundColor',[1,0,0]);
  APTParameters.filterPropertiesByLevel(handles.tree,handles.important_level);
  handles.htree_important = InitTree(handles.tree,handles.tab_important,'important',0);

  resetTreeVisible();
  APTParameters.filterPropertiesByLevel(handles.tree,lastLevel);
  fprintf('lastLevel = %s\n',lastLevel);
  handles.tabs_rest = gobjects(1,0);
  handles.htree_rest = {};
  resetTabsRest();

else

  APTParameters.filterPropertiesByLevel(handles.tree,lastLevel);
  handles.htree = InitTree(handles.tree,handles.gl_left,'track',0);

end

handles.gl_buttons = uigridlayout(handles.gl_left,[1,3],...
  'ColumnWidth',{'1x',80,80});
handles.popupmenu_level = uidropdown('Parent',handles.gl_buttons,...
  'Items',handles.levels_str,'Value',lastLevel,...
  'ValueChangedFcn',@cbkLevelChanged);
handles.pb_apply = uibutton(handles.gl_buttons,'Text','Apply',...
  'ButtonPushedFcn',@cbkApply,'Tag','pb_apply');
handles.pb_cancel = uibutton(handles.gl_buttons,'Text','Cancel',...
  'ButtonPushedFcn',@cbkCancel,'Tag','pb_cancel');

handles.panel_right = uipanel(handles.gl,'Tag','panel_right');
handles.tile_viz = tiledlayout(handles.panel_right,'vertical','TileSpacing','tight','Padding','compact');

handles.vizid = '';
handles.vizobj = [];
clearParamViz();

uiwait(handles.figure);
output = handles.output;

  function resetTreeVisible()
    APTParameters.setAllVisible(handles.tree);
    APTParameters.filterPropertiesByCondition(handles.tree,handles.labelerObj);
    if handles.istrain,
      % only show training parameters
      APTParameters.filterPropertiesByAffectsTraining(handles.tree,true);
    end

    % currently everything seems to affect training
    %APTParameters.filterPropertiesByAffectsTraining(handles.tree,handles.istrain);
  end

  function idxvisible = getChildrenIdxVisible(tprm)
    isvisible = false(1,numel(tprm.Children));
    for i = 1:numel(tprm.Children),
      isvisible(i) = tprm.Children(i).Data.Visible;
    end
    idxvisible = find(isvisible);
  end

  function deleteInvalidObjects(node)
    node.Data.UserData(~ishandle(node.Data.UserData)) = [];
  end

  function resetLevelTrack()
    delete(handles.htree.Data.handles.gl1);
    handles.htree = InitTree(handles.tree,handles.gl_left,'track',0);
    handles.htree.Data.handles.gl1.Layout.Row = 1;
  end

  function resetTabsRest()

    fprintf('nleaves = %d\n',numel(APTParameters.getVisibleLeaves(handles.tree)));

    currtab = handles.tabgroup_params.SelectedTab;
    currtab_tag = currtab.Tag;
    needset = ismember(currtab,handles.tabs_rest);
    for tab = handles.tabs_rest(:)',
      delete(tab);
    end
    handles.tree.traverse(@deleteInvalidObjects);
    if ~isempty(handles.tb_viz_curr) && ~ishandle(handles.tb_viz_curr),
      clearParamViz()
    end

    idxvisible = getChildrenIdxVisible(handles.tree);
    nchil = numel(idxvisible);
    handles.tabs_rest = gobjects(1,nchil);
    handles.htree_rest = cell(1,nchil);
    for i = 1:numel(idxvisible),
      chil = handles.tree.Children(idxvisible(i));
      tag = ['tab_',chil.Data.Field];
      handles.tabs_rest(i) = uitab('Parent',handles.tabgroup_params,...
        'Title',cleanDisplayName(chil.Data.DispNameUse),'Scrollable','on',...
        'Tag',tag);
      handles.htree_rest{i} = InitTree(chil,handles.tabs_rest(i),chil.Data.Field,0);
      if needset && strcmp(tag,currtab_tag),
        handles.tabgroup_params.SelectedTab = handles.tabs_rest(i);
        needset = false;
      end
    end
    if needset,
      handles.tabgroup_params.SelectedTab = handles.tab_important;
    end
  end

  function htree = InitNode(tprm,parent,tag,depth,htree)

    if depth == 0,
      color = [.94,.94,.94];
    else
      color = handles.nodeNum2Color(tprm.Data.Index);
    end
    
    if depth == 0,
      padding = 10;
      scrollable = 'on';
    else
      padding = 0;
      scrollable = 'off';
    end

    n = numel(getChildrenIdxVisible(tprm));
    if strcmp(tprm.Data.Field,'ROOT'),
      n1 = n;
    else
      n1 = n+1;
    end

    htree.Data.handles.gl1 = uigridlayout(parent,[n1,1],...
      'Tag',['gl_',tag],'Padding',padding+zeros(1,4),...
      'RowHeight',repmat({'fit'},1,n+2),'Scrollable',scrollable);

    if ~strcmp(tprm.Data.Field,'ROOT'),
      s = getDisplayName(tprm.Data);
      s = [repmat('>',[1,depth]),' ',s];
      ti = sprintf('<b>%s</b>: %s',s,tprm.Data.Description);
      htree.Data.handles.title = uilabel('Parent',htree.Data.handles.gl1,...
        'Text',ti,'Interpreter','html','WordWrap','on',...
        'BackgroundColor',color);
    end

    nchildren = numel(tprm.Children);
    for i = 1:nchildren,
      htreecurr = InitTree(tprm.Children(i),htree.Data.handles.gl1,tag,depth+1);
      if i == 1,
        htree.Children = htreecurr;
      else
        htree.Children(end+1) = htreecurr;
      end
    end

  end

  function s = getDisplayName(data)
    s = data.DispNameUse;
    s = cleanDisplayName(s);
    stage = APTParameters.getStage(data.FullPath);
    if handles.labelerObj.trackerIsTwoStage
      if strcmpi(stage,'first'),
        s = [s,' (detection stage)'];
      elseif strcmp(stage,'last'),
        s = [s,' (pose stage)'];
      end
    end    
  end


  function s = cleanDisplayName(s)
    s = regexprep(s,'([a-z])([A-Z])','$1 $2');
  end

  function leafhandles = InitLeaf(tprm,parent,varargin)
    
    [tag,suggestedvalue,extradescr] = myparse(varargin,...
      'tag',tprm.Data.Field,'suggestedvalue',[],'extradescr','');

    leafhandles = struct;
    s = getDisplayName(tprm.Data);
    padding = 0;
    color = handles.nodeNum2Color(tprm.Data.Index);

    leafhandles.gl1 = uigridlayout(parent,[2,1],'RowHeight',{'fit','fit'},'Padding',padding+zeros(1,4),...
        'BackgroundColor',color);
    leafhandles.gl2 = uigridlayout(leafhandles.gl1,[1,3],'ColumnWidth',{'3x','2x','fit'},'Padding',[0,0,0,0],...
        'BackgroundColor',color);
    leafhandles.label = uilabel('Parent',leafhandles.gl2,'Text',s,'FontWeight','bold');
    if iscell(tprm.Data.Type),
      leafhandles.value = uidropdown('Parent',leafhandles.gl2,...
        'Items',tprm.Data.Type,'Value',tprm.Data.Value,...
        'ValueChangedFcn',@cbkValueDropdown,...
        'Tag',tag);
    else
      switch tprm.Data.Type,
        case 'unsigned',
          leafhandles.value = uispinner(leafhandles.gl2,...
            'Limits',[0,inf],'Tag',tag,...
            'ValueChangedFcn',@cbkValueSpinner,...
            'Value',tprm.Data.Value,'RoundFractionalValues','on',...
            'ValueDisplayFormat','%d');
        case 'signed',
          leafhandles.value = uispinner(leafhandles.gl2,...
            'Limits',[-inf,inf],'Tag',tag,...
            'ValueChangedFcn',@cbkValueSpinner,...
            'Value',tprm.Data.Value,'RoundFractionalValues','on',...
            'ValueDisplayFormat','%d');
        case 'float',
          leafhandles.value = uieditfield(leafhandles.gl2,...
            'numeric','Limits',[-inf,inf],'Tag',tag,...
            'ValueChangedFcn',@cbkValueEditField,...
            'Value',tprm.Data.Value);
        case 'prctile',
          leafhandles.value = uieditfield(leafhandles.gl2,...
            'numeric','Limits',[0,100],'Tag',tag,...
            'ValueChangedFcn',@cbkValueEditField,...
            'Value',tprm.Data.Value);
        case 'string',
          leafhandles.value = uieditfield(leafhandles.gl2,...
            'text','Tag',tag,...
            'ValueChangedFcn',@cbkValueEditField,...
            'Value',tprm.Data.Value);
        case 'boolean',
          leafhandles.value = uidropdown('Parent',leafhandles.gl2,...
            'Items',{'False','True'},'ItemsData',[false,true],'ValueIndex',double(tprm.Data.Value)+1,...
            'ValueChangedFcn',@cbkValueBool,...
            'Tag',tag);
        otherwise
          error('Unknown parameter type %s',tprm.Data.Type);
      end
    end
    leafhandles.value.Enable = onIff(tprm.Data.isEditable);
    if ~isempty(tprm.Data.ParamViz),
      leafhandles.tb_viz = uibutton("state",'Parent',leafhandles.gl2,...
        'Text','Viz >>','ValueChangedFcn',@cbkVizButton,...
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
  end

  function htree = InitTree(tprm,parent,tag,depth)

    htree = struct;
    htree.Data = struct;
    htree.Data.handles = struct;
    htree.Data.Field = tprm.Data.Field;
    htree.Children = [];
    if ~tprm.Data.Visible,
      return;
    end
    idxvisible = getChildrenIdxVisible(tprm);
    tag = [tag,'_',tprm.Data.Field];

    if ~isempty(idxvisible),
      htree = InitNode(tprm,parent,tag,depth,htree);
    else
      htree.Data.handles = InitLeaf(tprm,parent,'tag',tag);
    end

  end

  function hkp = InitKeypointParamsButton(parent,color)

    if nargin < 2,
      color = [];
    end

    descr = ['If you augment your training data by flipping horizontally or vertically, you ',...
      'MUST set pairs of corresponding keypoints. '];
    hkp = InitButton(parent,...
      'tag','flippairs','titlestr','Keypoint pairs',...
      'descr',descr,'buttonlabel','Set','color',color,'Callback',@cbkKeypointParams);
    
  end

  function hauto = InitAutoTune()
    
    hauto = struct;    

    % automatically set the parameters based on labels.
    [handles.autoparams,handles.vizdata.autoparams] = apt.compute_auto_params(handles.labelerObj);
    kk = handles.autoparams.keys();

    [horz_flip_prm,vert_flip_prm] = APTParameters.getDataAugmentationFlipParams(handles.tree,false);

    align_trx_theta_prm = APTParameters.getAlignTrxTheta(handles.tree,false);

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
      nprm = handles.tree.findnode(kk{i});
      if nprm.Data.Visible,
        nfields = nfields + 1;
      end
    end

    hauto.gl = uigridlayout(handles.tab_autotune,[nfields+1,1],'RowHeight',repmat({'fit'},[1,nfields+1]),'Scrollable','on');
    parent = hauto.gl;
    hauto.auto = {};

    
    % todo: put these in compute_auto_parameters
    if align_trx_theta_prm.Data.Visible,
      % Using head-tail for the first stage
      align_trx_theta = align_trx_theta_prm.Data.Value;    
      hauto.auto{end+1} = InitLeaf(align_trx_theta_prm,parent,'suggestedvalue',true,...
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
      hauto.auto{end+1} = InitLeaf(horz_flip_prm,parent,'suggestedvalue',suggestedvalue.horz,...
        'extradescr',extradescr);
      flipcolor = handles.nodeNum2Color(horz_flip_prm.Data.Index);
    end
    if vert_flip_prm.Data.Visible,
      hauto.auto{end+1} = InitLeaf(vert_flip_prm,parent,'suggestedvalue',suggestedvalue.vert,...
        'extradescr',extradescr);
      flipcolor = handles.nodeNum2Color(vert_flip_prm.Data.Index);
    end

    if horz_flip_prm.Data.Visible || vert_flip_prm.Data.Visible,
      hauto.auto{end+1} = InitKeypointParamsButton(parent,flipcolor);
    end

    for i = 1:numel(kk),
      k = kk{i};
      nprm = handles.tree.findnode(k);
      if nprm.Data.Visible,
        hauto.auto{end+1} = InitLeaf(nprm,parent,'suggestedvalue',handles.autoparams(k));
      end
    end

    gl = uigridlayout(hauto.gl,[1,3],'Padding',[0,0,0,0]);
    hauto.pb_accept_suggestions = uibutton(gl,'Text','Accept suggestions',...
        'ButtonPushedFcn',@cbkAcceptSuggestions,'Tag','pb_accept_suggestions');
    hauto.pb_accept_suggestions.Layout.Column = 2;

  end

  function cbkStoreKeypointParams(state)
    handles.keypointParamState = state;
  end

  function cbkKeypointParams(src,evt)
    handles.vizobj = ParameterVisualizationKeypointParams();
    handles.vizobj.init(handles.tile_viz,handles.labelerObj,'',handles.tree,@cbkStoreKeypointParams,handles.keypointParamState);
  end

  function buttonhandles = InitButton(parent,varargin)
    
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
  end

  function cbkValueDropdown(src,evt)
    ud = src.UserData;
    value = src.Value;
    updateValue(ud,value,src);
  end

  function cbkValueEditField(src,evt)
    ud = src.UserData;
    value = src.Value;
    updateValue(ud,value,src);
  end

  function cbkValueSpinner(src,evt)
    ud = src.UserData;
    value = src.Value;
    updateValue(ud,value,src);
  end

  function cbkValueBool(src,evt)
    ud = src.UserData;
    value = src.ValueIndex==2;
    updateValue(ud,value,src);
  end

  function updateValue(ud,value,src)
    fprintf('updating value of %s to %s\n',ud.data.FullPath,mat2str(value));
    if nargin < 3,
      src = gobjects(1,0);
    end
    ud.data.Value = value;
    ud.data.UserData(~ishandle(ud.data.UserData)) = [];
    
    for hother = [src,ud.data.UserData],
      if ~isequal(hother,src),
        setUIValue(hother,ud.data.Type,value);
      end
      descr = ud.data.Description;
      if ~isempty(ud.suggestedvalue),

        ss = ['Suggested value: ',mat2str(ud.suggestedvalue)];
        if ~isequal(ud.suggestedvalue,ud.data.Value),
          ss = sprintf('<font color="red"><b>%s</b></font>',ss);
        end
        descr = [descr,'<br/>',ss];
      end
      if ~isempty(ud.extradescr),
        descr = [descr,'<br/>',ud.extradescr];
      end
      ud.hdescr.Text = descr;
    end

    if ~isempty(ud.data.ParamViz) && ud.htb_viz.Value,
      fprintf('calling updateParamViz\n');
      assert(isequal(ud.data.ParamViz,handles.vizid));
      updateParamViz(ud.data);
    end

  end

  function setUIValue(h,type,value)
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
  end

  function cbkAcceptSuggestions(src,evt)

    for i = 1:numel(handles.hauto.auto)
      h = handles.hauto.auto{i};
      if ~isfield(h,'value'),
        continue;
      end
      ud = h.value.UserData;
      if ~isempty(ud.suggestedvalue),
        setUIValue(h.value,ud.data.Type,ud.suggestedvalue)
        updateValue(ud,ud.suggestedvalue);
      end
    end

  end

  function cbkLevelChanged(src,evt)
    lastLevel = PropertyLevelsEnum(src.Value);
    resetTreeVisible();
    APTParameters.filterPropertiesByLevel(handles.tree,lastLevel);
    if handles.istrain,
      resetTabsRest();
    else
      resetLevelTrack();
    end
  end

  function cbkVizButton(src,evt)

    val = src.Value;

    data = src.UserData;
    fprintf('cbkVizButton: %s\n',data.ParamViz);

    if val == 1,
      initParamViz(data);
      handles.tb_viz_curr = src;
    else
      clearParamViz();

    end

  end

  function clearParamViz()
    if ~isempty(handles.vizobj),
      handles.vizobj.clear();
    end
    if ishandle(handles.tile_viz),
      delete(handles.tile_viz.Children);
    else
      handles.tile_viz = tiledlayout(handles.panel_right,'vertical','TileSpacing','compact','Padding','compact');
    end
    handles.vizid = '';
    handles.vizobj = [];
    if ~isempty(handles.tb_viz_curr) && ishandle(handles.tb_viz_curr),
      handles.tb_viz_curr.Value = 0;
      handles.tb_viz_curr = [];
    end
  end

  function initParamViz(data)
    vizid = data.ParamViz;
    clearParamViz();
    handles.vizid = vizid;
    % we are going to ignore paramVizID -- I don't understand its function
    [vizclassname,paramVizID] = ParameterVisualization.parseParamVizSpec(vizid); %#ok<ASGLU>
    handles.vizobj = feval(vizclassname);
    handles.vizobj.init(handles.tile_viz,handles.labelerObj,data.FullPath,handles.tree,handles.vizdata);
  end

  function updateParamViz(data)
    vizid = data.ParamViz;
    fprintf('Updating %s in %s to %s, handles.vizid = %s\n',data.FullPath,vizid,mat2str(data.Value),handles.vizid);
    if ~isequal(handles.vizid,vizid),
      return;
    end
    handles.vizobj.update();
  end

  function cbkApply(src,evt)
    clearParamViz();
    handles.output = {handles.tree.structize(),handles.keypointParamState};
    close(handles.figure);
  end

  function cbkCancel(src,evt)
    close(handles.figure);
  end

end