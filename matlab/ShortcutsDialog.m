function hfig = ShortcutsDialog(obj)

lObj = obj.labeler_;

W = 500;
H = 800;
buttonw = 120;

sc = lObj.getShortcuts();
tags = fieldnames(sc);
edit_shortcuts_data = cell(numel(tags),3);
isvalid = false(numel(tags),1);
for i = 1:numel(tags),
  [edit_shortcuts_data{i,1},isvalid(i)] = tag2desc(tags{i});
  edit_shortcuts_data{i,2} = sc.(tags{i});
  edit_shortcuts_data{i,3} = tags{i};
end

hfig = uifigure('Name','Shortcuts',...
  'Units','pixels','Position',[300 300 W H]);
gl = uigridlayout(hfig,[6 1]);
gl.RowHeight = {22,'1x',30,22,'1x',30};

hs = struct;
uilabel(gl,'Text','Editable shortcuts. Control + ','FontWeight','bold',...
  'HorizontalAlignment','center');
hs.edittable = uitable(gl,'Data',edit_shortcuts_data(:,1:2),...
  'ColumnName',{'Function','Shortcut'},...
  'RowName',{},'ColumnWidth',{'1x',150},...
  'SelectionType','row');

gladdremove = uigridlayout(gl,[1 3+2],'ColumnWidth',{'1x',buttonw,buttonw,buttonw,'1x'},'Padding',[0,0,0,0]);
hs.addbutton = uibutton(gladdremove,'Text','Add','ButtonPushedFcn',@Add);
hs.addbutton.Layout.Column = 2;
hs.editbutton = uibutton(gladdremove,'Text','Edit','ButtonPushedFcn',@Edit);
hs.editbutton.Layout.Column = 3;
hs.removebutton = uibutton(gladdremove,'Text','Remove','ButtonPushedFcn',@Remove);
hs.removebutton.Layout.Column = 4;

uilabel(gl,'Text','Fixed shortcuts','FontWeight','bold',...
  'HorizontalAlignment','center');
hs.fixedtable = uitable(gl,...
    'ColumnName',{'Function','Shortcut'},...
    'RowName',{},'ColumnWidth',{'1x',150},...
    'SelectionType','row');

fixed_ctrl_shortcuts = {};
if ~isempty(lObj.lblCore),
  fixed_shortcuts = lObj.lblCore.LabelShortcuts();
  for i = 1:size(fixed_shortcuts,1),
    sc = fixed_shortcuts(i,:);
    mods = sc{3};
    key = sc{2};
    if any(~ismember(mods,{'Shift','Ctrl'})),
      continue;
    end
    if ~ismember('Ctrl',mods),
      continue;
    end
    if ismember('Shift',mods),
      fixed_ctrl_shortcuts{end+1} = upper(key);
    else
      fixed_ctrl_shortcuts{end+1} = key;
    end
  end

  fixed_shortcuts_data = cell(size(fixed_shortcuts,1),2);
  fixed_shortcuts_data(:,1) = fixed_shortcuts(:,1);
  for i = 1:size(fixed_shortcuts,1),
    sc = fixed_shortcuts(i,:);
    mods = sc{3};
    key = sc{2};
    if ~isempty(mods),
      key = [sprintf('%s + ',mods{:}),key];
    end
    fixed_shortcuts_data{i,2} = key;
  end    

  hs.fixedtable.Data = fixed_shortcuts_data;

end

gldonecancel = uigridlayout(gl,[1 4],'Padding',[0,0,0,0],'ColumnWidth',{'1x',buttonw,buttonw,'1x'});

hs.donebutton = uibutton(gldonecancel,'Text','Done','ButtonPushedFcn',@Done);
hs.donebutton.Layout.Column = 2;
hs.cancelbutton = uibutton(gldonecancel,'Text','Cancel','ButtonPushedFcn',@Cancel);
hs.cancelbutton.Layout.Column = 3;

alloptions = struct('handle',{},'Tag',{},'desc',{});
alloptionsinit = false;

centerOnParentFigure(hfig,obj.mainFigure_);

  function isvis = isvisible(handle)

    if isempty(handle),
      isvis = false;
      return;
    end

    if isprop(handle,'Parent') && ~isempty(handle.Parent) && ~isvisible(handle.Parent),
      isvis = false;
      return;
    end

    if ~isprop(handle,'Visible') || ~isprop(handle,'Enable'),
      isvis = true;
      return;
    end

    isvis = strcmpi(handle.Visible,'on') && strcmpi(handle.Enable,'on');

  end

  function [desc] = handle2desc(handle)
    if isprop(handle,'Text') && ~isempty(handle.Text),
      desc = handle.Text;
    elseif isprop(handle,'String') && ~isempty(handle.String),
      desc = handle.String;
    else
      desc = handle.Tag;
    end
    if strcmp(handle.Parent.Type,'uimenu'),
      parent_desc = handle2desc(handle.Parent);
      desc = sprintf('%s > %s',parent_desc,desc);
    end
  end

  function [desc,isvalid] = tag2desc(tag)
    isvalid = true;
    if ~isprop(obj,tag) || isempty(obj.(tag)),
      isvalid = false;
      desc = tag;
    else
      desc = handle2desc(obj.(tag));
    end
  end

  function SetAllFunctions()

    if alloptionsinit,
      return;
    end

    fns = properties(obj);
    allowedtypes =  {'matlab.ui.container.Menu','matlab.ui.control.UIControl',...
      'matlab.ui.control.Button'};
    allowed_uicontrol_styles = {};
    for j = 1:numel(fns),
      fn = fns{j};
      val = obj.(fn);
      if isempty(val) || numel(val) > 1,
        continue;
      end
      if ~ishandle(val) || ~isprop(val,'Callback') || ~isprop(val,'Tag') || isempty(val.Callback),
        continue;
      end
      c = class(val);
      if ~ismember(c,allowedtypes),
        continue;
      end
      if strcmp(c,'matlab.ui.control.UIControl'),
        if ~ismember(val.Style,allowed_uicontrol_styles),
          continue;
        end
      end

      if ~ismember(fn,lObj.controller_.fakeMenuTags) && ~isvisible(val)
        continue;
      end

      desc = handle2desc(val);
      option = struct;
      option.handle = val;
      option.Tag = val.Tag;
      option.desc = desc;
      alloptions(end+1) = option;

    end

    alloptionsinit = true;

  end

  function rest = AvailableShortcuts(curr)

    allshortcuts = cellfun(@char,num2cell(('a'+0):('z'+0)),'Uni',false);
    used = edit_shortcuts_data(:,2)';
    if nargin >= 1,
      used(strcmp(used,curr)) = [];
    end
    rest = setdiff(allshortcuts,[used,fixed_ctrl_shortcuts]);

  end

  function Remove(src,evt)
    sel = get(hs.edittable,'Selection');
    if isempty(sel),
      warndlg('No shortcut selected','','modal');
      return;
    end
    rows = sel(:,1);
    if numel(rows) > 1,
      q = sprintf('Delete %d shortcuts',numel(rows));
    else
      q = sprintf('Delete shortcut %s = Ctrl+%s',edit_shortcuts_data{rows,1},edit_shortcuts_data{rows,2});
    end
    a = questdlg(q,'Really delete shortcut?','Yes','Cancel','Cancel');
    if ~strcmp(a,'Yes'),
      return;
    end
    edit_shortcuts_data(rows,:) = [];
    set(hs.edittable,'Data',edit_shortcuts_data,'Selection',[]);

  end

  function Edit(src,evt)
    sel = get(hs.edittable,'Selection');
    if isempty(sel),
      warndlg('No shortcut selected','','modal');
      return;
    end
    rows = sel(:,1);
    if numel(rows) > 1,
      warndlg('Multiple shortcuts selected. Select only one.','','modal');
      return;
    end
    desc = edit_shortcuts_data{rows,1};
    curr = edit_shortcuts_data{rows,2};
    shortcuts = AvailableShortcuts(curr);

    [sel,ok] = listdlg('ListString',shortcuts,'ListSize',[300,160],...
      'InitialValue',find(strcmp(shortcuts,curr)),...
      'SelectionMode','single',...
      'PromptString',[desc,': Ctrl +'],...
      'Name','Choose new shortcut');
    if ~ok,
      return;
    end
    edit_shortcuts_data{rows,2} = shortcuts{sel};
    set(hs.edittable,'Data',edit_shortcuts_data);

  end

  function Add(src,evt)
    hfig_add = uifigure('Name','Add shortcut',...
      'Units','pixels','Position',[300 300 W 300]);
    gladd = uigridlayout(hfig_add,[3,1]);
    gladd.RowHeight = {50,'1x',50};

    SetAllFunctions();
    [availablefuns,optionidx] = setdiff({alloptions.desc},edit_shortcuts_data(:,1));
    shortcuts = AvailableShortcuts();
    gladd1 = uigridlayout(gladd,[1,2]);
    gladd1.ColumnWidth = {'1x',60};
    dd_fun = uidropdown(gladd1,'ItemsData',{alloptions(optionidx).Tag});
    dd_fun.Items = availablefuns;
    dd_short = uidropdown(gladd1);
    dd_short.Items = shortcuts;
    gladd2 = uigridlayout(gladd,[1,2]);
    gladd2.Layout.Row = 3;
    addokbutton = uibutton(gladd2,'Text','OK','ButtonPushedFcn',@AddOK);
    addokbutton = uibutton(gladd2,'Text','Cancel','ButtonPushedFcn',@(src,evt) delete(hfig_add));
    centerOnParentFigure(hfig_add,hfig);

    uiwait(hfig_add);
    
    function AddOK(src,evt)
      tag = dd_fun.Value;
      short = dd_short.Value;
      addfun = dd_fun.Items{strcmp(dd_fun.ItemsData,tag)};
      edit_shortcuts_data(end+1,:) = {addfun,short,tag};
      set(hs.edittable,'Data',edit_shortcuts_data(:,1:2));
      delete(hfig_add);

    end

  end

  function Done(src,evt)

    newsc = struct;
    for j = 1:size(edit_shortcuts_data,1)
      tag = edit_shortcuts_data{j,3};
      short = edit_shortcuts_data{j,2};
      newsc.(tag) = short;
    end
    lObj.setShortcuts(newsc);

    delete(hfig);

  end

  function Cancel(src,evt)
    delete(hfig);
  end

end