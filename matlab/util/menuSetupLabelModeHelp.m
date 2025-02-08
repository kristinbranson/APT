function menuSetupLabelModeHelp(handles,labelMode)
% Set .Checked for menu_setup_<variousLabelModes> based on labelMode
menus = fieldnames(handles.setupMenu2LabelMode);
for m = menus(:)',m=m{1}; %#ok<FXSET>
  handles.(m).Checked = 'off';
end
hMenu = handles.labelMode2SetupMenu.(char(labelMode));
hMenu.Checked = 'on';
