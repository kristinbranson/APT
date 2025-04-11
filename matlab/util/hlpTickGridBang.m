function hlpTickGridBang(axes_all, menu_view_show_tick_labels, menu_view_show_grid)
tfTickOn = strcmp(menu_view_show_tick_labels.Checked,'on');
tfGridOn = strcmp(menu_view_show_grid.Checked,'on');
if tfTickOn || tfGridOn
  set(axes_all,'XTickMode','auto','YTickMode','auto');
else
  set(axes_all,'XTick',[],'YTick',[]);
end
if tfTickOn
  set(axes_all,'XTickLabelMode','auto','YTickLabelMode','auto');
else
  set(axes_all,'XTickLabel',[],'YTickLabel',[]);
end
if tfGridOn
  arrayfun(@(x)grid(x,'on'),axes_all);
else
  arrayfun(@(x)grid(x,'off'),axes_all);
end



