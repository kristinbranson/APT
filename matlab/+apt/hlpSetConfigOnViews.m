function tfAxLimsSpecifiedInCfg = hlpSetConfigOnViews(viewCfg,handles,centerOnTarget)
  axs = handles.axes_all;
  tfAxLimsSpecifiedInCfg = ViewConfig.setCfgOnViews(viewCfg,...
    handles.figs_all,axs,handles.images_all,handles.axes_prev);
  if ~centerOnTarget
    [axs.CameraUpVectorMode] = deal('auto');
    [axs.CameraViewAngleMode] = deal('auto');
    [axs.CameraTargetMode] = deal('auto');
    [axs.CameraPositionMode] = deal('auto');
  end
  [axs.DataAspectRatio] = deal([1 1 1]);
  handles.menu_view_show_tick_labels.Checked = onIff(~isempty(axs(1).XTickLabel));
  handles.menu_view_show_grid.Checked = axs(1).XGrid;
end
