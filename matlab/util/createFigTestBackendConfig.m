function [hfig,hedit] = createFigTestBackendConfig(figname)
  hfig = dialog('Name',figname,'Color',[0,0,0],'WindowStyle','normal');
  hedit = uicontrol(hfig,'Style','edit','Units','normalized',...
    'Position',[.05,.05,.9,.9],'Enable','inactive','Min',0,'Max',10,...
    'HorizontalAlignment','left','BackgroundColor',[.1,.1,.1],...
    'ForegroundColor',[0,1,0]);
end

