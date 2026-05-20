function hideAllAxesToolbarsInFigureBang(fig)
  % Hide the axes toolbar on every axes in fig.
  hs = findall(fig, '-property', 'Toolbar') ;
  for i = 1 : numel(hs)
    h = hs(i) ;
    htoolbar = get(h, 'Toolbar') ;
    if ishandle(htoolbar)
      set(htoolbar, 'Visible', 'off') ;
    end
  end
end  % function
