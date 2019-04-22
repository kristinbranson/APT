function hfig = setcurrentfigure(hfig)

if ishandle(hfig) && strcmpi(get(hfig,'type'),'figure'),
  set(0,'CurrentFigure',hfig);
elseif isnaturalnumber(hfig),
  figure(hfig);
else
  hfig = figure;
end