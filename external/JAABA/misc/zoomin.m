function zoomin(p,factor,hax)

if nargin < 3,
  hax = gca;
end
hfig = get(hax,'parent');
xlim = get(hax,'xlim');
ylim = get(hax,'ylim');
dx = diff(xlim)/factor;
dy = diff(ylim)/factor;
xlim = p(1) + [-1,1]*dx/2;
ylim = p(2) + [-1,1]*dy/2;
set(hax,'xlim',xlim,'ylim',ylim);
