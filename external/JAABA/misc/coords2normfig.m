function [xout,yout] = coords2normfig(x,y,h)

if ~exist('h','var')
  h = gca;
end

if ~strcmpi(get(h,'type'),'axes'),
  error('Handle must correspond to an axis');
end

xax  = get(h,'xlim');
yax = get(h,'ylim');

pos = get(h,'position');

xout = (x - pos(1))/pos(3)*(xax(2)-xax(1)+1) + xax(1);
yout = (y - pos(2))/pos(4)*(yax(2)-yax(1)+1) + yax(1);

yisreversed = strcmpi(get(h,'ydir'),'reverse');
xisreversed = strcmpi(get(h,'xdir'),'reverse');
if xisreversed,
  xout = (1-(x - xax(1))/(xax(2)-xax(1)))*pos(3) + pos(1);
else
  xout = (x - xax(1))/(xax(2)-xax(1))*pos(3) + pos(1);
end
if yisreversed,
  yout = (1-(y - yax(1))/(yax(2)-yax(1)))*pos(4) + pos(2);
else
  yout = (y - yax(1))/(yax(2)-yax(1))*pos(4) + pos(2);
end
