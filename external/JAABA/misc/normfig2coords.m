function [xout,yout] = normfig2coords(x,y,h)

if ~exist('h','var')
  h = gca;
end

if ~strcmpi(get(h,'type'),'axes'),
  error('Handle must correspond to an axis');
end

xax  = get(h,'xlim');
yax = get(h,'ylim');

pos = get(h,'position');
yisreversed = strcmpi(get(h,'ydir'),'reverse');
xisreversed = strcmpi(get(h,'xdir'),'reverse');

if xisreversed,
  xout = (1 - (x - pos(1))/pos(3))*(xax(2)-xax(1)) + xax(1);
else
  xout = (x - pos(1))/pos(3)*(xax(2)-xax(1)) + xax(1);
end
if yisreversed,
  yout = (1 - (y - pos(2))/pos(4))*(yax(2)-yax(1)) + yax(1);
else
  yout = (y - pos(2))/pos(4)*(yax(2)-yax(1)) + yax(1);
end
