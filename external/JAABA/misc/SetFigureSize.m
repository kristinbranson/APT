function SetFigureSize(hfig,width,height)

screen_size = get(0,'ScreenSize');
old_units = get(hfig,'Units');
set(hfig,'Units',get(0,'Units'));
pos = get(hfig,'Position');
center = pos(1) + pos(3)/2;
middle = pos(2) + pos(4)/2;
pos = [center-width/2,middle-height/2,width,height];

% make sure it is on the screen
screen_size([1,2]) = 1;
if pos(1) < screen_size(1),
  pos(1) = screen_size(1);
end
if pos(2) < screen_size(2),
  pos(2) = screen_size(2);
end
if pos(1)+pos(3) > screen_size(1)+screen_size(3),
  pos(1) = max(screen_size(1),screen_size(1)+screen_size(3)-pos(3));
end
if pos(2)+pos(4) > screen_size(2)+screen_size(4),
  pos(2) = max(screen_size(2),screen_size(2)+screen_size(4)-pos(4));
end

set(hfig,'Position',pos);

% reset units
set(hfig,'Units',old_units);