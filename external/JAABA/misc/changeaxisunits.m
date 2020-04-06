% changeaxisunits(factorx,factory)
% changeaxisunits(factorx,factory,hax)
% Changes the units of the x and y axes. 
% Suppose the original units of the x axis is frames and the original unit
% of the y-axis is pixels, and we want to change to seconds and
% centimeters, respectively. If a frame is .05 seconds and a pixel is .025
% cm, then we would call
% changeaxisunits(.05,.025)
% currently, changes all lines, patches, and text. 
%
function changeaxisunits(factorx,factory,hax)

if nargin < 3,
  hax = gca;
end

chil = findobj(hax,'-property','xdata','-and','-property','ydata','-and','-not','type','hggroup');
%oktypes = {'line','patch','text','hggroup'};
for i = 1:length(chil),
  if chil(i) == hax,
    continue;
  end
  x = get(chil(i),'xdata');
  y = get(chil(i),'ydata');
  set(chil(i),'xdata',x*factorx);
  set(chil(i),'ydata',y*factory);
end
chil = findobj(hax,'-property','position','-and','units','data');
%oktypes = {'line','patch','text','hggroup'};
for i = 1:length(chil),
  if chil(i) == hax,
    continue;
  end
  pos = get(chil(i),'position');
  pos(1) = pos(1)*factorx;
  pos(2) = pos(2)*factory;
  set(chil(i),'position',pos);
end
ax = axis;
ax(1:2) = ax(1:2)*factorx;
ax(3:4) = ax(3:4)*factory;
axis(ax);

if strcmpi(get(hax,'xtickmode'),'manual'),
  xtick = get(hax,'xtick');
  xtick = xtick*factorx;
  set(hax,'xtick',xtick);
end

if strcmpi(get(hax,'ytickmode'),'manual'),
  ytick = get(hax,'ytick');
  ytick = ytick*factory;
  set(hax,'ytick',ytick);
end