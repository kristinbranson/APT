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
function changelogaxisunits(factorx,factory,hax)

if nargin < 3,
  hax = gca;
end

chil = get(hax,'children');
oktypes = {'line','patch','text','hggroup'};
ax = axis;
for i = 1:length(chil),
  if ismember(get(chil(i),'type'),oktypes),
    x = get(chil(i),'xdata');
    y = get(chil(i),'ydata');
    set(chil(i),'xdata',x+log(factorx));
    set(chil(i),'ydata',y+log(factory));
  end
end
ax(1:2) = ax(1:2)+log(factorx);
ax(3:4) = ax(3:4)+log(factory);
axis(ax);