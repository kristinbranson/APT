function [xc,yc,radius,h] = fitcircle_manual(hfig,color,ti)

xc = [];
yc = [];
radius = [];
h = [];

if nargin < 1,
  hfig = gcf;
end
hold on;

if nargin < 2,
  color = 'm';
end

if nargin < 3,
  ti = 'Click points on circle';
end
title(ti);
try
  [x,y] = getline(hfig,'closed');
catch %#ok<CTCH>
  return;
end

if isempty(x),
  return;
end

% solve for parameters a, b, and c in the least-squares sense by
% using the backslash operator
abc = [x y ones(length(x),1)] \ -(x.^2+y.^2);
a = abc(1); b = abc(2); c = abc(3);

% calculate the location of the center and the radius
xc = -a/2;
yc = -b/2;
radius  =  sqrt((xc^2+yc^2)-c);

% display the calculated center
h(1) = plot(xc,yc,[color 'x'],'LineWidth',2);
h(2) = text(xc,yc,sprintf('  (%.1f, %.1f)',xc,yc),'Color',color,'FontWeight','bold');

% plot the entire circle
theta = 0:0.01:2*pi;

% use parametric representation of the circle to obtain coordinates
% of points on the circle
Xfit = radius*cos(theta) + xc;
Yfit = radius*sin(theta) + yc;

h(3) = plot(Xfit, Yfit,[color '-'],'LineWidth',2);

message = sprintf('The estimated radius is %2.3f pixels', radius);
h(4) = text(15,15,message,'Color',color,'FontWeight','bold','BackgroundColor','k');