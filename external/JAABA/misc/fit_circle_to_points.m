function [xc,yc,radius] = fit_circle_to_points(x,y)

x = x(:);
y = y(:);

% solve for parameters a, b, and c in the least-squares sense by
% using the backslash operator
abc = [x y ones(numel(x),1)] \ -(x.^2+y.^2);
a = abc(1); b = abc(2); c = abc(3);

% calculate the location of the center and the radius
xc = -a/2;
yc = -b/2;
radius  =  sqrt((xc^2+yc^2)-c);

