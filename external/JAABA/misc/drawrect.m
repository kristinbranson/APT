function h = drawrect(rect,varargin)

x = [rect(1),rect(1),rect(1)+rect(3),rect(1)+rect(3)];
y = [rect(2),rect(2)+rect(4),rect(2)+rect(4),rect(2)];

h = plot([x,x(1)],[y,y(1)],varargin{:});