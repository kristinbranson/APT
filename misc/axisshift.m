function axisshift(ax,xs,ys)
v = axis(ax);
v(1:2) = v(1:2)+xs;
v(3:4) = v(3:4)+ys;
axis(ax,v);