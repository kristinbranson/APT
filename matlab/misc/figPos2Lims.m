function [xlo,xhi,ylo,yhi] = figPos2Lims(pos)

assert(size(pos,2)==4);

xlo = pos(:,1);
xhi = pos(:,1)+pos(:,3);
ylo = pos(:,2);
yhi = pos(:,2)+pos(:,4);
