function imEll = drawellipseim(x,y,th,a,b,xg,yg,val)
% Draw an ellipse bitmap 
%
% x/y/th/a/b: ellipse vals
% xg/yg: full xgrid/ygrid (from meshgrid)
% val: foreground/ellipse value
%
% imEll: [same size as xg/yg] image with ellipse drawn with foreground px
%   val, background is 0

szassert(xg,size(yg));

[xgXFormed,ygXFormed] = xformShiftAndRot(xg,yg,x,y,th);
tfIn = (xgXFormed/a).^2 + (ygXFormed/b).^2<=1;

imEll = zeros(size(xg));
imEll(tfIn) = val;
