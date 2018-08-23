function err = ComputeScalingError(params,xl,xp,yl,yp)

cx = params(1);
cy = params(2);
%s = params(3);
s = 1;

xp2l = cx + s*xp;
yp2l = cy + s*yp;

err = sum((xl-xp2l).^2) + sum((yl-yp2l).^2);