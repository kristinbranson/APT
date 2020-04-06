% [x1,x2,y1,y2] = ellipse_to_bounding_box(x,y,a,b,theta)

function [x1,x2,y1,y2] = ellipse_to_bounding_box(x,y,a,b,theta)

if abs(cos(theta)) < eps,
    
    x1 = x-b;
    x2 = x+b;
    y1 = y-a;
    y2 = y+a;

elseif abs(cos(theta)) > 1 - eps,

    x1 = x-a;
    x2 = x+a;
    y1 = y-b;
    y2 = y+b;

else

    phix = atan(-(b/a)*tan(theta));
    phix = phix + [0,pi/2,pi,3*pi/2];
    phiy = atan(-(b/a)*cot(theta));
    phiy = phiy + [0,pi/2,pi,3*pi/2];

    xx = x+a*cos(theta)*cos(phix)-b*sin(theta)*sin(phix);
    yy = y-a*sin(theta)*cos(phiy)+b*cos(theta)*sin(phiy);
    x1 = min(xx); x2 = max(xx);
    y1 = min(yy); y2 = max(yy);
    
end;
