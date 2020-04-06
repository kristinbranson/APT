function [xproj,yproj] = conic_project_hack(type,center,theta,normform,x0,y0)

w = length(x0);

% convert to normal form coords
R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
pt = [x0-center(1),y0-center(2)]*R';
x = pt(:,1);
y = pt(:,2);
A = normform(1); C = normform(3); D = normform(4); E = normform(5);

xfit = nan(w,5);
yfit = nan(w,5);
if strcmpi(type,'parabola'),
  if normform(3) == 0, % x^2 + E*y = 0
    xfit(:,1) = x;
    yfit(:,1) = -x.^2 / E;
    xfit(:,2) = sqrt(-E*y);
    yfit(:,2) = y;
    xfit(:,3) = -xfit(:,2);
    yfit(:,3) = y;
  else % y^2 + D*x = 0
    yfit(:,1) = y;
    xfit(:,1) = -y.^2/D;
    xfit(:,2) = x;
    yfit(:,2) = sqrt(-D*x);
    xfit(:,3) = x;
    yfit(:,3) = -yfit(:,2);
  end
else % ellipse, hyperbola, circle
  yfit(:,1) = y;
  xfit(:,1) = sqrt( (1 - C*y.^2)/A );
  yfit(:,2) = y;
  xfit(:,2) = -xfit(:,1);
  yfit(:,3) = sqrt( (1 - A*x.^2)/C );
  xfit(:,3) = x;
  yfit(:,4) = -yfit(:,3);
  xfit(:,4) = x;
  a = sqrt(1/A);
  b = sqrt(1/C);
  if strcmpi(type,'hyperbola')
    phifit = atan2(y,x);
    a = abs(a); b = abs(b);
    rfit = a*b ./ sqrt(-a^2*sin(phifit).^2 + b^2*cos(phifit).^2);
    xfit(:,5) = rfit.*cos(phifit);
    yfit(:,5) = rfit.*sin(phifit);
  else
    phifit = atan2(y,x);
    rfit = a*b ./ sqrt( a^2*sin(phifit).^2 + b^2*cos(phifit).^2);
    xfit(:,5) = rfit.*cos(phifit);
    yfit(:,5) = rfit.*sin(phifit);
  end
end

notreal = imag(xfit) | imag(yfit);
xfit(notreal) = nan;
yfit(notreal) = nan;
err = (xfit - repmat(x,[1,5])).^2 + (yfit - repmat(y,[1,5])).^2;
[minv,i] = min(err,[],2);
idx = sub2ind([w,5],1:w,i');
xfit = xfit(idx)';
yfit = yfit(idx)';

% transform back
pt = [xfit,yfit]*R;
xproj = pt(:,1) + center(1);
yproj = pt(:,2) + center(2);