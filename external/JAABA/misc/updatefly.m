function updatefly(h,x,y,theta,maj,min)

% draw an isosceles triangle with center (x,y)
% with rotation theta
% with height maj*4
% with base min*4

if isstruct(x),
  trx = x;
  t = y;
  x = trx.x(t);
  y = trx.y(t);
  theta = trx.theta(t);
  maj = trx.a(t);
  min = trx.b(t);
end

if 0,
  
  ellipseupdate(h,maj*2,min*2,x,y,theta);

else

% isosceles triangle not yet rotated or centered
pts = [-maj*2,-min*2,
       -maj*2,min*2,
       maj*2,0];

% rotate
costheta = cos(theta);
sintheta = sin(theta);
R = [costheta,sintheta;-sintheta,costheta];
pts = pts*R;

% translate
pts(:,1) = pts(:,1) + x;
pts(:,2) = pts(:,2) + y;

% plot
set(h,'xdata',pts([1:3,1],1),'ydata',pts([1:3,1],2));

end