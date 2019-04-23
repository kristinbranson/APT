function ell = getellipse(hfig,varargin)

[color] = myparse(varargin,'color','r');

fprintf(['  Use normal button clicks to add points.  A shift-, right-, or\n',...
  '  double-click adds a final point and ends the selection.\n',...
  '  Pressing RETURN or ENTER ends the selection without adding\n',...
  '  a final point.  Pressing BACKSPACE or DELETE removes the\n',...
  '  previously selected point.\n\n']);
[x,y] = getpts(hfig);

% fit an ellipse
[Xc,Yc,A,B,Phi]=ellipsefit(x,y);

hs = ishold;
hold on;
h = ellipsedraw(A,B,Xc,Yc,Phi,'k-');
set(h,'color',color);
plot(x,y,'.','color',color);
if ~hs,
  hold off;
end

ell = struct;
ell.x = Xc;
ell.y = Yc;
ell.a = A;
ell.b = B;
ell.theta = Phi;