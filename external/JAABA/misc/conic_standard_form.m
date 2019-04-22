% conic_standard_form(A,B,C,D,E,F)
% A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0
% returns the rotation theta and translation (x0,y0) required to put the
% equation in the form
% A * x^2 + B*x*y + C * y^2 + D*x + E*y + F = 0
% if not a parabola, then B = D = E = 0 and F = -1
% if a parabola then B = F = 0 and either
%   A = 0, C = 1, D ~= 0, and E = 0, or
%   A = 1, C = 0, D = 0, and E ~= 0

function [type,x0,y0,theta,A2,B2,C2,D2,E2,F2] = conic_standard_form(A,B,C,D,E,F)

type = 'none';
A2 = 0;
B2 = 0;
C2 = 0;
D2 = 0;
E2 = 0;
F2 = 0;
x0 = 0;
y0 = 0;
theta = 0;
if (A == 0) && (B == 0) && (C == 0),
  return;
end

% determine the type
v = B^2 - 4*A*C;
if v < 0,
  if A == C && B == 0
    type = 'circle';
  else
    type = 'ellipse';
  end
elseif abs(v) <= .00001,
  type = 'parabola';
else
  type = 'hyperbola';
end

% determine the rotation
if A == C
  theta = pi/4;
else
  theta = .5*atan(B/(A-C));
end

% rotate
A1 = A*cos(theta)^2 + B*sin(theta)*cos(theta) + C*sin(theta)^2;
C1 = A*sin(theta)^2 - B*sin(theta)*cos(theta) + C*cos(theta)^2;
D1 = D*cos(theta) + E*sin(theta);
E1 = -D*sin(theta) + E*cos(theta);
F1 = F;

% determine the translation
if strcmpi(type,'parabola'),
  F2 = 0;
  % either C1 will be 0 or A1 will be 0
  if abs(C1) < abs(A1),
    x0 = -D1/(2*A1);
    y0 = -(F1 - D1^2/(4*A1))/E1;
    A2 = 1;
    E2 = E1/A1;
  else
    x0 = -(F1-E1^2/(4*C1))/D1;
    y0 = -E1/(2*C1);
    C2 = 1;
    D2 = D1/C1;
  end
else
  F2 = -1;
  x0 = -D1/(2*A1);
  y0 = -E1/(2*C1);
  
  % translate
  Z = (A1*x0^2 + C1*y0^2 - F1);
  A2 = A1 / Z;
  C2 = C1 / Z;
end
