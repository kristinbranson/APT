function [foci,ecc] = conic_foci(A,B,C,D,E,F)

[type,x0,y0,theta,A2,B2,C2,D2,E2,F2] = conic_standard_form(A,B,C,D,E,F);

if strcmpi(type,'circle')
  foci = [0,0];
  ecc = 0;
elseif strcmpi(type,'ellipse') || strcmpi(typ,'hyperbola'),
  a2 = 1 / min(abs(A2),abs(C2));
  b2 = 1 / max(abs(A2),abs(C2));
  c2 = sqrt(a2 - b2);
  if A2 < C2,
    foci = [-c2,0;c2,0];
  else
    foci = [0,-c2;0,c2];
  end
  ecc = c2 / a2;
elseif strcmpi(type,'parabola'),
  if abs(A2) > abs(C2),
    foci = [0,-E2/4];
  else
    foci = [-D2/4,0];
  end
  ecc = 1;
else % other
  foci = zeros(0,2);
  ecc = 0;
end

nfoci = size(foci,2);
if nfoci > 0,
  % translate
  foci(:,1) = foci(:,1) + x0;
  foci(:,2) = foci(:,2) + y0;
  % rotate
  R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
  foci = foci*R;
end