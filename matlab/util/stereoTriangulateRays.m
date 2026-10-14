function [P,Q,d,sc,tc] = stereoTriangulateRays(P0,u,Q0,v)
  % "Closest approach" analysis of two rays: find the points on each ray that
  % come closest to the other ray, and the distance between them.
  %
  % P0, Q0: [3x1], 3D "origin" pts
  % u, v: [3x1], vector emanating from P0, Q0, resp.
  %
  % P: [3x1]. 3D pt lying on P0+s*u that comes closest to Q ray
  % Q: [3x1]. etc.
  % d: Euclidean distance between P and Q
  % sc: value of s which minimimizes distance, ie P = P0 + sc*u
  % tc: etc

  assert(isequal(size(P0),[3 1]));
  assert(isequal(size(u),[3 1]));
  assert(isequal(size(Q0),[3 1]));
  assert(isequal(size(v),[3 1]));

  a = dot(u,u);
  b = dot(u,v);
  c = dot(v,v);
  w0 = P0-Q0;
  d = dot(u,w0);
  e = dot(v,w0);
  sc = (b*e-c*d)/(a*c-b*b);
  tc = (a*e-b*d)/(a*c-b*b);
  P = P0 + sc*u;
  Q = Q0 + tc*v;
  d = norm(P-Q,2);
end  % function
