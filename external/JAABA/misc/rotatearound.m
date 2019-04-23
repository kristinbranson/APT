function x = rotatearound(x0,theta,mu)

[n1,n2] = size(x0);
[m1,m2] = size(mu);
theta = -theta;
R = [cos(theta),  sin(theta),
     -sin(theta), cos(theta)];
didflip = false;
if n1 ~= 2 & n2 == 2,
  x0 = x0';
  n2 = n1;
  didflip = true;
end;
if m1 ~= 2 & m2 == 2,
  mu = mu';
  m2 = m1;
end;

if m2 ~= n2,
  mu = repmat(mu(:,1),[1,n2]);
end;

x = R * (x0 - mu) + mu;
if didflip,
  x = x';
end