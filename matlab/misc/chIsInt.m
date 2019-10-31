function tf = chIsInt(xyCH,x,y,expectsign)
assert(isequal(xyCH(end,:),xyCH(1,:)));
n = size(xyCH,1);
for i=1:n-1
  vi0 = xyCH(i,:)-[x y];
  vi1 = xyCH(i+1,:)-[x y];
  th0 = atan2(vi0(2),vi0(1));
  th1 = atan2(vi1(2),vi1(1));
  dtheta = modrange(th1-th0,-pi,pi);
  %disp(dtheta);
  if sign(dtheta)~=expectsign
    tf = false;
    return;
  end
end

tf = true;

  