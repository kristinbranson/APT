% d = angledist(theta1,theta2,[modpi=false])

function d = angledist(theta1,theta2,modpi)

if nargin < 3 || ~modpi,
  d = abs(mod(theta1-theta2+pi,2*pi)-pi);
else
  d = abs(mod(theta1-theta2+pi/2,pi)-pi/2);
end
