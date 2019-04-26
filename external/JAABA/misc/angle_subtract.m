function dtheta = angle_subtract(theta2,theta1)

theta11 = [theta1, theta1 - pi, theta1 + pi, theta1 + 2*pi, theta1 - 2*pi]';
theta22 = [theta2, theta2 - pi, theta2 + pi, theta2 + 2*pi, theta2 - 2*pi]';
D = dist2(theta11,theta22);
[dmin,imin] = min(D(:));
[i1,i2] = ind2sub(size(D),imin);
dtheta = theta22(i2) - theta11(i1);
