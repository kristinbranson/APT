% Function QUAT2EULZYX(q)
% 
% CALLING FUNCTION: obj_function (frame)
% ACTIONS: Convert quaternion to Euler angles in ZYX scheme
% PARENT PROGRAM: Kine_v3_0
% LAST MODIFIED: September 26, 2007 by gwyneth
%
% Input, q, is a unit quaternion representing a rotation.
% The outputs are Euler angles in radians representing a rotation around
% axes in the lab frame, in the following order:
%   1) phi (bank, rotation around x-axis)
%   2) theta (elevation, rotation around y-axis)
%   3) psi (heading, rotation around z-axis)
%
% This corresponds to a rotation matrix composed as: RzRyRx
%
% Note: Euler angles can be thought of as representing rotations around
% fixed axes in the lab frame OR as representing rotations around
% non-orthogonal axes that rotate sequentially as the object is rotated.

function [phi,theta,psi] = quat2eulzyx(q)

e0 = q(1);
ex = q(2);
ey = q(3);
ez = q(4);

% Quaternion to Euler
if round( 10^9*(e0*ey-ex*ez))/10^9 == 0.5
    
    ph = 2*asin(ex/cos(pi/4)) + head;
    th = pi/2;
    ps = 0;
    
elseif round( 10^9*(e0*ey-ex*ez))/10^9 == -0.5
    
    ph = 2*asin(ex/cos(pi/4)) - head;
    th = -pi/2;
    ps = 0;
    
else
    
    ph = atan2( 2*(e0*ex + ey*ez), e0^2+ez^2-ex^2-ey^2 );
    th = -asin( 2*(ex*ez-e0*ey) );
    ps = atan2( 2*(e0*ez + ex*ey), e0^2+ex^2-ey^2-ez^2 );
    
end

phi = ph;
theta = th;
psi = ps;