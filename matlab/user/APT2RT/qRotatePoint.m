function Protated = qRotatePoint( P, Qrotation )
% qRotatePoint: rotate a point according to rotation quaternion
% Protated = qRotatePoint( P, Qrotation )
% IN: 
%     P - point which is rotated
%     Qrotation - quaternion describing the rotation (axis and angle)
% 
% OUT:
%     Protated - rotated point 
%
% EXAMPLE:
%     Rotate point (1;2;3) around vector (4;5;6) by an angle of pi/2
%     P = [1;2;3];  % create the point
%     V = [4;5;6];  % create vector around which rotation is performed
%     Qrot = qGetRotQuaternion( pi/2, V );
%     P2 = qRotatePoint( P, Qrotate );  
%     
% VERSION: 03.03.2012

P = reshape( P, 3, 1 );
Q1 = [ 0; P ];
Q = qMul( Qrotation, Q1, qInv( Qrotation ) );
Protated = Q(2:4);