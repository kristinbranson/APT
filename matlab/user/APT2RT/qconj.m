function Q = qConj( Q1 )
% qConj: quaternion conjugation 
% Q = qConj( Q1 )
% IN: 
%     Q1 - input quaternion
% 
% OUT:
%     Q - output quaternion
%     
% VERSION: 03.03.2012

Q1 = reshape( Q1, 4, 1 );
Q = [Q1(1);-Q1(2:4)];