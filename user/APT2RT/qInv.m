function Q = qInv( Q1 )
% qInv: quaternion reciprocal (inverse)
% Q = qInv( Q1 )
% IN: 
%     Q1 - input quaternion
% 
% OUT:
%     Q - reciprocal of Q1, i.e. Q1*Q = 1
%     
% VERSION: 03.03.2012

Q = qConj( Q1 ) ./ qLength( Q1 )^2;