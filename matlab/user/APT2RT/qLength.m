function d = qLength( Q )
% qLength: quaternion length (norm) 
% d = qLength( Q )
% IN: 
%     Q - input quaternion
% 
% OUT:
%     d - quaternion's norm (length)
%     
% VERSION: 03.03.2012

d = sqrt( sum( Q .* Q ));

