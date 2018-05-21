function Q = qMul( Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10 )
% qMul: quaternion multiplication 
% Q = qMul( Q1, Q2, Q3, ..., Q10 );
% IN: 
%     Q1 - first quaternion
%     Q2 - second quaternion
%     Q3 - third quaternion
%     ....
% 
% OUT:
%     Q - output quaternion, Q = Q1*Q2*Q3*Q4*....
%     
% REMARKS:
%     1) Quaternion multiplication is not commutative, i.e. Q1*Q2 != Q2*Q1
%     2) Quaternion multiplication is associative, i.e. Q1*Q2*Q3 = Q1*(Q2*Q3)=(Q1*Q2)*Q3
% 
% VERSION: 03.03.2012

if( nargin >= 2 )
    Q = qMul2( Q1, Q2 );
end
if( nargin >= 3 )
    Q = qMul2( Q, Q3 );
end
if( nargin >= 4 )
    Q = qMul2( Q, Q4 );
end
if( nargin >= 5 )
    Q = qMul2( Q, Q5 );
end
if( nargin >= 6 )
    Q = qMul2( Q, Q6 );
end
if( nargin >= 7 )
    Q = qMul2( Q, Q7 );
end
if( nargin >= 8 )
    Q = qMul2( Q, Q8 );
end
if( nargin >= 9 )
    Q = qMul2( Q, Q9 );
end
if( nargin >= 10)
    Q = qMul2( Q, Q10);
end

function Q = qMul2( Q1, Q2 )
% qMul: quaternion multiplication 
% IN: 
%     Q1 - first quaternion
%     Q2 - second quaternion
% 
% OUT:
%     Q - output quaternion, Q = Q1*Q2
%     
% REMARKS:
%     1) Quaternion multiplication is not commutative, i.e. Q1*Q2 != Q2*Q1
%     2) Quaternion multiplication is associative, i.e. Q1*Q2*Q3 = Q1*(Q2*Q3)=(Q1*Q2)*Q3
% 
% VERSION: 03.03.2012

s1 = Q1(1);
s2 = Q2(1);
v1 = Q1(2:4);
v2 = Q2(2:4);

s =s1*s2 - dot( v1,v2);
v = s1*v2 + s2*v1 + cross( v1, v2 );
v = reshape( v, 3, 1 );
Q = [s;v];
end
end   