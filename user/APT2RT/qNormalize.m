function Q = qNormalize( Q1 )
% qNormalize: quaternion normalization, i.e. ||Q||=1
% Q = qNormalize( Q1 )
% IN: 
%     Q1 - input quaternion
% 
% OUT:
%     Q - normalized (unity) quaternion, i.e. Q1 = Q / ||Q||
%     
% VERSION: 03.03.2012

Q1 = reshape( Q1, 4, 1 );
if( Q1 ~= 0 )
    Q = Q1 ./ qLength( Q1 );
else
    Q = zeros( 4, 1 );
    fprintf( 'Q1 is 0!\n\r' );
end