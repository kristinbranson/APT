% inputs
% S are the covariance matrices. S should be 2 x 2 x N, where N is the number of
% covariance matrices.
% outputs:
% a are the semi-major axis lengths 
% b are the semi-minor axis lengths
% theta are the orientations

function [a,b,theta] = cov2ell(S)

[lambda,U] = eigs_2x2(S);
a = sqrt(lambda(1,:))*2;
b = sqrt(lambda(2,:))*2;
theta = shiftdim(atan2(U(2,1,:),U(1,1,:)),1);
theta = mod(theta+pi/2,pi)-pi/2;