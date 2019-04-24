%S = axes2cov(a,b,theta)
%
% Input:
% a are the lengths of the semi-major axes
% b are the length of the semi-minor axis
% theta are the angles (ccw from [1,0]? can't remember) 
% Output:
% S (2 x 2 x N) is the covariance matrix s.t. two standard deviations
% corresponds to the input ellipse parameters.  
function S = axes2cov(a,b,theta)

N = length(a);
costheta = cos(theta);
sintheta = sin(theta);
a = (a/2).^2; b = (b/2).^2;
S = zeros(2,2,N);
S(1,1,:) = costheta.^2.*a + sintheta.^2.*b;
S(2,2,:) = sintheta.^2.*a + costheta.^2.*b;
S(1,2,:) = sintheta.*costheta.*(a-b);
S(2,1,:) = S(1,2,:);
