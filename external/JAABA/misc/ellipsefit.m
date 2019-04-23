function [varargout]=ellipsefit(x,y)
%ELLIPSEFIT Stable Direct Least Squares Ellipse Fit to Data.
% [Xc,Yc,A,B,Phi,P]=ELLIPSEFIT(X,Y) finds the least squares ellipse that
% best fits the data in X and Y. X and Y must have at least 5 data points.
% Xc and Yc are the x- and y-axis center of the ellipse respectively.
% A and B are the major and minor axis of the ellipse respectively.
% Phi is the radian angle of the major axis with respect to the x-axis.
% P is a vector containing the general conic parameters of the ellipse.
% The conic representation of the ellipse is given by:
%
% P(1)*x^2 + P(2)*x*y + P(3)*y^2 + P(4)*x + P(5)*y + P(6) = 0
%
% S=ELLIPSEFIT(X,Y) returns the output data in a structure with field names
% equal to the variable names given above, e.g., S.Xc, S.Yc, S.A, S.B,
% S.Phi and S.P
%
% Reference: R. Halif and J. Flusser, "Numerically Stable Direct Least
% Squares FItting of Ellipses," Department of Software Engineering, Charles
% University, Czech Republic, 2000.

% Conversion from conic to conventional ellipse equation inspired by
% fit_ellipse.m on MATLAB Central

% D.C. Hanselman, University of Maine, Orono, ME 04469
% Mastering MATLAB 7
% 2005-02-28
% Rotation angle fixed 2005-08-09

%--------------------------------------------------------------------------
x=x(:); % convert data to column vectors
y=y(:);
if numel(x)~=numel(y) || numel(x)<5
   error('X and Y Must be the Same Length and Contain at Least 5 Values.')
end

D1=[x.*x x.*y y.*y]; % quadratic terms
D2=[x y ones(size(x))]; % linear terms
S1=D1'*D1;
S2=D1'*D2;

[Q2,R2]=qr(D2,0);
if condest(R2)>1.0e10
   warning('ellipsefit',...
      'Data is Poorly Conditioned and May Not Represent an Ellipse.')
end
T=-R2\(R2'\S2'); % -inv(S3) * S2'

M=S1+S2*T;
CinvM=[M(3,:)/2; -M(2,:); M(1,:)/2];
[V,na]=eig(CinvM);
c=4*V(1,:).*V(3,:) - V(2,:).^2;
A1=V(:,c>0);
P=[A1; T*A1];

% correct signs if needed
if ~isempty(P),
  P=sign(P(1))*P;
  
  Phi=atan(P(2)/(P(3)-P(1)))/2;
  c=cos(Phi);
  s=sin(Phi);
  
  % rotate the ellipse parallel to x-axis
  Pr=zeros(6,1);
  Pr(1)=P(1)*c*c - P(2)*c*s + P(3)*s*s;
  Pr(2)=2*(P(1)-P(3))*c*s + (c^2-s^2)*P(2);
  Pr(3)=P(1)*s*s + P(2)*s*c + P(3)*c*c;
  Pr(4)=P(4)*c - P(5)*s;
  Pr(5)=P(4)*s + P(5)*c;
  Pr(6)=P(6);
  
  % extract other data
  XcYc=[c s;-s c]*[-Pr(4)/(2*Pr(1));-Pr(5)/(2*Pr(3))];
  Xc=XcYc(1);
  Yc=XcYc(2);
  F=-Pr(6) + Pr(4)^2/(4*Pr(1)) + Pr(5)^2/(4*Pr(3));
  AB=sqrt(F./Pr(1:2:3));
  A=AB(1);
  B=AB(2);
  Phi=-Phi;
  if A<B % x-axis not major axis, so rotate it pi/2
    Phi=Phi-sign(Phi)*pi/2;
    A=AB(2);
    B=AB(1);
  end
  S.Xc=Xc;
  S.Yc=Yc;
  S.A=A;
  S.B=B;
  S.Phi=Phi;
  S.P=P;
else
  S.Xc=nan;
  S.Yc=nan;
  S.A=nan;
  S.B=nan;
  S.Phi=nan;
  S.P=nan;
end
if nargout==1
  varargout{1}=S;
else
  outcell=struct2cell(S);
  varargout=outcell(1:nargout);
end
    
