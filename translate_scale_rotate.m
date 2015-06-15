function [Y1,T,s,R] = translate_scale_rotate(X,Y)
%
% SCALE_ROTATE     [Y1,T,s,R] = translate_scale_rotate(X,Y)
%                  X, Y matrices of plane coordinates: N columns and 2
%                  rows.
%                  Given a collection of points X and another collection Y
%                  Calculate the translation, rotation and scale that maps
%                  Y into its closest approximation of X (i.e. X ~= sRY-T = Y1)
%                  The points X and Y are supposed to be points of the
%                  plane.
%
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]

[dx,Nx] = size(X);
[dy,Ny] = size(Y);
if (dx~=2)|(dy~=2), error('translate_scale_rotate: X is supposed to have 2 rows.'); end;
if Nx~=Ny, error('translate_scale_rotate: X and Y are supposed to have the same n. of columns.'); end;
N = Nx;
Y1 = Y;

CX = mean(X')';
CY = mean(Y')';

%% Center the data
X = X - repmat(CX,1,N);
Y = Y - repmat(CY,1,N);

s_X = sqrt(norm(cov(X')));
s_Y = sqrt(norm(cov(Y')));
s = s_X/s_Y; X = X / s_X; Y = Y / s_Y;

Num = X(2,:)*Y(1,:)' - X(1,:)*Y(2,:)';
Den = X(1,:)*Y(1,:)' + X(2,:)*Y(2,:)';
Norm = sqrt(Num.^2 + Den.^2);
sin_theta = Num/Norm; cos_theta = Den/Norm;
R = [cos_theta -sin_theta; sin_theta cos_theta];

T = s*R*CY-CX; Y1 = s*R*Y1 - repmat(T,1,N);