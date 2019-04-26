function b = myregress(y,X,varargin)
%MYREGRESS Multiple linear regression using least squares.
%   B = MYREGRESS(Y,X) returns the matrix B of regression coefficients in the
%   linear model Y = X*B.  X is an n-by-dx design matrix, with rows
%   corresponding to observations and columns to predictor variables.  Y is
%   an n-by-dy vector of response observations.

if  nargin < 2
    error('stats:regress:TooFewInputs', ...
          'REGRESS requires at least two input arguments.');
end

weights = myparse(varargin,'weights',ones(size(y)));

% Check that matrix (X) and left hand side (y) have compatible dimensions
[n,ncolX] = size(X);
[n2,ncolY] = size(y);
if n2 ~= n
  error('stats:regress:InvalidData', ...
        'The number of rows in Y must equal the number of rows in X.');
end

% Remove missing values, if any
wasnan = any(isnan(y),2) | any(isnan(X),2);
havenans = any(wasnan);
if havenans
   y(wasnan,:) = [];
   X(wasnan,:) = [];
   n = rows(y);
end

% min (A*X - y)*W*(A*X - y)'
% W*

% Use the rank-revealing QR to remove dependent columns of X.
[Q,R,perm] = qr(X,0);
p = sum(abs(diag(R)) > max(n,ncolX)*eps(R(1)));
if p < ncolX
%    warning('stats:regress:RankDefDesignMat', ...
%            'X is rank deficient to within machine precision.');
    R = R(1:p,1:p);
    Q = Q(:,1:p);
    perm = perm(1:p);
end

% Compute the LS coefficients, filling in zeros in elements corresponding
% to rows of X that were thrown out.
b = zeros(ncolX,ncolY);
b(perm,:) = R \ (Q'*y);
