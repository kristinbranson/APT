% D = dist2many2one( X , mu )
%
% inputs: X is an n x d matrix
%         mu is a 1 x d vector
% output: D is a n x 1 vector where D(i) is the squared Euclidean
% distance between X(i,:) and mu.

function D = dist2many2one(X,mu)

% check input sizes

if ndims(X) > 2 | ndims(mu) > 2,
  error('input X should be n x d, input mu should be 1 x d');
end;

[nr,nc] = size(mu);
if nr ~= 1 & nc ~= 1,
  error('input mu should be 1 x d');
end;

% make mu a column vector
d = length(mu);
mu = mu(:);

% make size(X) = n x d
[nr,nc] = size(X);
if nr == d,
  X = X';
elseif nc ~= d,
  error('input X should be n x d, input mu should be 1 x d');
end;

D = row_sum(X.^2) + sum(mu.^2) - 2*X*mu;

