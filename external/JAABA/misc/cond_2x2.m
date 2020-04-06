function c = cond_2x2(A)
%COND   Condition number with respect to inversion.
%   COND2d(X) returns the 2-norm condition number (the ratio of the
%   largest singular value of X to the smallest).  Large condition
%   numbers indicate a nearly singular matrix.
%
%   COND(X,P) returns the condition number of X in 2-norm:
%
%      NORM(X,2) * NORM(INV(X),2). 
%
%   Class support for input X:
%      float: double, single
%
%   See also RCOND, CONDEST, CONDEIG, NORM, NORMEST.

%   Copied from MATLAB's cond function. 

assert(~issparse(A));
[d1,d2,n] = size(A);
assert(d1==2 && d2==2);
A = reshape(A,[4,n]);

% A2 = A'*A 
tmp = A(1,:).*A(3,:) + A(2,:).*A(4,:);
A2 = cat(1,A(1,:).^2 + A(2,:).^2,...
  tmp,tmp,...
  A(3,:).^2 + A(4,:).^2);

s = sqrt(eigs_2x2(A2));
c = zeros([1,n],class(A));

issing = any(s==0,1);
c(issing) = inf;
c(~issing) = max(s(:,~issing),[],1)./min(s(:,~issing),[],1);
