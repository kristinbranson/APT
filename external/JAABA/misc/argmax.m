function index = argmax(x,varargin)
%ARGMAX   Index of maximum element.
% ARGMAX(X) returns an index I such that X(I) == MAX(X(:)).
%
% See also MAX, ARGMIN.

if nargin > 1,
  [ignore,index] = max(x,varargin{:});
else
[ignore,index] = max(x(:));
end
