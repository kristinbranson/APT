function sout = augmentOrTruncateVector(s,n)
% s: vector
% n: positive integer
% 
% sout: vector of length n
%
% If numel(s)>n, then sout is s(1:n). Otherwise, the last element of s is
% replicated.

assert(isvector(s));

ns = numel(s);
if ns>n
  sout = s(1:n);
elseif ns<=n
  sout = s;
  sout(end+1:n) = s(end);
end
