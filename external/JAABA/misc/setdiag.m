% M = setdiag(M,v,[k])
% sets diagonal k of square matrix M to v
% by default, k = 0
%
% examples:
% >> setdiag(rand(2),inf)
%
% ans =
% 
%        Inf    0.0746
%     0.5355       Inf
%
% >> setdiag(rand(2),inf,1)
% 
% ans =
% 
%     0.6837       Inf
%     0.3338    0.9480
% >> setdiag(rand(3),[1,2,3])
% 
% ans =
% 
%     1.0000    0.5783    0.7431
%     0.9225    2.0000    0.6708
%     0.2568    0.4746    3.0000

function M = setdiag(M,v,k)

if rows(M) ~= cols(M),
  error('Currently only implemented for square matrices.');
end;

if ~exist('k','var'),
  k = 0;
end

n = rows(M);

idx = logical(diag(ones(1,n),k));
M(idx) = v;
