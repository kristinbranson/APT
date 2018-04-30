function J = darkjet(m,alpha)
%DARKJET    Variant of HSV
%   DARKJET(M) returns an M-by-3 matrix containing the jet colormap *
%   alpha. By defaylt, alpha = 0.75

if nargin < 1 || isempty(m),
  J = jet;
else
  J = jet(m);
end

if nargin < 2,
  alpha = .75;
end

J = J*alpha;
