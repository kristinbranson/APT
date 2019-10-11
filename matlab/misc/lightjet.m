function J = lightjet(m,alpha)
%LIGHTJET    Variant of HSV
%   LIGHTJET(M) returns an M-by-3 matrix containing the jet colormap *
%   alpha + (1-alpha). By defaylt, alpha = 0.75

if nargin < 1 || isempty(m),
  J = jet;
else
  J = jet(m);
end

if nargin < 2,
  alpha = .75;
end

assert(alpha <= 1 && alpha >= 0);

J = J*alpha + (1-alpha);
