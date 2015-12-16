% h = drawcov(mu,S,varargin)
function h = drawcov(mu,S,varargin)

[a,b,theta] = cov2ell(S);
h = ellipsedraw(a,b,mu(1),mu(2),theta,varargin{:});
