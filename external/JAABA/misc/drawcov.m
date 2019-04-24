% h = drawcov(mu,S)
function varargout = drawcov(varargin)
if isstruct(varargin{1}),
  mu = varargin{1}.centres;
  S = varargin{1}.covars;
  rest = varargin(2:end);
else
  mu = varargin{1};
  S = varargin{2};
  rest = varargin(3:end);
end;
K = size(mu,1);

hv = ishold;

hold on;
colors = 'rgb';
h = zeros(1,K);
for k = 1:K,
    if ~any(isnan(S)),
      [a,b,theta] = cov2ell(S(:,:,k));
      h(k) = ellipsedraw(a,b,mu(k,1),mu(k,2),theta,colors(mod(k-1,numel(colors))+1),rest{:});
    end;
end;

if ~ishold,
  hold off;
end;

if nargout == 1,
    varargout{1} = h;
end;