function h = errorpatch(x,y,l,u,varargin)

% make inputs uniform
x = x(:);
y = y(:);
if numel(l) == 1,
  l = repmat(l,size(x));
else
  l = l(:);
end
if ~isnumeric(u),
  varargin = [{u},varargin];
  u = l;
elseif nargin < 4,
  u = l;
elseif numel(u) == 1,
  u = repmat(u,size(x));
else
  u = u(:);
end

% sort by x
[x,order] = sort(x);
y = y(order);
l = l(order);
u = u(order);

h = patch([x;flipud(x)],[y-l;flipud(y+u)],[.7,.7,.7],...
  'EdgeColor','none',varargin{:});