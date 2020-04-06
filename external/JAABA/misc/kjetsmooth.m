function J = kjetsmooth(m)
% almost the same as JET, except the first color is black, last color is
% white

if nargin < 1
   m = size(get(gcf,'colormap'),1);
end

mend = max(1,round(m/10));
m = m - 2*mend;

n = ceil(m/4);
u = [(1:1:n)/n ones(1,n-1) (n:-1:1)/n]';
g = ceil(n/2) - (mod(m,4)==1) + (1:length(u))';
r = g + n;
b = g - n;
g(g>m) = [];
r(r>m) = [];
b(b<1) = [];
J = zeros(m,3);
J(r,1) = u(1:length(r));
J(g,2) = u(1:length(g));
J(b,3) = u(end-length(b)+1:end);

J = [bsxfun(@times,linspace(0,1,mend+1)',J(1,:))
  J(2:end-1,:)
  bsxfun(@plus,linspace(0,1,mend+1)',...
  bsxfun(@times,linspace(1,0,mend+1)',J(end,:)))];