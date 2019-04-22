function cm = gold(n)

if nargin == 0,
  n = 64;
end

golds = [255,215,0
  255,215,0
  238,201,0
  205,173,0
  139,117,0]/255;

cm = zeros(n,3);
for i = 1:3,
  cm(:,i) = interp1(linspace(1,n,size(golds,1)),golds(:,i),1:n,'spline');
end
cm = max(0,min(cm,1));