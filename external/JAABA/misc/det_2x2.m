function detX = det_2x2(X)

[d1,d2,~] = size(X);
assert(d1==2 && d2==2);

detX = X(1,1,:).*X(2,2,:)-X(1,2,:).*X(2,1,:);