function invA = inv_2x2(A)

[d1,d2,~] = size(A);
assert(d1==2 && d2==2);

invA = cat(2, cat(1,A(2,2,:),-A(2,1,:)), cat(1,-A(1,2,:),A(1,1,:))) ./ ...
  (A(1,1,:).*A(2,2,:) - A(1,2,:).*A(2,1,:));