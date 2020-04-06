% mvnpdf_2d(x,mu,S)
% x is 2 x n
% mu is 2 x n
% S is 2 x 2 n
% mvnpdf : |2*pi*S|^(-1/2) * exp( -1/2 * (x-mu)'*inv(S)*(x-mu) )

function p = mvnpdf_2d(x,mu,S)

[d,n] = size(x);
assert(d==2);
[d1,n1] = size(mu);
assert(((n1==1) || (n1==n)) && d1 <= 2);
[d1,d2,n2] = size(S);
assert(((n2==1) || (n2==n)) && d1 == 2 && d2 == 2);

dx = x-mu;
invS = reshape(inv_2x2(S),[4,n2]);
dx2 = dx(1,:).^2.*invS(1,:) + dx(1,:).*dx(2,:).*(invS(2,:)+invS(3,:)) + dx(2,:).^2.*invS(4,:);
c = 1./( (2*pi)*sqrt( reshape(det_2x2(S),[1,n2]) ) );
p = exp(-1/2*dx2).*c;


