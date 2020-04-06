function varargout = eigs_2x2(S,neigs)

S = reshape(S,[4,numel(S)/4]);
[n1,n2] = size(S);
if ~exist('neigs'), neigs = 2; end;

% compute the eigenvalues
a = S(1,:) + S(4,:);
b = sqrt(4*S(3,:).*S(2,:)+(S(1,:)-S(4,:)).^2);
lambda = zeros(neigs,n2);
isless = b <= 0;
if any(isless),
  lambda(1,isless) = (a(isless) - b(isless))/2;
  if neigs > 1,
    lambda(2,isless) = (a(isless) + b(isless))/2;
  end;
end;
ismore = ~isless;
if any(ismore),
  lambda(1,ismore) = (a(ismore) + b(ismore))/2;
  if neigs > 1,
    lambda(2,ismore) = (a(ismore) - b(ismore))/2;
  end;
end;
varargout{1} = lambda;

if nargout > 1,
  %%% S * v = lambda * v
  %%% (S(1)-lambda)*a + S(3)*b = 0
  %%% S(2)*a + (S(4)-lambda)*b = 0
  %%% choose a = 1
  %%% b = (lambda - S(1))/S(3)
  %%%   = S(2) / (lambda - S(4)) ?
 
  varargout{2} = zeros(2,neigs,n2);

  a = S(3,:);
  b = lambda(1,:) - S(1,:);
  Z = sqrt(a.^2 + b.^2);
  zeroinds = abs(Z) < eps;
  isfirstv = true(size(zeroinds));
  if any(zeroinds),
    a(zeroinds) = lambda(1,zeroinds) - S(4,zeroinds);
    b(zeroinds) = S(2,zeroinds);
    Z(zeroinds) = sqrt(a(zeroinds).^2 + b(zeroinds).^2);
    zeroinds = abs(Z) < eps;
    isfirstv(zeroinds) = false;
  end;
  if any(zeroinds),
    a(zeroinds) = S(3,zeroinds);
    b(zeroinds) = lambda(2,zeroinds) - S(1,zeroinds);
    Z(zeroinds) = sqrt(a(zeroinds).^2 + b(zeroinds).^2);
    zeroinds = abs(Z) < eps;
  end;
  if any(zeroinds),
    a(zeroinds) = lambda(2,zeroinds) - S(4,zeroinds);
    b(zeroinds) = S(2,zeroinds);
    Z(zeroinds) = sqrt(a(zeroinds).^2 + b(zeroinds).^2);
    zeroinds = abs(Z) < eps;
    isfirstv(zeroinds) = true;
  end;
  if any(zeroinds),
    a(zeroinds) = 1;
    b(zeroinds) = 0;
    Z(zeroinds) = 1;
  end;
  if any(isfirstv),
    varargout{2}(1,1,isfirstv) = a(isfirstv)./Z(isfirstv);
    varargout{2}(2,1,isfirstv) = b(isfirstv)./Z(isfirstv);
    varargout{2}(1,2,isfirstv) = -varargout{2}(2,1,isfirstv);
    varargout{2}(2,2,isfirstv) = varargout{2}(1,1,isfirstv);
  end;
  if ~all(isfirstv),
    varargout{2}(1,2,~isfirstv) = a(isfirstv)./Z(isfirstv);
    varargout{2}(2,2,~isfirstv) = b(isfirstv)./Z(isfirstv);
    varargout{2}(1,1,~isfirstv) = -varargout{2}(2,2,isfirstv);
    varargout{2}(2,1,~isfirstv) = varargout{2}(1,2,isfirstv);
  end;

  % make sure the first and last terms have the same sign
  isneg = sign(varargout{2}(1,1,:)) ~= sign(varargout{2}(2,2,:));
  if any(isneg),
    varargout{2}(:,2,isneg) = -varargout{2}(:,2,isneg);
  end;
  % compute the sign of the determinant
  isneg = varargout{2}(1,1,:).*varargout{2}(2,2,:) < ...
    varargout{2}(1,2,:).*varargout{2}(2,1,:);
  % flip if negative
  if any(isneg),
    varargout{2}(:,:,isneg) = -varargout{2}(:,:,isneg);
  end;
  
end;