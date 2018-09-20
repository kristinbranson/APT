function [P,err,Pfront,Pside,ptfront_re,ptside_re] = dlt_2D_to_3D_point(dlt_front,dlt_side,ptfront,ptside,varargin)

[geometricerror,Sfront,Sside] = myparse(varargin,'geometricerror',true,...
  'Sfront',[],'Sside',[]);

if size(ptfront,1)==1 && ndims(ptfront)==2,
  ptfront = ptfront';
end
if size(ptside,1)==1 && ndims(ptside)==2,
  ptside = ptside';
end

a = nan(2,3);
[a(:,1),a(:,2),a(:,3)] = dlt_2D_to_3D(dlt_front,ptfront(1,:),ptfront(2,:));
% X(t) = a(1,1) + a(2,1)*t
% Y(t) = a(1,2) + a(2,2)*t
% Z(t) = a(1,3) + a(2,3)*t
 
[b(:,1),b(:,2),b(:,3)] = dlt_2D_to_3D(dlt_side,ptside(1,:),ptside(2,:));
% X(t) = b(1,1) + b(2,1)*t
% Y(t) = b(1,2) + b(2,2)*t
% Z(t) = b(1,3) + b(2,3)*t
 
% prob a better way to do this wrt reconstruction error, but let's do it
% like this for now
 
snum = sum(b(1,:).*a(2,:),2) + ...
  ( sum(a(1,:).*b(2,:) - b(1,:).*b(2,:),2) ./ sum(b(2,:).^2,2) ) - ...
  sum(a(1,:).*a(2,:),2);
sden = sum( a(2,:).^2, 2) + ...
  ( sum(a(2,:).*b(2,:),2) ./ sum(b(2,:).^2) ) .* sum(b(2,:).*a(2,:));

s = snum./sden;

t = ( sum(a(1,:).*b(2,:) - b(1,:).*b(2,:),2) + s.*sum(a(2,:).*b(2,:),2) ) ./ ...
  sum(b(2,:).^2,2); 

Pfront = a(1,:) + a(2,:).*s;
Pside = b(1,:) + b(2,:).*t;

% Xfront = a(1,1) + a(2,1).*s;
% Yfront = a(1,2) + a(2,2).*s;
% Zfront = a(1,3) + a(2,3).*s;
% 
% Xside = b(1,1) + b(2,1).*t;
% Yside = b(1,2) + b(2,2).*t;
% Zside = b(1,3) + b(2,3).*t;

P = (Pfront+Pside)/2;
err = sum((Pfront-Pside).^2);


% X = (Xside+Xfront)/2;
% Y = (Yside+Yfront)/2;
% Z = (Zside+Zfront)/2;

%err = (Xside-Xfront).^2 + (Yside-Yfront).^2 + (Zside-Zfront).^2;

if geometricerror,
  
  isS = ~isempty(Sfront) && ~isempty(Sside);
  if isS,
    [Rfront,errchol] = cholcov(Sfront,0);
    if errchol ~= 0
      error('Problem finding Cholesky decomposition of Sfront');
    end
    [Rside,errchol] = cholcov(Sfront,0);
    if errchol ~= 0
      error('Problem finding Cholesky decomposition of Sfront');
    end
  end  
  params = optimset('Display','off');
  [P,err] = lsqnonlin(@compute_geometric_error,P,[],[],params);
  
end

[ptfront_re(1),ptfront_re(2)] = dlt_3D_to_2D(dlt_front,P(1),P(2),P(3));
[ptside_re(1),ptside_re(2)] = dlt_3D_to_2D(dlt_side,P(1),P(2),P(3));

  function err = compute_geometric_error(P)
    
    [u_front,v_front] = dlt_3D_to_2D(dlt_front,P(1),P(2),P(3));
    [u_side,v_side] = dlt_3D_to_2D(dlt_side,P(1),P(2),P(3));
    
    if isS,
      err = cat(2,(ptfront'-[u_front,v_front])/Rfront,...
        (ptside'-[u_side,v_side])/Rside);
    else
      err = cat(1,u_front-ptfront(1),...
        u_side-ptside(1),...
        v_front-ptfront(2),...
        v_side-ptside(2));
    end
     
    
  end

end