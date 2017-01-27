classdef DLT 
  
  methods (Static)
  
    function x0y0z0 = misc(L)
      szassert(L,[11 1]);
      tmpM = [L(1:3)'; L(5:7)'; L(9:11)'];
      tmpA = -[L(4);L(8);1];
      x0y0z0 = tmpM\tmpA;
    end
    
    function [gam,s,p0,K,R,om,t] = dlt2cam(L)
      % gam: [2] u/v scale facs
      % s: [1] skew
      % p0: [2] u/v principal point coords
      % K: [3x3] intrinsic mat combining lam,s,p0
      % R: [3x3] rotation matrix
      % om: [3] rotation vec corresponding to R
      % t: [3] extrinsic offset vector
      %
      % If you have X = [x;y;z;1] in global/world coords
      % Then 
      %
      %     x = K*[R t]*X
      %
      % is a projective vec x=[u;v;w] for the image coords
      
      szassert(L,[11 1]);
      
      tmpM = [L(1:3)'; L(5:7)'; L(9:11)'];
      tmpA = -[L(4);L(8);1];
      x0y0z0 = tmpM\tmpA;
      
      D2 = 1/sum(L(9:11).^2);
      D = sqrt(D2);
      u0 = D2*(L(1)*L(9)+L(2)*L(10)+L(3)*L(11));
      v0 = D2*(L(5)*L(9)+L(6)*L(10)+L(7)*L(11));
      
      du2 = D2*( (u0*L(9)-L(1))^2 + (u0*L(10)-L(2))^2 + (u0*L(11)-L(3))^2);
      dv2 = D2*( (v0*L(9)-L(5))^2 + (v0*L(10)-L(6))^2 + (v0*L(11)-L(7))^2);
      du = sqrt(du2);
      dv = sqrt(dv2);
      
      % K
      gam = [du;dv];
      s = 0;
      p0 = [u0;v0];
      K = [gam(1) s u0;0 gam(2) v0;0 0 1];
      
      % R, t, om
      TIO = [ (u0*L(9)-L(1))/du  (u0*L(10)-L(2))/du  (u0*L(11)-L(3))/du;
              (v0*L(9)-L(5))/dv  (v0*L(10)-L(6))/dv  (v0*L(11)-L(7))/dv;
              L(9)               L(10)                L(11)];
      TIO = D*TIO;
      if det(TIO)<0
        TIO = -TIO;
      end  
            
      R = TIO;
      om = rodrigues(R);
      t = -R*x0y0z0;
    end
    
    function L = cam2dlt(gam,s,u0v0,om,t)
      
      R = rodrigues(om);
      x0y0z0 = -R\t;
      x0 = x0y0z0(1);
      y0 = x0y0z0(2);
      z0 = x0y0z0(3);
      u0 = u0v0(1);
      v0 = u0v0(2);
      du = gam(1);
      dv = gam(2);

      if s~=0
        warning('dlt:dlt','Nonzero skew value s. This will be ignored for DLT.');
      end
      
      D = -(x0*R(3,1) + y0*R(3,2) + z0*R(3,3));
      
      L = nan(11,1);
      L(1) = (u0*R(3,1)-du*R(1,1))/D;
      L(2) = (u0*R(3,2)-du*R(1,2))/D;
      L(3) = (u0*R(3,3)-du*R(1,3))/D;
      L(4) = ((du*R(1,1)-u0*R(3,1))*x0 + (du*R(1,2)-u0*R(3,2))*y0 + (du*R(1,3)-u0*R(3,3))*z0)/D;
      L(5) = (v0*R(3,1)-dv*R(2,1))/D;
      L(6) = (v0*R(3,2)-dv*R(2,2))/D;
      L(7) = (v0*R(3,3)-dv*R(2,3))/D;
      L(8) = ((dv*R(2,1)-v0*R(3,1))*x0 + (dv*R(2,2)-v0*R(3,2))*y0 + (dv*R(2,3)-v0*R(3,3))*z0)/D;
      L(9) = R(3,1)/D;
      L(10) = R(3,2)/D;
      L(11) = R(3,3)/D;
    end
    
    function K = camK(gam,s,u0v0)
      K = [ gam(1)  s       u0v0(1); ...
            0       gam(2)  u0v0(2); ...
            0       0       1        ];
    end
    
    function [u,v] = camProj(X,gam,s,u0v0,om,t)
      n = size(X,1);
      szassert(X,[n 3]);
      
      K = DLT.camK(gam,s,u0v0);
      R = rodrigues(om);
      P = K*[R t]; % [3x4]
      
      X4 = [X ones(n,1)]';
      x3 = P*X4; % [3xn]
      uv = bsxfun(@rdivide,x3(1:2,:),x3(3,:));
      uv = uv';
      
      % XXX HACK
      uv(:,1) = 2*u0v0(1)-uv(:,1);
      uv(:,2) = 2*u0v0(2)-uv(:,2);
      
      u = uv(:,1);
      v = uv(:,2);
    end
    
    function [d,uv] = objFcnNoSkew(p,X,x)
      % Create objective function handle, no-skew
      %
      % p: parameter vector
      % X: [nx3] 3D world coords
      % x: [nx2] 2D u/v coords
      %
      % d: [nx1] vector of squared-eq distances (projected, in u/v space) 
      %    for each pt
      % uv: [nx2] 2D u/v coords projected
      
      szassert(p,[10 1]);
      n = size(X,1);
      szassert(X,[n 3]);
      szassert(x,[n 2]);
      
      gam = p(1:2);
      u0v0 = p(3:4);
      om = p(5:7);
      t = p(8:10);
      
      K = DLT.camK(gam,0,u0v0);
      R = rodrigues(om);
      P = K*[R t]; % [3x4]
      
      X4 = [X ones(n,1)]';
      x3 = P*X4; % [3xn]
      uv = bsxfun(@rdivide,x3(1:2,:),x3(3,:));
      uv = uv';
      
      % XXX HACK
      uv(:,1) = 2*u0v0(1)-uv(:,1);
      uv(:,2) = 2*u0v0(2)-uv(:,2);
      
      d = sqrt(sum((x-uv).^2,2));
      szassert(d,[n 1]);      
    end
    
  end
  
end