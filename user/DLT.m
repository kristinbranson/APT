classdef DLT 
  
  % Coord Sys Notes
  %
  % The world coord system assigns 3-tuples to points: X=[x;y;z]. The units
  % are in eg cm. Its origin is typically at some object/physical reference 
  % pt.
  % 
  % The camera coord sys assigns 3-tuples to points: U=[u;v;w]. The units
  % are the same as the world coord sys. Its origin is at the camera focal
  % point.
  %
  % To get from the world to camera coord sys, we have U = [R t]*[X;1]
  % where [X;1] is the projective 4-vec corresponding to X. R is a 3x3
  % rotation matrix and t is a 3x1 offset representing the location of the
  % world origin in U-space.
  %
  % The projected/image coord sys assigns 2-tuples to coords: x=[u,v]. The
  % units are in pixels. The u- and v- axis are parallel to the camera u-
  % and v-axes (and point in the same directions/orientations) but are
  % offset so that U=(0,0,1) corresponds to the principal point (u0,v0).
  %
  % To go from camera coords to image coords, we use the camera matrix: 
  % x = K*U where K is the 3x3 camera matrix typically with zero skew and 
  % x is a projective 2-vec (normalize by the 3rd coord to get u and v).
  
  methods (Static)  
    
    function [detval,normval] = checkRot(L)
      [~,~,~,TIO] = DLT.dlt2cam(L);
      detval = det(TIO);
      normval = norm(TIO*TIO.' - eye(3));
    end     
    
    function [gam,s,u0v0,R,om,x0y0z0,t] = dlt2cam(L,varargin)
      % Convert 11-vec DLT params to camera params
      %
      % gam: [2] u/v scale facs, in pixels
      % s: [1] skew
      % u0v0: [2] u/v principal point coords, in pixels
      % R: [3x3] rotation matrix. Given a vector p=[x;y;z] in world coords,
      %   R*p gives the vector in the cam coord sys. Note, depending on L
      %   this may not be a true (orthogonal) rotation 
      % om: [3] rotation vec corresponding to R. Note, if R is not a true
      %   rotation matrix then the relationship of om to R is
      %   indeterminate.
      % x0y0z0: [3] location of focal pt in world coords.
      % t: [3] location of world origin, in cam coords.
      %
      % If you have X = [x;y;z;1] in global/world coords
      % Then 
      %
      %     x = K*[R t]*X
      %
      % is a projective vec x=[u;v;w] for the image coords
      
      tfdiag = myparse(varargin,...
        'tfdiag',false);
      
      szassert(L,[11 1]);
      
      tmpM = [L(1:3)'; L(5:7)'; L(9:11)'];
      tmpA = -[L(4);L(8);1];
      x0y0z0 = tmpM\tmpA;
      
      D2 = 1/sum(L(9:11).^2);
      u0 = D2*(L(1)*L(9)+L(2)*L(10)+L(3)*L(11));
      v0 = D2*(L(5)*L(9)+L(6)*L(10)+L(7)*L(11));
      
      du2 = D2*( (u0*L(9)-L(1))^2 + (u0*L(10)-L(2))^2 + (u0*L(11)-L(3))^2);
      dv2 = D2*( (v0*L(9)-L(5))^2 + (v0*L(10)-L(6))^2 + (v0*L(11)-L(7))^2);
      
      % SIGNS, SIGNS, EVERYWHERE ETC
      %
      % This follows the derivation at www.kwon3d.com.
      %
      % D is equal to 
      %
      %     - (vec-to-focal-pt) dot (camera-z-axis)  [in world coords].
      %
      %   This will be negative if the cam z-axis is oriented away from the
      %   world origin, which is as presented in the derivation. Note that
      %   it does not appear that this must necessarily hold for an
      %   arbitrary rig.
      %
      % If D is indeed negative, then we expect du and dv to be positive
      % per the derivation (the d quantity is positive).
      %
      % At the moment we assert that D<0, du>0, dv>0, and check that the
      % resulting rotation matrix has det(R)>0. 
      
      du = sqrt(du2);
      dv = sqrt(dv2);
      R = [ (u0*L(9)-L(1))/du  (u0*L(10)-L(2))/du  (u0*L(11)-L(3))/du;
            (v0*L(9)-L(5))/dv  (v0*L(10)-L(6))/dv  (v0*L(11)-L(7))/dv;
            L(9)               L(10)                L(11)];
      D = -sqrt(D2);
      R = D*R;
      if det(R)<0
        warning('Sign assumptions incorrect.');      
        R = -R;
      end
      %assert(det(R)>0,'Sign assumptions incorrect.');      

      om = rodrigues(R);
      t = -R*x0y0z0;

      % intrinsic params
      gam = [du;dv];
      s = 0;
      u0v0 = [u0;v0];
      
      AXCOLORS = {[1 0 0] [0 1 0] [1 1 0]};
      if tfdiag
        % plot world axis and cam in world-space
        hFig = figure;
        axs = createsubplots(1,2);

        ax = axs(1);
        hold(ax,'on');
        plot3(ax,[0 1],[0 0],[0 0],'Color',AXCOLORS{1},'LineWidth',5); 
        plot3(ax,[0 0],[0 1],[0 0],'Color',AXCOLORS{2},'LineWidth',5); 
        plot3(ax,[0 0],[0 0],[0 1],'Color',AXCOLORS{3},'LineWidth',5); 
        
        for i=1:3
          plot3(ax,x0y0z0(1)+[0 R(i,1)],x0y0z0(2)+[0 R(i,2)],x0y0z0(3)+[0 R(i,3)],...
            'Color',AXCOLORS{i},'LineWidth',2); 
        end
        
        axis(ax,'square','equal');
        grid(ax,'on');
        view(ax,3);
        
        ax = axs(2);
        [u0,v0] = dlt_3D_to_2D(L,0,0,0);
        [ux,vx] = dlt_3D_to_2D(L,1,0,0);
        [uy,vy] = dlt_3D_to_2D(L,0,1,0);
        [uz,vz] = dlt_3D_to_2D(L,0,0,1);
        hold(ax,'on');
        plot(ax,[u0 ux],[v0 vx],'-','Color',AXCOLORS{1},'LineWidth',5);
        plot(ax,[u0 uy],[v0 vy],'-','Color',AXCOLORS{2},'LineWidth',5);
        plot(ax,[u0 uz],[v0 vz],'-','Color',AXCOLORS{3},'LineWidth',5);
        axis(ax,'square','equal','ij');
        grid(ax,'on');
      end
    end
    
    function L = cam2dlt(gam,s,u0v0,om,x0y0z0)
      
      R = rodrigues(om);
      %x0y0z0 = -R\t;
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
      
      % see sign discussion in dlt2cam.
      assert(du>0 && dv>0,'Expect positive gammas.');
      assert(D<0,'Sign assumption incorrect.');
      
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
   
    % IMPORTANT: does not include distortion
    function [xyz,d] = stereoTriangulate1(...
        uv1,gam1,u0v0_1,R1,x0y0z0_1,...
        uv2,gam2,u0v0_2,R2,x0y0z0_2)
      % uv1/2: [2]
      %
      % xyz: [3]
      % d: scalar triangulation error in world units, minimum euclidean 
      % distance between worldrays for uv1 and uv2
      
%       [gam1,~,u0v0_1,TIO1,~,x0y0z0_1] = DLT.dlt2cam(L1);
%       [gam2,~,u0v0_2,TIO2,~,x0y0z0_2] = DLT.dlt2cam(L2);
      
      xyz1 = DLT.worldRay(uv1,gam1,u0v0_1,R1);
      xyz2 = DLT.worldRay(uv2,gam2,u0v0_2,R2);
      
      [P,Q,d] = CalibratedRig2.stereoTriangulateRays(...
        x0y0z0_1(:),xyz1(:),x0y0z0_2(:),xyz2(:));
      xyz = (P+Q)/2;
    end
    
    % IMPORTANT: does not include distortion
    function xyz = stereoTriangulate2(uv1,L1,uv2,L2)
      u1 = uv1(1);
      v1 = uv1(2);
      u2 = uv2(1);
      v2 = uv2(2);
      
      LHS1 = DLT.hlpLHS(u1,v1,L1);
      LHS2 = DLT.hlpLHS(u2,v2,L2);
      rhs1 = DLT.hlpRHS(u1,v1,L1);
      rhs2 = DLT.hlpRHS(u2,v2,L2);
      
      LHS = [LHS1;LHS2];
      rhs = [rhs1;rhs2];
      
      szassert(LHS,[4 3]);
      szassert(rhs,[4 1]);      
      % Solve (minimize for least squares): 
      %       LHS*[x;y;z] = rhs 
      
      xyz = LHS\rhs;
    end
    function LHS = hlpLHS(u,v,L)
      LHS = [...
        u*L(9)-L(1) u*L(10)-L(2) u*L(11)-L(3);
        v*L(9)-L(5) v*L(10)-L(6) v*L(11)-L(7)];
    end
    function rhs = hlpRHS(u,v,L)
      rhs = [L(4)-u; L(8)-v];
    end
    
    function xyz = worldRay(uv,gam,u0v0,R)
      % Compute ray from focal point towards object corresponding to
      % projected pts uv
      %
      % uv: [2] u/v image coords in pixels
      %
      % xyz: unit vector in world coords. The worldline corresponding to uv
      %   is x0y0z0 + xyz * t
      
      assert(numel(uv)==2);      
      xyz = R'*[ (uv(1)-u0v0(1))/gam(1); (uv(2)-u0v0(2))/gam(2); -1];
      xyz = xyz/norm(xyz);
    end
    
    function uv = camProj(X,gam,s,u0v0,om,t)
      % Assumes params derived with image plane at w=-d
      %
      % X: [nx3]
      %
      % uv: [nx2]
      
      n = size(X,1);
      szassert(X,[n 3]);
      
      % TODO: sign issues
      K = [-gam(1) s u0v0(1);0 -gam(2) u0v0(2);0 0 1];

      R = rodrigues(om);
      P = K*[R t]; % [3x4]
      
      X4 = [X ones(n,1)]';
      x3 = P*X4; % [3xn]
      uv = bsxfun(@rdivide,x3(1:2,:),x3(3,:));
      uv = uv';
    end
    
    function uv = camProj2(X,gam,s,u0v0,om,t,kc)
      % Like .camProj but with distortions
      %
      % X: [nx3]
      %
      % uv: [nx2]
      
      n = size(X,1);
      szassert(X,[n 3]);
     
      R = rodrigues(om);
      X4 = [X ones(n,1)]';
      x3 = [R t]*X4; % [3xn]
      xn = x3(1:2,:)./x3(3,:);
      xn = xn(1:2,:);
      r2 = sum(xn.^2,1);
      
      szassert(xn,[2 n]);
      szassert(r2,[1 n]);
      radfac = 1 + kc(1)*r2 + kc(2)*r2.^2 + kc(5)*r2.^3;
      szassert(radfac,[1 n]);
      tang = [...
        2*kc(3)*xn(1,:).*xn(2,:) + kc(4)*(r2+2*xn(1,:).^2);
        kc(3)*(r2+2*xn(2,:).^2) + 2*kc(4)*xn(1,:).*xn(2,:)
        ];
      szassert(tang,[2 n]);
      xd = xn.*radfac + tang;
      szassert(xd,[2 n]);      
      
      K = [-gam(1) s u0v0(1);0 -gam(2) u0v0(2);0 0 1];
      xd3 = [xd;ones(1,n)];
      
      uv = K*xd3;
      assert(all(uv(3,:)==1));
      uv = uv(1:2,:)';
    end
    
    function [d,d2sum,uvProj,uvRef] = objFcnSkewKC4(p,X,x)
      n = size(X,1);
      szassert(X,[n 3]);
      szassert(x,[n 2]);
      
      assert(numel(p)==15 && iscolumn(p));
      
      gam = p(1:2);
      s = p(3);
      u0v0 = p(4:5);
      om = p(6:8);
      t = p(9:11);
      kc = [p(12:15); 0];
      
      uvRef = x;
      uvProj = DLT.camProj2(X,gam,s,u0v0,om,t,kc);
       
      d2 = sum((uvProj-uvRef).^2,2);
      d2sum = sum(d2); 
      d = sqrt(d2);
      szassert(d,[n 1]); 
    end
    
    function [d,d2sum,uvProj,uvRef] = objFcnNoPP(p,u0v0,X,x)
      % Single cam, no princ pt
      %
      % p: [8] parameter vector
      % u0v0: [2] principal pt
      % X: [nx3] 3D world coords
      % x: [nx2] 2D u/v coords 
      %
      % d: [nx1] vector of eq distances (projected, in u/v space) for each pt
      % d2sum: [1] sum-of-squares of d
      % uv: [nx2] 2D u/v coords projected using (X,p)
      % x: [nx2] Same as x, 2D u/v reference/gt coords
      
      n = size(X,1);
      szassert(X,[n 3]);
      szassert(x,[n 2]);
      
      assert(numel(p)==8);
      assert(numel(u0v0)==2);
      
      gam = p(1:2);
      om = p(3:5);
      t = p(6:8);
      s = 0;
      
      uvRef = x;
      uvProj = DLT.camProj(X,gam,s,u0v0,om,t);
            
      d2 = sum((uvProj-uvRef).^2,2);
      d2sum = sum(d2); 
      d = sqrt(d2);
      szassert(d,[n 1]);      
    end
    
    function [d,d2sum,uvProj,uvRef] = objFcnSkew(p,X,x)
      % Single cam, w/skew
      %
      % p: [11] parameter vector
      % X: [nx3] 3D world coords
      % x: [nx2] 2D u/v coords 
      %
      % d: [nx1] vector of eq distances (projected, in u/v space) for each pt
      % d2sum: [1] sum-of-squares of d
      % uv: [nx2] 2D u/v coords projected using (X,p)
      % x: [nx2] Same as x, 2D u/v reference/gt coords
      
      n = size(X,1);
      szassert(X,[n 3]);
      szassert(x,[n 2]);
      
      assert(numel(p)==11);
      
      gam = p(1:2);
      s = p(3);
      u0v0 = p(4:5);
      om = p(6:8);
      t = p(9:11);
      
      uvRef = x;
      uvProj = DLT.camProj(X,gam,s,u0v0,om,t);
            
      d2 = sum((uvProj-uvRef).^2,2);
      d2sum = sum(d2); 
      d = sqrt(d2);
      szassert(d,[n 1]);      
    end
    function f = objFcnSkew2(p,X,x)
      [~,d2sum] = DLT.objFcnSkew(p,X,x);
      n = size(X,1);
      f = sqrt(d2sum/n);
    end
    
    function [d,d2sum,uv,x] = objFcnNoSkew(p,X,x)
      % Objective function, single cam, no skew
      %
      % p: [10] parameter vector
      % X: [nx3] 3D world coords
      % x: [nx2] 2D u/v coordsfor cam
      %
      % d: [nx1] vector of eq distances (projected, in u/v space) for each pt
      % d2sum: [1] sum-of-squares of d
      % uv: [nx2] 2D u/v coords projected using (X,p)
      % x: [nx2] 2D u/v reference/gt coords 
      
      n = size(X,1);
      szassert(X,[n 3]);
      szassert(x,[n 2]);
      
      gam = p(1:2);
      u0v0 = p(3:4);
      om = p(5:7);
      x0y0z0 = p(8:10);
      
      L = DLT.cam2dlt(gam,0,u0v0,om,x0y0z0);
      [uv(:,1),uv(:,2)] = dlt_3D_to_2D(L,X(:,1),X(:,2),X(:,3));      
      
      d2 = sum((x-uv).^2,2);
      d = sqrt(d2);
      szassert(d,[n 1]);
      
      d2sum = sum(d2); 
    end
    
    function [d,d2Proj1,d2Proj2,d2ReProj1,d2ReProj2,...
        uvCal1,uvCalProj1,...
        uvCal2,uvCalProj2,...
        uvStro1,uvStro2,uvReProj1,uvReProj2] = ...
        objFcnStro(p,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2)
      % Objective function for stereo-calibrated data, no-skew. Includes
      % GT calibration points + stereo-labeled data
      %
      % p: [20] parameter vector. [gam,u0v0,om,x0y0z0] for cam1 then cam2.
      % Xcal: [nx3] 3D world coords, GT calibration points
      % uvCal1: [nx2] 2D u/v coords (in pixels), manually labeled
      %   calibration pts, cam 1
      % uvCal2: [nx2] " cam 2
      % uvStro1: [mx2] 2D u/v coords (in pixels), stereo-labeled fly data, 
      %   cam 1
      % uvStro2: [mx2] " cam 2
      %
      % dSumCal: scalar. sqrt(sum-of-squares) of projection error for 
      %   calibration data (view1 and view2)
      % dSumStro: scalar, sqrt(sum-of-squares) of reprojection error for
      %   stereo-labeled data (view1 and view2)
      
      szassert(p,[20 1]);
      
      gam1 = p(1:2);
      u0v0_1 = p(3:4);
      om1 = p(5:7);
      x0y0z0_1 = p(8:10);
      gam2 = p(11:12);
      u0v0_2 = p(13:14);
      om2 = p(15:17);
      x0y0z0_2 = p(18:20);      
      
      L1 = DLT.cam2dlt(gam1,0,u0v0_1,om1,x0y0z0_1);
      L2 = DLT.cam2dlt(gam2,0,u0v0_2,om2,x0y0z0_2);
     [d,d2Proj1,d2Proj2,d2ReProj1,d2ReProj2,...
        uvCal1,uvCalProj1,...
        uvCal2,uvCalProj2,...
        uvStro1,uvStro2,uvReProj1,uvReProj2] = ...
        DLT.objFcnStroCore(L1,L2,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);
    end
    function [d,d2Proj1,d2Proj2,d2ReProj1,d2ReProj2,...
        uvCal1,uvCalProj1,...
        uvCal2,uvCalProj2,...
        uvStro1,uvStro2,uvReProj1,uvReProj2] = ...
        objFcnStroL(p,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2)
      % p: [22]
      
      szassert(p,[22 1]);
      L1 = p(1:11);
      L2 = p(12:end);
      
      [d,d2Proj1,d2Proj2,d2ReProj1,d2ReProj2,...
        uvCal1,uvCalProj1,...
        uvCal2,uvCalProj2,...
        uvStro1,uvStro2,uvReProj1,uvReProj2] = ...
        DLT.objFcnStroCore(L1,L2,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);
    end
    function [d,d2Proj1,d2Proj2,d2ReProj1,d2ReProj2,...
        uvCal1,uvCalProj1,...
        uvCal2,uvCalProj2,...
        uvStro1,uvStro2,uvReProj1,uvReProj2] = ...
        objFcnStroCore(L1,L2,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2)
      % Xcal1: [n1x3], 3d/world pts for view1 cal
      % Xcal2: [n2x3], " view2
      % uvCal1: [n1x2], 2d manual clicked pts corresponding to Xcal1
      % uvCal2: [n2x2], " view2
      % uvStro1: [mx2], 2d stereo labeled pts view1
      % uvStro2: [mx2], 2d stereo labeled pts view2
      
%       TIO1 = rodrigues(om1);
%       TIO2 = rodrigues(om2);

      n1 = size(Xcal1,1);
      szassert(Xcal1,[n1 3]);
      szassert(uvCal1,[n1 2]);
      
      n2 = size(Xcal2,1);
      szassert(Xcal2,[n2 3]);
      szassert(uvCal2,[n2 2]);
      
      m = size(uvStro1,1);
      szassert(uvStro1,[m 2]);
      szassert(uvStro2,[m 2]);
      
      [uvCalProj1(:,1),uvCalProj1(:,2)] = dlt_3D_to_2D(L1,Xcal1(:,1),Xcal1(:,2),Xcal1(:,3));
      [uvCalProj2(:,1),uvCalProj2(:,2)] = dlt_3D_to_2D(L2,Xcal2(:,1),Xcal2(:,2),Xcal2(:,3));
      d2Proj1 = sum((uvCal1-uvCalProj1).^2,2);
      d2Proj2 = sum((uvCal2-uvCalProj2).^2,2);
      szassert(d2Proj1,[n1 1]);
      szassert(d2Proj2,[n2 1]);
      
      uvReProj1 = nan(m,2);
      uvReProj2 = nan(m,2);
      for i=1:m
        xyz = DLT.stereoTriangulate2(uvStro1(i,:)',L1,uvStro2(i,:)',L2);
        [uvReProj1(i,1),uvReProj1(i,2)] = dlt_3D_to_2D(L1,xyz(1),xyz(2),xyz(3));
        [uvReProj2(i,1),uvReProj2(i,2)] = dlt_3D_to_2D(L2,xyz(1),xyz(2),xyz(3));
      end      
      d2ReProj1 = sum((uvStro1-uvReProj1).^2,2);
      d2ReProj2 = sum((uvStro2-uvReProj2).^2,2);
      szassert(d2ReProj1,[m 1]);
      szassert(d2ReProj2,[m 1]);
            
      % Three equally-weighted parts: 
      % 1. RMS cal data, view1
      % 2. RMS cal data, view2
      % 3. RMS stro data, view1+view2
      d = [sqrt(sum(d2Proj1)/n1); sqrt(sum(d2Proj2)/n2); sqrt(sum(d2ReProj1+d2ReProj2)/2/m)];
    end
    
    function [d,d2Proj1,d2Proj2,d2ReProj1,d2ReProj2,...
        uvCal1,uvCalProj1,...
        uvCal2,uvCalProj2,...
        uvStro1,uvStro2,uvReProj1,uvReProj2] = ...
        objFcnStroSkewKC4(p,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2)
      
      n1 = size(Xcal1,1);
      szassert(Xcal1,[n1 3]);
      szassert(uvCal1,[n1 2]);
      
      n2 = size(Xcal2,1);
      szassert(Xcal2,[n2 3]);
      szassert(uvCal2,[n2 2]);
      
      m = size(uvStro1,1);
      szassert(uvStro1,[m 2]);
      szassert(uvStro2,[m 2]);
      
      szassert(p,[30 1]);
      
      gam1 = p(1:2);
      s1 = p(3);
      u0v0_1 = p(4:5);
      om1 = p(6:8);
      t1 = p(9:11);
      kc1 = [p(12:15); 0];
      gam2 = p(16:17);
      s2 = p(18);
      u0v0_2 = p(19:20);
      om2 = p(21:23);
      t2 = p(24:26);
      kc2 = [p(27:30); 0];
      
      R1 = rodrigues(om1);
      R2 = rodrigues(om2);
      x0y0z0_1 = -R1'*t1;
      x0y0z0_2 = -R2'*t2;
      
      uvCalProj1 = DLT.camProj2(Xcal1,gam1,s1,u0v0_1,om1,t1,kc1);
      uvCalProj2 = DLT.camProj2(Xcal2,gam2,s2,u0v0_2,om2,t2,kc2);
      
      d2Proj1 = sum((uvCal1-uvCalProj1).^2,2);
      d2Proj2 = sum((uvCal2-uvCalProj2).^2,2);
      szassert(d2Proj1,[n1 1]);
      szassert(d2Proj2,[n2 1]);
      
      uvReProj1 = nan(m,2);
      uvReProj2 = nan(m,2);
      for i=1:m
        xyz = DLT.stereoTriangulate1(...
          uvStro1(i,:)',gam1,u0v0_1,R1,x0y0z0_1,...
          uvStro2(i,:)',gam2,u0v0_2,R2,x0y0z0_2);
        xyz = xyz';
        szassert(xyz,[1 3]);
        uvReProj1(i,:) = DLT.camProj2(xyz,gam1,s1,u0v0_1,om1,t1,kc1);
        uvReProj2(i,:) = DLT.camProj2(xyz,gam2,s2,u0v0_2,om2,t2,kc2);
      end      
      d2ReProj1 = sum((uvStro1-uvReProj1).^2,2);
      d2ReProj2 = sum((uvStro2-uvReProj2).^2,2);
      szassert(d2ReProj1,[m 1]);
      szassert(d2ReProj2,[m 1]);
            
      % Three equally-weighted parts: 
      % 1. RMS cal data, view1
      % 2. RMS cal data, view2
      % 3. RMS stro data, view1+view2
      
      %d = [sqrt(sum(d2Proj1)/n1); sqrt(sum(d2Proj2)/n2); sqrt(sum(d2ReProj1+d2ReProj2)/2/m)];
      d = [sqrt(d2Proj1)/n1; sqrt(d2Proj2)/n2; sqrt(d2ReProj1)/m; sqrt(d2ReProj2)/m];
    end
    
    function hFig = vizPorRP(movIdx,uvP,uvRef)
      % uvP: [nx2] projected or reprojected uv
      % uvRef: [nx2] reference
      
      switch movIdx
        case 1
          MOV = 'f:\stephen\data\flp-chrimson_experiments\fly_359_to_373_17_18_16_SS00325norpAmalesFlpChrimson\fly371\C001H001S0020\C001H001S0020_c.avi';
        case 2
          MOV = 'f:\stephen\data\flp-chrimson_experiments\fly_359_to_373_17_18_16_SS00325norpAmalesFlpChrimson\fly371\C002H001S0020\C002H001S0020_c.avi';
      end
      mr = MovieReader; 
      mr.open(MOV);
      im = mr.readframe(1);
      im = im+25;
      
      hFig = figure('units','normalized','outerposition',[0 0 1 1]);
      ax = axes;
      GAMMA = .4;
      mgray = gray(256);
      mgray2 = imadjust(mgray,[],[],GAMMA);
      
      imagesc(ax,im);
      hold(ax,'on');
      axis(ax,'ij','image');
      colormap(ax,mgray2);
      
      hProj = plot(ax,nan,nan,'.','Color',[0 1 1],'MarkerSize',14);
      hRef = plot(ax,nan,nan,'+','Color',[1 .5 .33],'MarkerSize',7,'linewidth',2);
      ax.YTickLabel = [];
      ax.XTickLabel = [];
                  
      n = size(uvP,1);
      szassert(uvP,[n 2]);
      szassert(uvRef,[n 2]);
      drms = sqrt(sum(sum((uvP-uvRef).^2,2))/n);
      
      set(hProj,'XData',uvP(:,1),'YData',uvP(:,2));
      set(hRef,'XData',uvRef(:,1),'YData',uvRef(:,2));
      tstr = sprintf('rms: %.3f',drms);
      title(ax,tstr,'fontweight','bold','interpreter','none');
    end
    
  end
  
end