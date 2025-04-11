classdef OrthoCam
  % OrthoCam  Weak Perspective Calibrated Camera(s)
  %
  % If the z-depth and focal length of a pinhole camera (model) are much 
  % larger than the z-dimension of a target/calibration object, the 
  % perspective information obtained by viewing the calibration object at
  % multiple rotations can be quite small. In this regime, a pinhole
  % camera effectively has an extra DOF, as the z-depth and focal length
  % will be approximately linearly dependent.
  %
  % Single-camera calibration is still possible, but with very high 
  % uncertainty along the linearly dependent (z-depth, focallength) 
  % manifold. It may be impossible to reasonably estimate either the 
  % z-depth of a target or the intrinsic focal length.
  %
  % Stereo calibration is further complicated, as this requires
  % reconciliation of the inferred 3D positions of calibration patterns as
  % viewed from two cameras. Since these 3D positions as given by the
  % single-cam calibrations are highly uncertain along the optical axes,
  % reasonable optimization/reconciliation may become very difficult.
  %
  % OrthoCam addresses these problems by reducing the DOF of the camera
  % model by 1. A camera is modeled as "at infinity" along a particular
  % optical axis. The target is simply projected onto the image plane, and 
  % a scale/zoom factor is applied. The model includes radial distortion 
  % about the optical axis.
  
  % Coord systems
  %
  % World Coords are 3D coords, eg X=[x y z]'. In practice these are 
  % defined relative to calibration patterns (eg origin in "upper left", 
  % x and y axes in-plane, z normal vec).
  %
  % Cam Coords are 2D coords, eg W=[p q]', where the origin is located
  % on the optical axis at infinity.
  %
  % To get from World Coords to Cam Coords, we have W = R2*X + t2, where
  % the last row of R2 is irrelevant/discarded and t2=[tx ty]' is a
  % two-dimensional translation vec representing the location of the 
  % World origin as seen in Cam Coords at infinity. R2 and t2 together 
  % (5 DOF) fully specify an OrthoCam's extrinsic position in the World 
  % Sys.
  %
  % Equivalent to the 5 DOFs (R2,t2) is the triple (Xopt,nopt,phi) where
  % Xopt=[xopt yopt 0]' is the intersection of the optical axis with the
  % z=0 World plane, nopt=[nx ny nz]' is the unit vector pointing from Xopt 
  % towards the camera at infinity along the optical axis, and phi 
  % specifies the rotation of the Cameras x-y axes about its optical axis. 
  % Both sets (R2,t2) and (Xopt,nopt,phi) fully specify an OrthoCam's
  % extrinsic position with 5 DOF.
  %
  % Image coords are 2D coords, eg I = [u v]', where I=[1 1] is the
  % upper-left pixel etc. To go from Cam Coords to Image Coords, we have
  % the usual Wd = W*radialdistortion(r) and [I;1] = K*[Wd;1].
  
  methods (Static) % single-cam calib
    function t = summarizeIntrinsics(p,nCalIm)
      [mx,my,u0,v0,k1,k2] = OrthoCam.unpack1cam(p,nCalIm);
      t = table(mx,my,u0,v0,k1,k2);
    end
    function p = pack1cam(mx,my,u0,v0,k1,k2,r2vecs,t2vecs)
      sclrassert(mx);
      sclrassert(my);
      sclrassert(u0);
      sclrassert(v0);
      sclrassert(k1);
      sclrassert(k2);
      nCalIm = size(r2vecs,2);
      szassert(r2vecs,[3 nCalIm]);
      szassert(t2vecs,[2 nCalIm]);
      p = [mx;my;u0;v0;k1;k2;r2vecs(:);t2vecs(:)];
      szassert(p,[6+nCalIm*3+nCalIm*2 1]);
    end
    function [mx,my,u0,v0,k1,k2,r2vecs,t2vecs] = unpack1cam(p,nCalIm)
      p = p(:); % lsqnonlin is calling this with a row apparently due to lb/ub
      szassert(p,[6+nCalIm*3+nCalIm*2 1]);
      mx = p(1);
      my = p(2);
      u0 = p(3);
      v0 = p(4);
      k1 = p(5);
      k2 = p(6);
      r2vecs = reshape(p(7:7+nCalIm*3-1),3,nCalIm);
      t2vecs = reshape(p(7+nCalIm*3:7+nCalIm*3+nCalIm*2-1),2,nCalIm);      
    end    
    function [d,dsum] = oFcn(p,nCalIm,calibWorldPtsXYZ,calibImPts)
      % Objective fcn for Orthogonal projection single-cam calib
      %
      % p = [mx; my; u0; v0; k1; k2; r2vecs; t2vecs] where
      %   r2vecs: [3xnCalIm] rotation vecs for calib images
      %   t2vecs: [2xnCalIm] tx ty vecs for calib images
      %   all others: scalars
      %
      % calibWorldPtsXYZ: [3xnCalPt] calibration world pts (x, y in calib pattern world frame)
      % calibImPts: [2 x nCalPt x nCalIm] x, y image pts for each cal pattern/pt
      %
      % d: [nCalPt*nCalIm x 1] euclidean dist reproj err for each cal pt
            
      [mx,my,u0,v0,k1,k2,r2vecs,t2vecs] = OrthoCam.unpack1cam(p,nCalIm);
      
      % compute projected pts
      nCalPt = size(calibWorldPtsXYZ,2);
      szassert(calibWorldPtsXYZ,[3 nCalPt]);
      szassert(calibImPts,[2 nCalPt nCalIm]);
      
      uvAll = nan(2,nCalPt,nCalIm);
      for iCalIm=1:nCalIm
        R = vision.internal.calibration.rodriguesVectorToMatrix(r2vecs(:,iCalIm));
        R2 = R(1:2,:);
        t2 = t2vecs(:,iCalIm);
        
        X2 = R2*calibWorldPtsXYZ + t2;
        szassert(X2,[2 nCalPt]);
        r2 = sum(X2.^2,1); % [1xnCalPt]
        distort = 1 + k1*r2 + k2*r2.^2; % [1xnCalPt]
        
        uv = [ mx*X2(1,:).*distort + u0 ; ...
               my*X2(2,:).*distort + v0 ]; % [2xnCalPt]             
        uvAll(:,:,iCalIm) = uv;
      end
      
      % compute RP err/residual
      d2 = sum((uvAll-calibImPts).^2,1); % [1 nCalPt nCalIm]
      d2 = d2(:); % [nCalPt nCalIm];
      d = sqrt(d2);
      
      dsum = sum(d);
    end
    function d = oFcn1D(p,nCalIm,calibWorldPtsXYZ,calibImPts)
      d = OrthoCam.oFcn(p,nCalIm,calibWorldPtsXYZ,calibImPts);
      d = sqrt(mean(d.^2));
    end
    function uv = project(X,R2,t2,k1,k2,mx,my,u0,v0)
      % Project world pts to image using intrinsic params
      %
      % X: [3xnpts] World coords
      % R2: [3x3], rotation mat for WorldSys->CamSys
      % t2: [2x1], translation vec for WorldSys->CamSys
      % k1, k2: radial distort
      % mx, my: zoom
      % u0, v0: principal pts
      %
      % uv: [2xnpts] Image pts
            
      npts = size(X,2);
      szassert(X,[3 npts]);
      
      W = R2(1:2,:)*X + t2;
      szassert(W,[2 npts]);
      uv = OrthoCam.normalized2projected(mx,my,u0,v0,k1,k2,W);
    end
    function [x0y0,n,x1y0,x0y1,ijkCamWorld] = opticalCenter(R2cam,t2cam)
      % Find the "optical center" for a cam; the WorldPoint (x0,y0,0) where
      % the cam's optical axis intersects the World plane z=0
      %
      % R2cam, t2cam: Rot, translation to go from World->cam
      %
      % x0y0: [2x1] [x0;y0] optical center 
      % n: [3x1] [nx;ny;nz] normal vec pointing from optical center to cam
      %   (at infinity) along optical axis. Cam is assumed to be at
      %   negative z.
      % x1y0: [2x1] [x;y] that gets mapped to uv=[1;0]
      % x0y1: [2x1]    " uv=[0;1]
      % ijkCamWorld: [3x3]. Definition of "Camera World Coords". These are
      %   world coords with:
      %   * Origin at WorldCoords=(x0,y0,0), ie the optical center
      %   * x-axis aligned with cam x-axis
      %   * y-axis aligned with cam y-axis
      %   * z-axis equal to -n
      %   The columns of ijkCamWorld are the CamWorldCoords i, j, and k 
      %   unit vectors in the original WorldCoordSys.
      
      szassert(R2cam,[3 3]);
      szassert(t2cam,[2 1]);
      
      R12 = R2cam(1:2,1:2);
      x0y0 = -R12\t2cam;
      
      n = cross(R2cam(1,:),R2cam(2,:));
      if n(3)>0
        n = -n;
      end
      n = n/sqrt(sum(n.^2));
      n = n(:);
      
      x1y0 = R12\([1;0]-t2cam);
      x0y1 = R12\([0;1]-t2cam);
      
      xcam = [x1y0-x0y0;0]; % vector in z=0 plane pointing to positive cam1-x (when projected)
      ycam = [x0y1-x0y0;0];
      % remove components pxcam1,... along n1 and n2
      xcam = xcam-dot(xcam,n)*n; % Vector normal to n that projects to cam1-x
      ycam = ycam-dot(ycam,n)*n;
      xcam = xcam/norm(xcam); % unit vector, normal to n, that projects to cam1-x
      ycam = ycam/norm(ycam);
      
      ijkCamWorld = [xcam(:) ycam(:) -n(:)];
    end
    function [theta,phi,az,el] = azEl(n)
      theta = acos(n(3)/norm(n));
      phi = atan2(n(2),n(1));      
      if 0<=phi && phi<=pi/2
        az = phi+pi/2;
      elseif pi/2<phi && phi<pi
        az = -(3/2*pi-phi);
      elseif -pi/2<phi && phi<0
        az = pi/2+phi;
      else % phi in (-pi/2,-pi)
        az = phi+pi/2;
      end
      el = pi/2-theta;
    end
    function uv = normalized2projected(mx,my,u0,v0,k1,k2,pq)
      % pq: [2xn]
      % uv: [2xn]
      
      assert(size(pq,1)==2);
      
      r2 = sum(pq.^2,1); % [1xn]
      distort = 1 + k1*r2 + k2*r2.^2; % [1xn]
      
      pqD = pq.*distort;
      uv = [ mx*pqD(1,:) + u0 ; ...
             my*pqD(2,:) + v0 ]; % [2xn]  
    end
    function pq = projected2normalized(mx,my,u0,v0,k1,k2,uv)
      % uv: [2xn] 
      % pq: [2xn]
      
      n = size(uv,2);
      szassert(uv,[2 n]);
      
      udel = uv(1,:)-u0;
      vdel = uv(2,:)-v0;
      th = atan2(mx*vdel,my*udel);
      szassert(th,[1 n]);
      
      rfuncreate = @(zUdel,zVdel) @(zR) zR^2*(1+k1*zR^2+k2*zR^4)^2 - zUdel^2/mx^2 - zVdel^2/my^2;
      r = nan(1,n);
      for i=1:n
        rfun = rfuncreate(udel(i),vdel(i));
        r(i) = fzero(rfun,0);
      end
      r = abs(r);
      
      p = r.*cos(th);
      q = r.*sin(th);
      pq = [p;q];
      szassert(pq,[2 n]);
    end
    function p0 = p0default1cam(nCalIm)
      mx0 = 255;
      my0 = 255;
      u0 = 384;
      v0 = 256;
      k1_0 = 0;
      k2_0 = 0;
      r2vecs0 = zeros(3,nCalIm);
      t2vecs0 = zeros(2,nCalIm);
      
      p0 = OrthoCam.pack1cam(mx0,my0,u0,v0,k1_0,k2_0,r2vecs0,t2vecs0);
    end
    function p0 = p0fromRsTs(Rmats,Ts)
      % Create a p0 vector from ML-generated Rmatrices and T vecs

      nCalIm = size(Rmats,3);
      szassert(Rmats,[3 3 nCalIm]);
      szassert(Ts,[nCalIm 3]);
      r2vecs = nan(3,nCalIm);
      t2vecs = nan(2,nCalIm);
      for iCalIm=1:nCalIm
        r2vecs(:,iCalIm) = vision.internal.calibration.rodriguesMatrixToVector(Rmats(:,:,iCalIm));
        t2vecs(:,iCalIm) = Ts(iCalIm,1:2)';
      end
      
      [mx,my,u0,v0,k1,k2,r2vecs0,t2vecs0] = ...
        OrthoCam.unpack1cam(OrthoCam.p0default1cam(nCalIm),nCalIm);
      szassert(r2vecs,size(r2vecs0));
      szassert(t2vecs,size(t2vecs0));
      p0 = OrthoCam.pack1cam(mx,my,u0,v0,k1,k2,r2vecs,t2vecs);
    end
    function opts = defaultopts1cam()
      opts = optimset;
      opts.Display = 'iter';
      opts.TolFun = 1e-8;
      opts.TolX = 1e-8;
      opts.MaxFunEvals = 1e6;
      opts.MaxIter = 1e3;
    end
    function [pOpt,oFcn] = calibrate1cam(nCalIm,worldPoints,imPtsUV,p0,varargin)
      % p0: optional
      
      [extonly,opts] = myparse(varargin,...
        'extonly',false,...
        'opts',[]);      
      if isempty(opts)
        opts = OrthoCam.defaultopts1cam();
      end
      
      nPts = size(worldPoints,1);
      szassert(worldPoints,[nPts 2]);
      worldPtsXYZ = [worldPoints zeros(nPts,1)]';           
      szassert(imPtsUV,[nPts 2 nCalIm]);
      calibImPts = permute(imPtsUV,[2 1 3]);
      
      oFcn = @(p)OrthoCam.oFcn(p,nCalIm,worldPtsXYZ,calibImPts);
      
      if exist('p0','var')==0 || isempty(p0)
        p0 = OrthoCam.p0default1cam(nCalIm);
      end
      [~,dsum0] = oFcn(p0);
      fprintf('Starting residual: %.4g\n',dsum0);      
      
      if extonly
%         lb = -inf(size(pOpt));
%         ub = inf(size(pOpt));
%         lb(1:6) = pOpt(1:6);
%         ub(1:6) = pOpt(1:6);
        [pOpt,resnorm,res] = lsqnonlin(oFcn,p0,p0(1:6),p0(1:6),opts);
      else
        [pOpt,resnorm,res] = lsqnonlin(oFcn,p0,[],[],opts);
      end

      [~,dsum1] = oFcn(pOpt);
      fprintf('Ending residual: %.4g\n',dsum1);
    end    
    function [hFig,tffliped,r2vecs,t2vecs] = viewExtrinsics1cam(worldPts,r2vecs,t2vecs,varargin)
      [dOptAx,patByPat] = myparse(varargin,...
        'dOptAx',10,... % length of optical axis to plot (world coords)
        'patByPat',false... % if true, scroll through patterns one by one
        );
      
      nPts = size(worldPts,1);
      szassert(worldPts,[nPts 2]);
      patPtsXYZ = worldPts';
      patPtsXYZ(3,:) = 0; % z=0
      nPat = size(r2vecs,2);
      szassert(r2vecs,[3 nPat]);
      szassert(t2vecs,[2 nPat]);
            
      % z-depth of patterns is undefined for single-cam. Assume 0, ie
      % pattern origin is at z=0 in cam sys.
      t2vecs(3,:) = 0;
      
      hFig = figure('Name','OrthoCam: Calibration Extrinsics',...
        'units','normalized','outerposition',[0 0 1 1]);
      ax = axes;
      hold(ax,'on');
      
      optCtr1 = [0;0;0];
      n1 = [0;0;-1];
      ijkCam1 = eye(3);
      DX = 1;
      optAx1 = [optCtr1 optCtr1+n1*dOptAx];
      optAx1Plus = optCtr1+n1*(dOptAx+DX);
      optAx1Mid = optCtr1+n1*dOptAx/2;
      plot3(optAx1(1,:),optAx1(2,:),optAx1(3,:),'--','linewidth',2,'color',[0 0 0]);
      BROWN = [139 69 19]/255;
      text(optAx1Plus(1),optAx1Plus(2),optAx1Plus(3),'C',...
        'fontweight','bold','fontsize',12,'color',BROWN);

      % Check: x0y0cam1+pxcam1 should project to [1 0] etc
      % draw pxcam1,... at dOptAx/2
      optAxMid1CamXax = [optAx1Mid optAx1Mid+ijkCam1(:,1)];
      optAxMid1CamYax = [optAx1Mid optAx1Mid+ijkCam1(:,2)];
      optAxMid1CamXaxPlus = optAx1Mid+1.5*ijkCam1(:,1);
      optAxMid1CamYaxPlus = optAx1Mid+1.5*ijkCam1(:,2);
      
      plot3(optAxMid1CamXax(1,:),optAxMid1CamXax(2,:),optAxMid1CamXax(3,:),'b-','linewidth',2);
      plot3(optAxMid1CamYax(1,:),optAxMid1CamYax(2,:),optAxMid1CamYax(3,:),'b-','linewidth',2);
      text(optAxMid1CamXaxPlus(1),optAxMid1CamXaxPlus(2),optAxMid1CamXaxPlus(3),...
        'x','fontweight','bold','fontsize',9,'color',[0 0 1]);
      text(optAxMid1CamYaxPlus(1),optAxMid1CamYaxPlus(2),optAxMid1CamYaxPlus(3),...
        'y','fontweight','bold','fontsize',9,'color',[0 0 1]);

      grid on;
      tstr = sprintf('%d pats',nPat);
      title(ax,tstr,'fontweight','bold','interpreter','tex');
      xlabel(ax,'x (mm)','fontweight','bold');
      ylabel(ax,'y (mm)','fontweight','bold');
      zlabel(ax,'z (mm)','fontweight','bold');  
      axis(ax,'square');      
      axis(ax,'equal');
      view(0,0);
      
      % plot the pats
      patPtsMins = min(patPtsXYZ,[],2);
      patPtsMaxs = max(patPtsXYZ,[],2);
      patX0 = patPtsMins(1);
      patX1 = patPtsMaxs(1);
      patY0 = patPtsMins(2);
      patY1 = patPtsMaxs(2);
      patZ0 = patPtsMins(3);
      patZ1 = patPtsMaxs(3);
      assert(patZ0==patZ1);
      patPtsCorners = [ ...
        patX0 patX1 patX1 patX0; ...
        patY0 patY0 patY1 patY1; ...
        patZ0 patZ0 patZ0 patZ0; ];
      clrs = jet(nPat);
      hPat = gobjects(3,1);
      tffliped = false(nPat,1);
      iPat = 1;
      while iPat<=nPat
        r2vecCurr = r2vecs(:,iPat);
        t2vecCurr = t2vecs(:,iPat);
        if patByPat
          deleteValidGraphicsHandles(hPat);
        end
        
        RPatI2World = vision.internal.calibration.rodriguesVectorToMatrix(r2vecCurr);
        tPatI2World = t2vecCurr;
        patPtsCornersWorld = RPatI2World*patPtsCorners + tPatI2World;
        hPat(1) = fill3(patPtsCornersWorld(1,:),patPtsCornersWorld(2,:),...
              patPtsCornersWorld(3,:),clrs(iPat,:),'FaceAlpha',0.5);
            
        % plot origin + yaxis in bold
        orig = patPtsCornersWorld(:,1);
        hPat(2) = plot3(orig(1),orig(2),orig(3),'.','markersize',26,'color',[0 0 0]);
        hPat(3) = plot3(patPtsCornersWorld(1,[1 4]),patPtsCornersWorld(2,[1 4]),...
          patPtsCornersWorld(3,[1 4]),'-','linewidth',3,'color',[0 0 0]);
        
        if patByPat
          in = input(sprintf('Pattern %d/%d. Enter -1 for flip.',iPat,nPat));
          if isequal(in,-1)
            assert(t2vecCurr(3)==0);
            [r2vecs(:,iPat),t2vecs(1:2,iPat)] = OrthoCam.flipPattern(r2vecCurr,t2vecCurr(1:2));
            % iPat unchanged
            tffliped(iPat) = ~tffliped(iPat);
          else
            iPat = iPat+1;
          end
        end
      end
      
      assert(all(ismembertol(t2vecs(3,:),0)));
      t2vecs = t2vecs(1:2,:);
    end
    function [Rp,tp,Q2theta] = computeDualPattern(R,t,c,khat)
      % Compute "dual" pattern extrinsic 
      %
      % R: [3x3]. With t, implicitly defines extrinsic position of calpat
      % t: [3x1].
      % c: [3x1]. Location of calibration pattern Center in calpat coords
      % khat: [3x1]. Unit optical vector pointing to cam1 at infinity, in worldcoords
      
      nhat = R*[0;0;-1]; % unit normal vector for pattern, in worldcoords
      assert(ismembertol(norm(nhat),1),'nhat is not a unit vec.');
      assert(ismembertol(norm(khat),1),'khat is not a unit vec.');      
      ehat = cross(nhat,khat);
      ehat = ehat/norm(ehat);
      
      theta = acos(dot(nhat,khat));
      Q2theta = vision.internal.calibration.rodriguesVectorToMatrix(2*theta*ehat);
      % Q2theta rotates pattern into dual pattern
      
      Rp = Q2theta*R;
      tp = -Q2theta*R*c + R*c + t;
      
%       nhat2 = Q2theta*nhat;
%       ehat2 = cross(nhat2,khat);
      
      fprintf(1,'Check: %.3f == 0\n',norm(R*c+t - (Rp*c+tp)));
    end
    function [r2vecnew,t2vecnew] = flipPattern(r2vec,t2vec,varargin)
      % Flip pattern to dual pattern
      %
      % r2vec: [3] rot vector for pattern
      % t2vec: [2] translation 2-vec for pattern
      %
      % r2vecnew: [3] rot vector for flipped/dual pattern
      % t2vecnew: [2] etc
      
      xyPatCtr = myparse(varargin,...
        'xyPatCtr',[4;2.5]*.1); % center of pattern in pat-coords. Default 20170605: 8x5 pattern, .1mm checkboard size
      
      assert(numel(r2vec)==3);
      assert(numel(t2vec)==2);
      assert(numel(xyPatCtr)==2);
      
      R = vision.internal.calibration.rodriguesVectorToMatrix(r2vec(:));
      t2vec = t2vec(:);
      t2vec(3) = 0;
      xyPatCtr = xyPatCtr(:);
      xyPatCtr(3) = 0;
      khat = [0;0;-1];
      
      [Rp,tp] = OrthoCam.computeDualPattern(R,t2vec,xyPatCtr,khat);
      r2vecnew = vision.internal.calibration.rodriguesMatrixToVector(Rp);
      t2vecnew = tp(1:2);      
    end
    function [r2vecs,t2vecs] = flipPatternSet(r2vecs,t2vecs,tfflip,varargin)
      nPat = size(r2vecs,2);
      szassert(r2vecs,[3 nPat]);
      szassert(t2vecs,[2 nPat]);
      assert(numel(tfflip)==nPat && islogical(tfflip));
      for iPat=1:nPat
        if tfflip(iPat)
          [r2vecs(:,iPat),t2vecs(:,iPat)] = ...
            OrthoCam.flipPattern(r2vecs(:,iPat),t2vecs(:,iPat),varargin{:});
        end
      end
    end
  end
  methods (Static) % stereo calib
    
    % Stereo Calibration
    % 
    % p = [mx1 my1 u01 v01 k1_1 k2_1 % intrinsics, cam1 (6 DOF)
    %      mx2 my2 u02 v02 k1_2 k2_2 % intrinsics, cam2 (6 DOF)
    %      r2vec1 t2vec1 % extrinsics, cam1 (5 DOF)
    %      r2vec2 t2vec2 % extrinsics, cam2 (5 DOF)
    %      rvecs1 ... rvecs_nPat tvecs1 ... tvecs_nPat  % extrinsics, cal pats (6*(nPat-1) DOF)
    %
    % Despite the name similarity, (r2vec1,t2vec1) and (rvecs1,tvecs1)
    % represent entirely different things. (r2vec1,t2vec1) (5 DOF) sets 
    % the extrinsic position of Camera1 relative to the World Sys.
    % (rvecs1,tvecs1) (6 DOF per calPat) sets the extrinsic position of
    % calPat i relative to the World Sys (calPat 1).
           
    function tblInts = summarizeIntrinsicsStro(p,nPat)
      [mx1,my1,u01,v01,k1_1,k2_1,...
     mx2,my2,u02,v02,k1_2,k2_2] = OrthoCam.unpackParamsStro(p,nPat);
      VARNAMES = {'mx' 'my' 'u0' 'v0' 'k1' 'k2'};
      t1 = table(mx1,my1,u01,v01,k1_1,k2_1,'VariableNames',VARNAMES);
      t2 = table(mx2,my2,u02,v02,k1_2,k2_2,'VariableNames',VARNAMES);
      tblInts = [t1;t2];
      tblInts.Properties.RowNames = {'cam1' 'cam2'};
    end
    function p = packParamsStro(...
        mx1,my1,u01,v01,k1_1,k2_1,...
        mx2,my2,u02,v02,k1_2,k2_2,...
        r2vec1,t2vec1,r2vec2,t2vec2,rvecs,tvecs)
      sclrassert(mx1);
      sclrassert(my1);
      sclrassert(u01);
      sclrassert(v01);
      sclrassert(k1_1);
      sclrassert(k2_1);
      sclrassert(mx2);
      sclrassert(my2);
      sclrassert(u02);
      sclrassert(v02);
      sclrassert(k1_2);
      sclrassert(k2_2);      
      szassert(r2vec1,[3 1]);
      szassert(t2vec1,[2 1]);
      szassert(r2vec2,[3 1]);
      szassert(t2vec2,[2 1]);
      nPatm1 = size(rvecs,1);
      szassert(rvecs,[nPatm1 3]);
      szassert(tvecs,[nPatm1 3]);
      
      p = [mx1;my1;u01;v01;k1_1;k2_1;mx2;my2;u02;v02;k1_2;k2_2;...
        r2vec1(:);t2vec1(:);r2vec2(:);t2vec2(:);rvecs(:);tvecs(:)];
      szassert(p,[22+6*nPatm1 1]);
    end
    function [...
        mx1,my1,u01,v01,k1_1,k2_1,...
        mx2,my2,u02,v02,k1_2,k2_2,...
        r2vec1,t2vec1,r2vec2,t2vec2,rvecs,tvecs] = unpackParamsStro(p,nPat) 
      
      nPatm1 = nPat-1;
      szassert(p,[22+6*nPatm1 1]);
      mx1 = p(1);
      my1 = p(2);
      u01 = p(3);
      v01 = p(4);
      k1_1 = p(5);
      k2_1 = p(6);
      mx2 = p(7);
      my2 = p(8);
      u02 = p(9);
      v02 = p(10);
      k1_2 = p(11);
      k2_2 = p(12);
      r2vec1 = p(13:15);
      t2vec1 = p(16:17);
      r2vec2 = p(18:20);
      t2vec2 = p(21:22);
      rvecs = reshape(p(23:23+3*nPatm1-1),[nPatm1 3]);
      tvecs = reshape(p(23+3*nPatm1:23+3*nPatm1+3*nPatm1-1),[nPatm1 3]);
    end
    function [d,dsum,uvcam1,uvcam2] = oFcnStro(p,nPat,patPtsXYZ,patImPts1,patImPts2)
      % Objective Fcn, stereo OrthoCam calib
      %
      % TODO: near-dup of OrthoCamCalPair.oFcnStro,
      % OrthoCamCalPair.computeRPerrStc
      %
      % patPtsXYZ: [3 x nPts] calibration world pts (x, y, z in calib pattern world frame)
      % patImPts1: [2 x nPts x nPat] x, y image pts for each cal pattern/pt in cam 1
      % patImPts2: [2 x nPts x nPat] " cam 2
      % 
      % d: [nPts*nPat*2 x 1] euclidean dist reproj err for each cal pt in
      %   each view. d is conceptually [nPts x nPat x 2] where 3rd dim is
      %   [cam1 cam2].
      % dsum: sum(d).

      nPts = size(patPtsXYZ,2);
      szassert(patPtsXYZ,[3 nPts]);
      szassert(patImPts1,[2 nPts nPat]);
      szassert(patImPts2,[2 nPts nPat]);

      [mx1,my1,u01,v01,k1_1,k2_1,...
        mx2,my2,u02,v02,k1_2,k2_2,...
        r2vec1,t2vec1,r2vec2,t2vec2,rvecs,tvecs] = OrthoCam.unpackParamsStro(p,nPat);
      
      % compute projected pts
      R2WorldToCam1 = vision.internal.calibration.rodriguesVectorToMatrix(r2vec1);
      t2WorldToCam1 = t2vec1;
      R2WorldToCam2 = vision.internal.calibration.rodriguesVectorToMatrix(r2vec2);
      t2WorldToCam2 = t2vec2;
      
      uvcam1 = nan(2,nPts,nPat);
      uvcam2 = nan(2,nPts,nPat);
      for iPat=1:nPat
        if iPat==1
          patPtsWorld = patPtsXYZ; % World 3D frame defined to be Pat1 3D frame
        else
          RPatIToWorld = vision.internal.calibration.rodriguesVectorToMatrix(rvecs(iPat-1,:)');
          tPatIToWorld = tvecs(iPat-1,:)';
          patPtsWorld = RPatIToWorld*patPtsXYZ + tPatIToWorld;
        end
        
        uvcam1(:,:,iPat) = OrthoCam.project(patPtsWorld,R2WorldToCam1,t2WorldToCam1,k1_1,k2_1,mx1,my1,u01,v01);
        uvcam2(:,:,iPat) = OrthoCam.project(patPtsWorld,R2WorldToCam2,t2WorldToCam2,k1_2,k2_2,mx2,my2,u02,v02);
      end
     
      d2cam1 = sum((uvcam1-patImPts1).^2,1); % [1 nPts nPat]
      d2cam2 = sum((uvcam2-patImPts2).^2,1); % [1 nPts nPat]

      d2 = cat(3,squeeze(d2cam1),squeeze(d2cam2));
      szassert(d2,[nPts nPat 2]);
      
      d2 = d2(:);
      d = sqrt(d2);      
      dsum = sum(d);
    end
    function [r2vec1,t2vec1,r2vec2,t2vec2,rvecs,tvecs] = ...
        estimateStroExtrinsics(r2vecsCalIms1,t2vecsCalIms1,r2vecsCalIms2,t2vecsCalIms2)
      % estimate Extrinsic components of parameters from intrinsic
      % calibration results.
      %
      % r2vecsCalIms1 [nPat x 3]: Rotation vecs bringing Pat i coords to Cam 1 sys
      % t2vecsCalIms1 [nPat x 2]: Translate2 vecs bringing Pat i coords to Cam 1 sys
      % r2vecsCalIms2 [nPat x 3]: etc
      % t2vecsCalIms2 [nPat x 2]: etc
      % 
      % r2vec1 [3x1],t2vec1 [2x1]: Extrinsic loc of Cam1 relative to World Sys 
      %   (defined to be coord sys of calpat 1)
      % r2vec2 [3x1],t2vec2 [2x1]: " Cam2
      % rvecs [nPat-1 x 3]: Rotation vecs of calpat i relative to World Sys
      % tvecs [nPat-1 x 3]: Translation "
      
      nPat = size(r2vecsCalIms1,1);
      szassert(r2vecsCalIms1,[nPat 3]);
      szassert(t2vecsCalIms1,[nPat 2]);
      szassert(r2vecsCalIms2,[nPat 3]);
      szassert(t2vecsCalIms2,[nPat 2]);
      
      r2vec1 = r2vecsCalIms1(1,:)';
      t2vec1 = t2vecsCalIms1(1,:)';
      r2vec2 = r2vecsCalIms2(1,:)';
      t2vec2 = t2vecsCalIms2(1,:)';
      
      nPatm1 = nPat-1;
      rvecs = nan(nPatm1,3);
      tvecs = nan(nPatm1,3);
      RPat1ToCam1 = vision.internal.calibration.rodriguesVectorToMatrix(r2vec1);
      RPat1ToCam2 = vision.internal.calibration.rodriguesVectorToMatrix(r2vec2);
      t2Pat1ToCam1 = t2vec1;
      t2Pat1ToCam2 = t2vec2;
      for iPat=2:nPat
        RPatIToCam1 = vision.internal.calibration.rodriguesVectorToMatrix(r2vecsCalIms1(iPat,:)');
        RPatIToCam2 = vision.internal.calibration.rodriguesVectorToMatrix(r2vecsCalIms2(iPat,:)');
        t2PatIToCam1 = t2vecsCalIms1(iPat,:)';
        t2PatIToCam2 = t2vecsCalIms2(iPat,:)';
        
        RPatIToPat1_fromcam1 = RPat1ToCam1\RPatIToCam1;
        tPatIToPat1_fromcam1 = RPat1ToCam1\[t2PatIToCam1-t2Pat1ToCam1;0];
        %RPatIToPat1_fromcam2 = RPat1ToCam2\RPatIToCam2;
        %tPatIToPat1_fromcam2 = RPat1ToCam2\[t2PatIToCam2-t2Pat1ToCam2;0];
        
        R = RPatIToPat1_fromcam1; % just take fromcam1 for now, don't try to average
        t = tPatIToPat1_fromcam1;
        r = vision.internal.calibration.rodriguesMatrixToVector(R);
        rvecs(iPat-1,:) = r(:)';
        tvecs(iPat-1,:) = t(:)';
      end
    end
    
    function [optCtr,n,ijkCam] = hlpExtrinsics(camInfo,r2vec,t2vec)
      if ~isempty(camInfo)
        optCtr = camInfo.optCtr;
        n = camInfo.n;
        ijkCam = camInfo.ijkCamWorld;
      else
        RWorldToCam = vision.internal.calibration.rodriguesVectorToMatrix(r2vec);
        t2WorldToCam = t2vec;
        [x0y0cam,n,~,~,ijkCam] = OrthoCam.opticalCenter(RWorldToCam,t2WorldToCam);
        optCtr = [x0y0cam(:);0];
      end
    end
    function hFig = viewExtrinsics(patPtsXYZ,rvecsFull,tvecsFull,...
        r2vec1,t2vec1,r2vec2,t2vec2,varargin)
      
      [dOptAx,cam1info,cam2info] = myparse(varargin,...
        'dOptAx',10,... % length of optical axis to plot (world coords)
        'cam1info',[],... % if nonempty, supplemental extrinsics info for cam1. Otherwise, recomputed
        'cam2info',[]...
        );                    
      
      nPts = size(patPtsXYZ,2);
      szassert(patPtsXYZ,[3 nPts]);
      nPat = size(rvecsFull,1);
      szassert(rvecsFull,[nPat 3]);
      szassert(tvecsFull,[nPat 3]);        
      szassert(r2vec1,[3 1]);
      szassert(t2vec1,[2 1]);
      szassert(r2vec2,[3 1]);
      szassert(t2vec2,[2 1]);
      
      hFig = figure('Name','OrthoCam: Calibration Extrinsics');
      ax = axes;
      hold(ax,'on');
      
      % plot the pats
      patPtsMins = min(patPtsXYZ,[],2);
      patPtsMaxs = max(patPtsXYZ,[],2);
      patX0 = patPtsMins(1);
      patX1 = patPtsMaxs(1);
      patY0 = patPtsMins(2);
      patY1 = patPtsMaxs(2);
      patZ0 = patPtsMins(3);
      patZ1 = patPtsMaxs(3);
      assert(patZ0==patZ1);
      patPtsCorners = [ ...
        patX0 patX1 patX1 patX0; ...
        patY0 patY0 patY1 patY1; ...
        patZ0 patZ0 patZ0 patZ0; ];
      clrs = jet(nPat);
      for iPat=1:nPat
        RPatI2World = vision.internal.calibration.rodriguesVectorToMatrix(rvecsFull(iPat,:)');
        tPatI2World = tvecsFull(iPat,:)';
        patPtsCornersWorld = RPatI2World*patPtsCorners + tPatI2World;
        fill3(patPtsCornersWorld(1,:),patPtsCornersWorld(2,:),...
              patPtsCornersWorld(3,:),clrs(iPat,:),'FaceAlpha',0.5);
            
        % plot origin + yaxis in bold
        orig = patPtsCornersWorld(:,1);
        plot3(orig(1),orig(2),orig(3),'.','markersize',26,'color',[0 0 0]);
        plot3(patPtsCornersWorld(1,[1 4]),patPtsCornersWorld(2,[1 4]),...
          patPtsCornersWorld(3,[1 4]),'-','linewidth',3,'color',[0 0 0]);
      end
            
      % plot the optical ctrs/axes
      [optCtr1,n1,ijkCam1] = OrthoCam.hlpExtrinsics(cam1info,r2vec1,t2vec1);
      [optCtr2,n2,ijkCam2] = OrthoCam.hlpExtrinsics(cam2info,r2vec2,t2vec2);
      [~,~,az1,el1] = OrthoCam.azEl(n1);
      [~,~,az2,el2] = OrthoCam.azEl(n2);
     
      plot3(optCtr1(1),optCtr1(2),optCtr1(3),'x','markersize',10,'linewidth',3,'color',[0 0 0]);
      plot3(optCtr2(1),optCtr2(2),optCtr2(3),'x','markersize',10,'linewidth',3,'color',[0 0 0]);
      
      DX = 1;
      optAx1 = [optCtr1 optCtr1+n1*dOptAx];
      optAx2 = [optCtr2 optCtr2+n2*dOptAx];
      optAx1Plus = optCtr1+n1*(dOptAx+DX);
      optAx2Plus = optCtr2+n2*(dOptAx+DX);
      optAx1Mid = optCtr1+n1*dOptAx/2; 
      optAx2Mid = optCtr2+n2*dOptAx/2;
      
      plot3(optAx1(1,:),optAx1(2,:),optAx1(3,:),'--','linewidth',2,'color',[0 0 0]);
      plot3(optAx2(1,:),optAx2(2,:),optAx2(3,:),'--','linewidth',2,'color',[0 0 0]);      
      BROWN = [139 69 19]/255;
      text(optAx1Plus(1),optAx1Plus(2),optAx1Plus(3),'C1',...
        'fontweight','bold','fontsize',12,'color',BROWN);
      text(optAx2Plus(1),optAx2Plus(2),optAx2Plus(3),'C2',...
        'fontweight','bold','fontsize',12,'color',BROWN);
     
      
      % Check: x0y0cam1+pxcam1 should project to [1 0] etc
      % draw pxcam1,... at dOptAx/2
      optAxMid1CamXax = [optAx1Mid optAx1Mid+ijkCam1(:,1)];
      optAxMid1CamYax = [optAx1Mid optAx1Mid+ijkCam1(:,2)];
      optAxMid2CamXax = [optAx2Mid optAx2Mid+ijkCam2(:,1)];
      optAxMid2CamYax = [optAx2Mid optAx2Mid+ijkCam2(:,2)];
      optAxMid1CamXaxPlus = optAx1Mid+1.5*ijkCam1(:,1);
      optAxMid1CamYaxPlus = optAx1Mid+1.5*ijkCam1(:,2);
      optAxMid2CamXaxPlus = optAx2Mid+1.5*ijkCam2(:,1);
      optAxMid2CamYaxPlus = optAx2Mid+1.5*ijkCam2(:,2);
      
      plot3(optAxMid1CamXax(1,:),optAxMid1CamXax(2,:),optAxMid1CamXax(3,:),'b-','linewidth',2);
      plot3(optAxMid1CamYax(1,:),optAxMid1CamYax(2,:),optAxMid1CamYax(3,:),'b-','linewidth',2);
      plot3(optAxMid2CamXax(1,:),optAxMid2CamXax(2,:),optAxMid2CamXax(3,:),'b-','linewidth',2);
      plot3(optAxMid2CamYax(1,:),optAxMid2CamYax(2,:),optAxMid2CamYax(3,:),'b-','linewidth',2);
      text(optAxMid1CamXaxPlus(1),optAxMid1CamXaxPlus(2),optAxMid1CamXaxPlus(3),...
        'x','fontweight','bold','fontsize',9,'color',[0 0 1]);
      text(optAxMid1CamYaxPlus(1),optAxMid1CamYaxPlus(2),optAxMid1CamYaxPlus(3),...
        'y','fontweight','bold','fontsize',9,'color',[0 0 1]);
      text(optAxMid2CamXaxPlus(1),optAxMid2CamXaxPlus(2),optAxMid2CamXaxPlus(3),...
        'x','fontweight','bold','fontsize',9,'color',[0 0 1]);
      text(optAxMid2CamYaxPlus(1),optAxMid2CamYaxPlus(2),optAxMid2CamYaxPlus(3),...
        'y','fontweight','bold','fontsize',9,'color',[0 0 1]);
      
      angleInDeg = acos(dot(n1,n2))/pi*180;
      grid on;
      tstr = sprintf('%d pats. camAngle=%.1f. cam1 (az,el)=(%.2f,%.2f). cam2 (%.2f,%.2f)\n',...
        nPat,angleInDeg,az1/pi*180,el1/pi*180,az2/pi*180,el2/pi*180);
      title(ax,tstr,'fontweight','bold','interpreter','tex');
      xlabel(ax,'x (mm)','fontweight','bold');
      ylabel(ax,'y (mm)','fontweight','bold');
      zlabel(ax,'z (mm)','fontweight','bold');  
      axis(ax,'square');      
      axis(ax,'equal');
      view(0,0);
    end
    function opts = defaultoptsStro()
      opts = optimset;
      opts.Display = 'iter';
      opts.TolFun = 1e-6;
      opts.TolX = 1e-6;
      opts.MaxFunEvals = 1e4; 
      opts.MaxIter = 1e4;
    end
    function [pOpt,oFcn,dsum0,dsum1] = calibrateStro(nPat,worldPoints,imPtsUV1,imPtsUV2,p0,varargin)
      opts = myparse(varargin,...
        'opts',[]);
      if isempty(opts)
        opts = OrthoCam.defaultoptsStro();
      end
      
      nPts = size(worldPoints,1);
      szassert(worldPoints,[nPts 2]);
      patPtsXYZ = [worldPoints zeros(nPts,1)]';           
      szassert(imPtsUV1,[nPts 2 nPat]);
      szassert(imPtsUV2,[nPts 2 nPat]);
      patImPts1 = permute(imPtsUV1,[2 1 3]);
      patImPts2 = permute(imPtsUV2,[2 1 3]);
            
      oFcn = @(p)OrthoCam.oFcnStro(p,nPat,patPtsXYZ,patImPts1,patImPts2);
      
      [~,dsum0] = oFcn(p0);
      fprintf('Starting residual: %.4g\n',dsum0);
      
      [pOpt,resnorm,res] = lsqnonlin(oFcn,p0,[],[],opts);

      [~,dsum1] = oFcn(pOpt);
      fprintf('Ending residual: %.4g\n',dsum1);
    end
  end
  methods (Static) % misc 
    function hBar = vizRPerr(ax,dRP)
      [npts,npat] = size(dRP); %#ok<ASGLU>
      [~,edges] = histcounts(dRP(:));
      ctrs = (edges(1:end-1)+edges(2:end))/2;
      nbin = numel(ctrs);
      
      counts = nan(nbin,npat);
      for ipat=1:npat
        counts(:,ipat) = histcounts(dRP(:,ipat),edges);
      end

      axes(ax);
      hBar = bar(ctrs,counts,'stacked');
      grid on;
      xlabel('RP err (px)','fontweight','bold');
    end
  end
end