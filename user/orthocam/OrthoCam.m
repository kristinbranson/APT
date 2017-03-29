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
  % Stereo calibration may be even further complicated, as this requires 
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
      r2 = sum(W.^2,1); % [1xnpts]
      distort = 1 + k1*r2 + k2*r2.^2; % [1xnpts]
      
      Wd = W.*distort;
      uv = [ mx*Wd(1,:) + u0 ; ...
             my*Wd(2,:) + v0 ]; % [2xnpts]
    end
    function [x0y0,n] = opticalCenter(R2cam,t2cam)
      % Find the "optical center" for a cam; the WorldPoint (x0,y0,0) where
      % the cam's optical axis intersects the World plane z=0
      %
      % R2cam, t2cam: Rot, translation to go from World->cam
      %
      % x0y0: [2x1] [x0;y0] optical center 
      % n: [3x1] [nx;ny;nz] normal vec pointing from optical center to cam
      %   (at infinity) along optical axis. Cam is assumed to be at
      %   negative z.
      
      szassert(R2cam,[3 3]);
      szassert(t2cam,[2 1]);
      
      R12 = R2cam(1:2,1:2);
      x0y0 = -R12\t2cam;
      
      n = cross(R2cam(1,:),R2cam(2,:));
      if n(3)>0
        n = -n;
      end
      n = n/sqrt(sum(n.^2));
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
    function pOpt = calibrate1cam(nCalIm,worldPoints,imPtsUV,p0,varargin)
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
    function [d,dsum] = oFcnStro(p,nPat,patPtsXYZ,patImPts1,patImPts2)
      % Objective Fcn, stereo OrthoCam calib
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
        estimateExtrinsics(r2vecsCalIms1,t2vecsCalIms1,r2vecsCalIms2,t2vecsCalIms2)
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
        RPatIToPat1_fromcam2 = RPat1ToCam2\RPatIToCam2;
        tPatIToPat1_fromcam2 = RPat1ToCam2\[t2PatIToCam2-t2Pat1ToCam2;0];
        
        R = RPatIToPat1_fromcam1; % just take fromcam1 for now, don't try to average
        t = tPatIToPat1_fromcam1;
        r = vision.internal.calibration.rodriguesMatrixToVector(R);
        rvecs(iPat-1,:) = r(:)';
        tvecs(iPat-1,:) = t(:)';
      end
    end

  end
end