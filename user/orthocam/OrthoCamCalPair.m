classdef OrthoCamCalPair < handle
  % A pair of calibrated Orthocams, with calibration info
  
  properties
    tblInt % table of intrinsics
    
    % extrinsics
    r2vec1 % [3x1] 
    t2vec1 % [2x1]
    r2vec2 % [3x1]
    t2vec2 % [2x1]
    rvecs % [nPatx3] IMPORTANT: Note this has a row for EVERY PATTERN not patterns 2..nPat.
    tvecs % [nPatx3]
    
    % extrinsics2 -- these are derived from extrinsics, but for convenience
    % we compute them initially and store them b/c after a transformation
    % it may be figure out eg the sign of n1/n2, there may be numerical
    % issues due to the cameras being at precisely 90deg, etc.
    optCtr1 % [3x1] World coords where cam1 optical axis intersected original WorldSys (cal pattern) at z=0 
    optCtr2 % [3x1] " cam2
    n1 % [3x1] unit normal vec pointing from optCtr1 to cam1 at infinity; optical axis
    n2 % [3x1] " cam2
    ijkCamWorld1 % [3x3] columns are "CamWorldCoords" i/j/k unit vecs in WorldSys for cam1
    ijkCamWorld2 % "

    calNumPatterns % number of calibration patterns used
    calNumPoints % number of points in calibration pattern
    calWorldPoints % [3xnPts] (x,y,z) of calibration points in PatternWorldSys
    calImPoints % [2xnPtsxnPatx2] (x,y) for each point, pattern, camera
    calPatternFPNs % [nPatx2] cellstr of calibration images
    calTS % timestamp of most recent calibration
  end

  methods
    
    function obj = OrthoCamCalPair(p,nPat,nPts,worldPts,imPts,calPatFPNs)
      obj.tblInt = OrthoCam.summarizeIntrinsicsStro(p,nPat);
      [~,~,~,~,~,~,~,~,~,~,~,~,...
       obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,rvecs2thrun,tvecs2thrun] = ...
         OrthoCam.unpackParamsStro(p,nPat);
      obj.rvecs = [0 0 0;rvecs2thrun];
      obj.tvecs = [0 0 0;tvecs2thrun];
      
      obj.calNumPatterns = nPat;
      obj.calNumPoints = nPts;
      szassert(worldPts,[3 nPts]);
      szassert(imPts,[2 nPts nPat 2]);
      szassert(calPatFPNs,[nPat 2]);
      obj.calWorldPoints = worldPts;
      obj.calImPoints = imPts;
      obj.calPatternFPNs = calPatFPNs;
      [obj.optCtr1,obj.n1,~,~,obj.ijkCamWorld1] = ...
        OrthoCam.opticalCenter(vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec1),obj.t2vec1);
      [obj.optCtr2,obj.n2,~,~,obj.ijkCamWorld2] = ...
        OrthoCam.opticalCenter(vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2),obj.t2vec2);
      obj.optCtr1(end+1) = 0;
      obj.optCtr2(end+1) = 0;
      obj.calTS = now;
    end
    
    function hFig = viewExtrinsics(obj)
      hFig = OrthoCam.viewExtrinsics(obj.calWorldPoints,...
        obj.rvecs,obj.tvecs,obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,...
        'cam1info',struct('optCtr',obj.optCtr1,'n',obj.n1,'ijkCamWorld',obj.ijkCamWorld1),...
        'cam2info',struct('optCtr',obj.optCtr2,'n',obj.n2,'ijkCamWorld',obj.ijkCamWorld2));
    end
    
    function xformWorldSys(obj,R)
      % Transform WorldCoords by rotation (origin unchanged)
      % R: [3x3]. x_newWorldSys = R*x_oldWorldSys
            
      szassert(R,[3 3]);
      for iPat=1:obj.calNumPatterns
        R0 = vision.internal.calibration.rodriguesVectorToMatrix(obj.rvecs(iPat,:)');
        R1 = R*R0;
        obj.rvecs(iPat,:) = vision.internal.calibration.rodriguesMatrixToVector(R1);
        
        t0 = obj.tvecs(iPat,:)';
        t1 = R*t0;
        obj.tvecs(iPat,:) = t1;
      end
      
      obj.r2vec1 = lclHelpR(obj.r2vec1,R);
      obj.r2vec2 = lclHelpR(obj.r2vec2,R);
      
      obj.optCtr1 = R*obj.optCtr1;
      obj.optCtr2 = R*obj.optCtr2;
      obj.n1 = R*obj.n1;
      obj.n2 = R*obj.n2;
      obj.ijkCamWorld1 = R*obj.ijkCamWorld1;
      obj.ijkCamWorld2 = R*obj.ijkCamWorld2;
    end
    
    % Coordsys Notes
    % X=[x;y;z] 3D worldsys coords
    % pq=[p;q] Normalized coords; undistorted, unmagnified x-y deviations
    % from optical axis along camera x-y axes
    % uv=[u;v] Image coords; (col,row) pixel coords on image
    
    function [X,d,uvreL,uvreR] = stereoTriangulate(obj,uvL,uvR)
      % Stereo triangulation
      %
      % uvL, uvR: [2xN] x-y image coords
      % 
      % X: [3xN]: reconstructed world coords
      % d: [1xN]: error/discrepancy in closest approach. d=0 indicates
      % apparently "perfect" reconstruction where epipolar rays meet
      % uvreL, uvreR: [2xN]: reprojected x-y image coords
      
      pqL = obj.projected2normalized(uvL,1);
      pqR = obj.projected2normalized(uvR,2);
      
      n = size(pqL,2);
      szassert(pqL,[2 n]);
      szassert(pqR,[2 n]);
      
      OL = obj.optCtr1 + pqL(1,:).*obj.ijkCamWorld1(:,1) ...
                       + pqL(2,:).*obj.ijkCamWorld1(:,2);
      OR = obj.optCtr2 + pqR(1,:).*obj.ijkCamWorld2(:,1) ...
                       + pqR(2,:).*obj.ijkCamWorld2(:,2);
      szassert(OL,[3 n]);
      szassert(OR,[3 n]);
      X = nan(3,n);
      d = nan(1,n);      
      for i=1:n
        [P,Q,d(i)] = CalibratedRig.stereoTriangulateRays(OL(:,i),obj.n1,...
          OR(:,i),obj.n2);
        X(:,i) = (P+Q)/2;
      end
      
      R2L = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec1);
      t2L = obj.t2vec1;
      R2R = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2);
      t2R = obj.t2vec2;
      int1 = obj.tblInt(1,:);
      int2 = obj.tblInt(2,:);     
      uvreL = OrthoCam.project(X,R2L,t2L,...
        int1.k1,int1.k2,int1.mx,int1.my,int1.u0,int1.v0);
      uvreR = OrthoCam.project(X,R2R,t2R,...
        int2.k1,int2.k2,int2.mx,int2.my,int2.u0,int2.v0);
    end    
    
    function pq = projected2normalized(obj,uv,icam)
      % uv: [2xN]
      % icam: camera index
      
      assert(isnumeric(icam) && (icam==1 || icam==2));
      ints = obj.tblInt(icam,:);
      pq = OrthoCam.projected2normalized(ints.mx,ints.my,ints.u0,ints.v0,...
        ints.k1,ints.k2,uv);
    end
    
  end
  
end

function r2vecnew = lclHelpR(r2vec,R)
Rworld2cam = vision.internal.calibration.rodriguesVectorToMatrix(r2vec);
Rworld2camNew = Rworld2cam*R';
r2vecnew = vision.internal.calibration.rodriguesMatrixToVector(Rworld2camNew);
end