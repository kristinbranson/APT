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
      obj.calTS = now;
    end
    
    function hFig = viewExtrinsics(obj)
      hFig = OrthoCam.viewExtrinsics(obj.calWorldPoints,...
        obj.rvecs,obj.tvecs,obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2);
    end
    
    function xformWorldSys(obj,R)
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
    end    
  end
  
end

function r2vecnew = lclHelpR(r2vec,R)
Rworld2cam = vision.internal.calibration.rodriguesVectorToMatrix(r2vec);
Rworld2camNew = Rworld2cam*R';
r2vecnew = vision.internal.calibration.rodriguesMatrixToVector(Rworld2camNew);
end