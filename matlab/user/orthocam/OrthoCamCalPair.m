classdef OrthoCamCalPair < CalRig
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
    ijkCamWorld1 % [3x3] columns are "CamWorldCoords" i/j/k unit vecs in WorldSys for cam1; k is negative optical axis
    ijkCamWorld2 % "

    calNumPatterns % number of calibration patterns used
    calNumPoints % number of points in calibration pattern
    calWorldPoints % [3xnPts] (x,y,z) of calibration points in PatternWorldSys
    calImPoints % [2xnPtsxnPatx2] (x,y) for each point, pattern, camera
    calPatternFPNs % [nPatx2] cellstr of calibration images
    calTS % timestamp of most recent calibration
  end
  
  properties % CalRig
    nviews = 2;
    viewNames = {'side' 'front'};
%     viewSizes = [1024 1024;1024 1024];
    
    estimatedEpipolarZCoords; % [nviewx1] cell. estimatedEpipolarZCoords{iView} gives estimated [zmin zmax] to use when plotting EPlines originating in iView
  end

  methods
    
    function obj = OrthoCamCalPair(varargin)
      % obj = OrthoCamCalPair(p,nPat,nPts,worldPts,imPts,calPatFPNs)
      % Default constructor
      % p has been packed with (nPat-1) rvecs/tvecs; the remaining/first
      % rvec/tvec is assumed to be all zeros b/c the first calpat is used
      % as the WorldSys.
      %
      % obj = OrthoCamCalPair('alt',p,nPat,nPts,worldPts,imPts,calPatFPNs)
      % Alternative constructor
      % p has been packed with nPat rvecs/tvecs, one for every pattern. 
      % Meanwhile the r2vec1/t2vec1/r2vec2/t2vec2 are relative to some
      % arbitrary WorldSys.
      
      if isstruct(varargin{1}),
        s = varargin{1};
        fns = fieldnames(s);
        for i = 1:numel(fns),
          fn = fns{i};
          if strcmp(fn,'class_name'),
            continue;
          end
          obj.(fn) = s.(fn);
        end
      else
        tfAlt = strcmp(varargin{1},'alt');
        if tfAlt
          varargin = varargin(2:end);
        end
        p = varargin{1};
        nPat = varargin{2};
        nPts = varargin{3};
        worldPts = varargin{4};
        imPts = varargin{5};
        calPatFPNs = varargin{6};

        if ~tfAlt
          obj.tblInt = OrthoCam.summarizeIntrinsicsStro(p,nPat);
          [~,~,~,~,~,~,~,~,~,~,~,~,...
            obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,rvecs2thrun,tvecs2thrun] = ...
            OrthoCam.unpackParamsStro(p,nPat);
          obj.rvecs = [0 0 0;rvecs2thrun];
          obj.tvecs = [0 0 0;tvecs2thrun];
        else
          assert(false,'TODO unsupported.');
          %         obj.tblInt = OrthoCam.summarizeIntrinsicsStro(p,nPat+1);
          %         [~,~,~,~,~,~,~,~,~,~,~,~,...
          %           obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,obj.rvecs,obj.tvecs] = ...
          %           OrthoCam.unpackParamsStro(p,nPat+1);
        end

        obj.calNumPatterns = nPat;
        obj.calNumPoints = nPts;
        szassert(worldPts,[3 nPts]);
        szassert(imPts,[2 nPts nPat 2]);
        szassert(calPatFPNs,[nPat 2]);
        obj.calWorldPoints = worldPts;
        obj.calImPoints = imPts;
        obj.calPatternFPNs = calPatFPNs;
        [obj.optCtr1,~,~,~,obj.ijkCamWorld1] = ...
          OrthoCam.opticalCenter(vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec1),obj.t2vec1);
        [obj.optCtr2,~,~,~,obj.ijkCamWorld2] = ...
          OrthoCam.opticalCenter(vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2),obj.t2vec2);
        obj.optCtr1(end+1) = 0;
        obj.optCtr2(end+1) = 0;
        obj.calTS = now;
      end
    end
    
    function hFig = viewExtrinsics(obj,varargin)
      firstPatOnly = myparse(varargin,...
        'firstPatOnly',false);
      
      if firstPatOnly
        iRTVecsPlot = 1;
      else
        iRTVecsPlot = 1:size(obj.rvecs,1);
      end
      hFig = OrthoCam.viewExtrinsics(obj.calWorldPoints,...
        obj.rvecs(iRTVecsPlot,:),obj.tvecs(iRTVecsPlot,:),...
        obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,...
        'cam1info',struct('optCtr',obj.optCtr1,'n',-obj.ijkCamWorld1(:,3),'ijkCamWorld',obj.ijkCamWorld1),...
        'cam2info',struct('optCtr',obj.optCtr2,'n',-obj.ijkCamWorld2(:,3),'ijkCamWorld',obj.ijkCamWorld2));
    end
   
  end
  methods (Hidden)
    function hFig = viewRPerrHlp(obj,dRP,titleStr,varargin)
      % Helper for RP viz that does plotting
      %
      % dRP: [npts nPat 2] where dRP(:,:,1) is cam1 and dRP(:,:,2) is cam2
      
      showLegPatFilenames = myparse(varargin,...
        'showLegPatFilenames',true);
      
      npts = obj.calNumPoints;
      nPat = obj.calNumPatterns;
      szassert(dRP,[npts nPat 2]);

      hFig = figure('Name',titleStr);
      dRP1 = dRP(:,:,1);
      dRP2 = dRP(:,:,2);
      mu1 = mean(dRP1(:));
      mu2 = mean(dRP2(:));
      ax(1) = subplot(1,2,1);
      hBar{1} = OrthoCam.vizRPerr(ax(1),dRP1);
      title(sprintf('Stereo calib, cam1. %dpats, %dpts. mean RPerr=%.3f px',nPat,npts,mu1),...
        'fontweight','bold');
      ylabel('count','fontweight','bold');
      ax(2) = subplot(1,2,2);
      hBar{2} = OrthoCam.vizRPerr(ax(2),dRP2);
      title(sprintf('Stereo calib, cam2. meanRP err=%.3f px',mu2),'fontweight','bold');
      
      linkaxes(ax,'x');
      if showLegPatFilenames
        fpns = obj.calPatternFPNs;
        fpns = cellfun(@basename,fpns,'uni',0);
        %legend(ax(1),hBar{1},fpns(:,1),'interpreter','none');
        legend(ax(2),hBar{2},fpns(:,2),'interpreter','none');
      end
    end
  end
  methods
    function hFig = viewRPerr(obj)
      % View histograms of RP error using calibrated/estimated extrinsic
      % positions, known structure of calpats etc.      
      [~,dRP] = obj.computeRPerr();
      dRP = reshape(dRP,[obj.calNumPoints obj.calNumPatterns 2]);
      TITLE = 'RP error using extrinsic positions';
      hFig = obj.viewRPerrHlp(dRP,TITLE);
    end
    
    function hFig = viewRPerrStroTri(obj)
      % View histograms of RP error using only stereo triangulation of
      % calibration image points.      
      dRP = obj.computeRPerrStroTriCalPts();
      TITLE = 'RP error, stereo-triangulation only';
      hFig = obj.viewRPerrHlp(dRP,TITLE);
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
      obj.ijkCamWorld1 = R*obj.ijkCamWorld1;
      obj.ijkCamWorld2 = R*obj.ijkCamWorld2;
    end
    
    % Coordsys Notes
    % X=[x;y;z] 3D worldsys coords
    % pq=[p;q] Normalized coords; undistorted, unmagnified x-y deviations
    % from optical axis along camera x-y axes
    % uv=[u;v] Image coords; (col,row) pixel coords on image
    
    function [X,uvrp,rpe] = triangulate(obj,uv)
      % CalRig impl
      
      assert(size(uv,3)==obj.nviews);
      
      uvL = uv(:,:,1);
      uvR = uv(:,:,2);
      [X,d,uvreL,uvreR,rpeL,rpeR] = obj.stereoTriangulate(uvL,uvR);
      uvrp = cat(3,uvreL,uvreR);
      rpe = cat(2,rpeL,rpeR);
    end
    
    function [X,d,uvreL,uvreR,rperrL,rperrR] = stereoTriangulate(obj,uvL,uvR)
      % [X,d,uvreL,uvreR] = stereoTriangulate(obj,uvL,uvR)
      % Stereo triangulation
      %
      % uvL, uvR: [2xN] x-y image coords
      % 
      % X: [3xN]: reconstructed world coords
      % d: [1xN]: error/discrepancy in closest approach. d=0 indicates
      %   apparently "perfect" reconstruction where epipolar rays meet
      % uvreL, uvreR: [2xN]: reprojected x-y image coords
      % rperrL, rperrR: [N], L2 error between reprojected and input coords
      
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
      n1 = -obj.ijkCamWorld1(:,3);
      n2 = -obj.ijkCamWorld2(:,3);
      for i=1:n
        [P,Q,d(i)] = CalibratedRig.stereoTriangulateRays(OL(:,i),n1,OR(:,i),n2);
        X(:,i) = (P+Q)/2;
      end
      
      uvreL = obj.project(X,1);
      uvreR = obj.project(X,2);
      
      rperrL = sqrt(sum((uvreL-uvL).^2,1));
      rperrR = sqrt(sum((uvreR-uvR).^2,1));
    end
    
    function uv = project(obj,X,icam)
      % uv = project(obj,X,icam)
      % Project 3D world coords to camera image
      % 
      % X: [3xn] world coords
      % icam: camera index, 1/2 for left/right resp
      %
      % uv: [2xn] x- and y-pixel coords (NOT row/col) in camera icam
      
      assert(size(X,1)==3);
      assert(icam==1 || icam==2);
      
      switch icam
        case 1
          if verLessThan('matlab','9.13')  %#ok<VERLESSMATLAB>
            % For Matlab < 2022b
            R2 = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec1);
          else
            % For Matlab >= 2022b
            R2 = rotvec2mat3d(obj.r2vec1);
          end            
          t2 = obj.t2vec1;
        case 2
          if verLessThan('matlab','9.13')  %#ok<VERLESSMATLAB>
            % For Matlab < 2022b
            R2 = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2);
          else
            % For Matlab >= 2022b
            R2 = rotvec2mat3d(obj.r2vec1);
          end            
          t2 = obj.t2vec2;
      end
      int = obj.tblInt(icam,:);
      uv = OrthoCam.project(X,R2,t2,int.k1,int.k2,int.mx,int.my,int.u0,int.v0);
    end
    
    function pq = projected2normalized(obj,uv,icam)
      % uv: [2xn]
      % icam: camera index
      %
      % pq: [2xn]
      
      assert(isnumeric(icam) && (icam==1 || icam==2));
      ints = obj.tblInt(icam,:);
      pq = OrthoCam.projected2normalized(ints.mx,ints.my,ints.u0,ints.v0,...
        ints.k1,ints.k2,uv);
    end
    
    function [dmu,d,uvcam] = computeRPerr(obj)
      [dmu,d,uvcam] = OrthoCamCalPair.computeRPerrStc(obj.r2vec1,obj.t2vec1,...
        obj.r2vec2,obj.t2vec2,obj.rvecs,obj.tvecs,...
        obj.tblInt(1,:),obj.tblInt(2,:),obj.calWorldPoints,obj.calImPoints);        
    end
    
    function d = computeRPerrStroTriCalPts(obj)
      % Compute RP err, using calpat image points only: stereo triangulate 
      % and reproject.
      %
      % d: [nPts nPat 2] Eucld RP distance for iPt,iPat,cam

      nPat = obj.calNumPatterns;
      nPtsPat = obj.calNumPoints;      
      uv1 = obj.calImPoints(:,:,:,1);
      uv2 = obj.calImPoints(:,:,:,2);
      uv1 = reshape(uv1,[2 nPtsPat*nPat]); % dim2 raster order: ... allptsPat1..., allptsPat2 ... , ....
      uv2 = reshape(uv2,[2 nPtsPat*nPat]);
      [~,d] = obj.computeRPerrStroTriGeneral(uv1,uv2);
      % d is [nPtsPat*nPat x 2]
      d = reshape(d,[nPtsPat nPat 2]);
    end
    
    function [dmu,d,uvre1,uvre2] = computeRPerrStroTriGeneral(obj,uv1,uv2)
      n = size(uv1,2);
      szassert(uv1,[2 n]);
      szassert(uv2,[2 n]);
      
      [~,~,uvre1,uvre2] = obj.stereoTriangulate(uv1,uv2);
      d2_1 = sum((uvre1-uv1).^2,1); % [1 n]
      d2_2 = sum((uvre2-uv2).^2,1); 
      d2 = [d2_1;d2_2]; % [2 n]
      d = sqrt(d2)';
      szassert(d,[n 2]);
      
      dmu = mean(d);      
    end
    
    function viewCompare(objs,varargin)
      % View/Compare multiple OrthoCamCalPairs
      %
      % objs: [ncal] vector of OrthoCamCalPairs
      
      [viewLimX,viewLimY] = myparse(varargin,...
        'viewLimX',[1 768],...
        'viewLimY',[1 512]);
      
      hFig = figure;
      axs = createsubplots(1,2,.1);
      title(axs(1),'Cam1','fontsize',14,'fontweight','bold');
      title(axs(2),'Cam2','fontsize',14,'fontweight','bold');
      arrayfun(@(x)axis(x,'equal'),axs);
      arrayfun(@(x)axis(x,[viewLimX viewLimY]),axs);
      arrayfun(@(x)grid(x,'on'),axs);
      arrayfun(@(x)hold(x,'on'),axs);
      [axs.Color] = deal([0 0 0]);
      [axs.GridColor] = deal([1 1 1]);
      [axs.FontSize] = deal(12);
      
      ncal = numel(objs);
      hLine = gobjects(2,ncal);
      % hLine(iView,iCal) is EPline in view iView calib iCal
      colors = lines(ncal);
      for ical=1:ncal
        hLine(:,ical) = [...
          plot(axs(1),nan,nan,'color',colors(ical,:),'linewidth',2,'displayname',['cal' num2str(ical)]);
          plot(axs(2),nan,nan,'color',colors(ical,:),'linewidth',2,'displayname',['cal' num2str(ical)])];
      end
      hLeg = legend(axs(2),'show');
      set(hLeg,'color',[0.15 0.15 0.15],'textcolor',[1 1 1]);
      hPt1 = impoint(axs(1),100,200);
      hPt2 = impoint(axs(2),100,200);
      addNewPositionCallback(hPt1,@(xy) nstUpdateEP(xy,1,2) );
      addNewPositionCallback(hPt2,@(xy) nstUpdateEP(xy,2,1) );
      
      function nstUpdateEP(xy,iViewPt,iViewEP)
        for icalnst=1:ncal
          [xEPL,yEPL] = objs(icalnst).computeEpiPolarLine(iViewPt,xy,iViewEP); % xxx out of date api

          tfIB = viewLimX(1)<=xEPL & xEPL<=viewLimX(2) & ...
                 viewLimY(1)<=yEPL & yEPL<=viewLimY(2);
          xEPL = xEPL(tfIB);
          yEPL = yEPL(tfIB);
          set(hLine(iViewEP,icalnst),'XData',xEPL,'YData',yEPL);
        end
      end
    end
    
    function [optCtr,ijkCam] = getOptCtrCamWorldView(obj,iView)
      switch iView
        case 1
          optCtr = obj.optCtr1;
          ijkCam = obj.ijkCamWorld1;
        case 2
          optCtr = obj.optCtr2;
          ijkCam = obj.ijkCamWorld2;
      end
    end
    
  end
  
  methods (Static)
    
    function [d,dsum] = oFcnStro(p,nPat,patPtsXYZ,patImPts)
      % TODO: near-dup of OrthoCam.oFcnStro. Watch out for nPat+1!!

      nPts = size(patPtsXYZ,2);
      szassert(patPtsXYZ,[3 nPts]);
      szassert(patImPts,[2 nPts nPat 2]);
      
      [int1.mx,int1.my,int1.u0,int1.v0,int1.k1,int1.k2,...
       int2.mx,int2.my,int2.u0,int2.v0,int2.k1,int2.k2,...
       r2vec1,t2vec1,r2vec2,t2vec2,rvecs,tvecs] = ...
        OrthoCam.unpackParamsStro(p,nPat+1); % quirk of unpackParamsStro
      [~,d] = OrthoCamCalPair.computeRPerrStc(r2vec1,t2vec1,r2vec2,t2vec2,...
        rvecs,tvecs,int1,int2,patPtsXYZ,patImPts);
      d = d(:);
      dsum = sum(d);
    end
    
    function [dmu,d,uvcam] = computeRPerrStc(r2vec1,t2vec1,r2vec2,t2vec2,...
        rvecs,tvecs,int1,int2,patPtsXYZ,patImPts)
      % Compute RP err, using known pattern points and optimized/estimated
      % extrinsics for each pattern
      %
      % dmu: [2] mean of d for cam1, cam2
      % d: [nPts nPat 2] Eucld RP distance for iPt,iPat,cam
      % uvcam: [2 nPts nPat 2]. (x,y) x iPt x iPat x (cam1,cam2)
      
      szassert(r2vec1,[3 1]);
      szassert(t2vec1,[2 1]);
      szassert(r2vec2,[3 1]);
      szassert(t2vec2,[2 1]);
      nPat = size(rvecs,1);
      szassert(rvecs,[nPat 3]);
      szassert(tvecs,[nPat 3]);      
      nPts = size(patPtsXYZ,2);
      szassert(patPtsXYZ,[3 nPts]);
      szassert(patImPts,[2 nPts nPat 2]);
      
      R2WorldToCam1 = vision.internal.calibration.rodriguesVectorToMatrix(r2vec1);
      t2WorldToCam1 = t2vec1;
      R2WorldToCam2 = vision.internal.calibration.rodriguesVectorToMatrix(r2vec2);
      t2WorldToCam2 = t2vec2;
      
      uvcam = nan(2,nPts,nPat,2);
      for iPat=1:nPat
        RPatIToWorld = vision.internal.calibration.rodriguesVectorToMatrix(rvecs(iPat,:)');
        tPatIToWorld = tvecs(iPat,:)';
        patPtsWorld = RPatIToWorld*patPtsXYZ + tPatIToWorld;
        uvcam(:,:,iPat,1) = OrthoCam.project(patPtsWorld,R2WorldToCam1,...
          t2WorldToCam1,int1.k1,int1.k2,int1.mx,int1.my,int1.u0,int1.v0);
        uvcam(:,:,iPat,2) = OrthoCam.project(patPtsWorld,R2WorldToCam2,...
          t2WorldToCam2,int2.k1,int2.k2,int2.mx,int2.my,int2.u0,int2.v0);
      end
     
      d2 = sum((uvcam-patImPts).^2,1); % [1 nPts nPat 2]
      d2 = squeeze(d2);
      szassert(d2,[nPts nPat 2]);      
      d = sqrt(d2);
      
      dtmp = reshape(d,[nPts*nPat 2]);
      dmu = mean(dtmp);
    end
   
  end
  
  methods
    
    function invertSH(obj)
      % Specialized inversion for SH-style rig, where cam1 and cam2 are at
      % right angles with a common y (down-in-image) axis
      
      % extrinsics
      R2 = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2);
      R2(1,:) = -R2(1,:); % x-, z-coord in camera frame flipped
      R2(3,:) = -R2(3,:);
      obj.r2vec2 = vision.internal.calibration.rodriguesMatrixToVector(R2);      
      obj.t2vec2(1) = -obj.t2vec2(1);
      c = [4*.1;2.5*.1;0];
      khat = -obj.ijkCamWorld1(:,3);
      for iPat=1:obj.calNumPatterns
        r = obj.rvecs(iPat,:);
        t = obj.tvecs(iPat,:);
        R = vision.internal.calibration.rodriguesVectorToMatrix(r);
        [Rp,tp] = OrthoCam.computeDualPattern(R,t(:),c(:),khat(:));
        rp = vision.internal.calibration.rodriguesMatrixToVector(Rp);

        obj.rvecs(iPat,:) = rp;
        obj.tvecs(iPat,:) = tp;
      end
      
      % optCtr1 unchanged
      % optCtr2 unchanged
      % ijkCamWorld1 unchanged % [3x3] columns are "CamWorldCoords" i/j/k unit vecs in WorldSys for cam1; k is negatice optical axis
      ijkCW2 = [-1 0 0;0 1 0;0 0 -1]*obj.ijkCamWorld2; % x,z flipped
      [~,~,~,~,obj.ijkCamWorld2] = OrthoCam.opticalCenter(R2,obj.t2vec2);
      fprintf(1,'Manually adjusted ijkCamWorld2: \n');
      disp(ijkCW2);
      fprintf(1,'Recomputed ijkCamWorld2: \n');
      disp(obj.ijkCamWorld2);      
    end
     
    function objNew = recalibrate(obj)
      % Create new OrthoCamCalPair object
      
      assert(false,'Not working yet 20180615.');
      
      nPat = obj.calNumPatterns;
      nPts = obj.calNumPoints;
      patPtsXYZ = obj.calWorldPoints;
      patImPts = obj.calImPoints;
      oFcn = @(p)OrthoCamCalPair.oFcnStro(p,nPat,patPtsXYZ,patImPts);

      int1 = obj.tblInt(1,:);
      int2 = obj.tblInt(2,:);
      p0 = OrthoCam.packParamsStro(...
        int1.mx,int1.my,int1.u0,int1.v0,int1.k1,int1.k2,...
        int2.mx,int2.my,int2.u0,int2.v0,int2.k1,int2.k2,...
        obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,obj.rvecs,obj.tvecs);

      [~,dsum0] = oFcn(p0);
      fprintf('Starting residual: %.4g\n',dsum0);
      opts = OrthoCam.defaultoptsStro();
      pOpt = p0;      
      
      while 1
        pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);
%         [~,dsum0] = oFcn(pOpt);
%         if dsum0>1000
%           % none
%         else
%           break;
%         end
%         
        STOP = 'Stop optimization, looks good';
        RESTART = 'Restart optimization';
        CANCEL = 'Cancel';
        resp = questdlg('Restart optimization?','Optimization waypoint',...
          STOP,RESTART,CANCEL,RESTART);
        if isempty(resp)
          resp = CANCEL;
        end
        switch resp
          case STOP
            break;
          case RESTART
            % none; while loop will proceed
          case CANCEL
            error('Calibration canceled.');
        end
      end

      objNew = OrthoCamCalPair(pOpt,nPat,nPts,patPtsXYZ,patImPts,...
        obj.calPatternFPNs);      
    end
    
  end
  
  methods %CalRig

    function [zmin,zmax] = estimateEpiPolarZCoords(obj,iView,varargin)
      % Estimate z-range (along optical axis for iView's cam, relative to
      % optical ctr for iView) to sample when plotting epipolar lines in
      % *iView's complement view*.
      %
      % 20180202. up to this point, we have been hardcoding a z-range to
      % sample in computeEpiPolarLine. This z-range should depend on the 
      % Rig extrinsics (configuration, dimensions etc), as well the choice
      % of World CoordSys. Usually these don't change that much, so a 
      % single hardcoded range mostly works for a given rig.
      %
      % However, occassionally the range will be off enough that
      % nonlinearities cause two or more epipolar lines to appear on the
      % complement view. These extra EP lines arise from
      % far-out-of-realistic z-values compounded with nonlinear
      % distortions.
      %
      % To solve this problem in general, we compute a z-range of interest
      % based on all calibration patterns contained/observed in obj. Ie we 
      % assume that the union of all calpats reasonably spans the full FOV 
      % for both cameras, at least up to a modest scale factor.
      
      rangescalefac = myparse(varargin,...
        'rangescalefac',3.0); % fudge factor; expand z-span by this factor
      
      
      nPat = obj.calNumPatterns;
      nPts = obj.calNumPoints;
      rvcs = obj.rvecs;
      tvcs = obj.tvecs;
      patPtsXYZ = obj.calWorldPoints;
            
      if verLessThan('matlab','9.1')
        error('This method requires MATLAB version R2016b or later.');
      end
           
      [optCtr,ijkCam] = obj.getOptCtrCamWorldView(iView);
      
      zmin = inf;
      zmax = -inf;
      for iPat=1:nPat
        RPatIToWorld = vision.internal.calibration.rodriguesVectorToMatrix(rvcs(iPat,:)');
        tPatIToWorld = tvcs(iPat,:)';
        patPtsWorld = RPatIToWorld*patPtsXYZ + tPatIToWorld;
        szassert(patPtsWorld,[3 nPts]);
        
        patPtsOC = patPtsWorld-optCtr; % singleton expans
        patPtsZOC = sum(patPtsOC.*ijkCam(:,3),1);
        zmin = min(zmin,min(patPtsZOC(:)));
        zmax = max(zmax,max(patPtsZOC(:)));
      end      
      
      zmean = (zmin+zmax)/2;
      zmin = zmean - (zmean-zmin)*rangescalefac;
      zmax = zmean + (zmax-zmean)*rangescalefac;     
    end
    
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,uv1,iViewEpi,roi)
      % [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
      % 
      % iView1: either 1 (L) or 2 (R)
      % uv1: [2] x-y image coords
      % iViewEpi: either 1 (L) or 2 (R)
      %
      % xEPL, yEPL: [nx1] each; points in epipolar line
        
      if verLessThan('matlab','9.1')
        error('This method requires MATLAB version R2016b or later.');
      end
      
      assert(iView1==1 || iView1==2);
      assert(numel(uv1)==2);
      assert(iViewEpi==1 || iViewEpi==2);
      
      pq1 = obj.projected2normalized(uv1(:),iView1);      
      [optCtr,ijkCam] = obj.getOptCtrCamWorldView(iView1);
      
      O1 = optCtr + pq1(1)*ijkCam(:,1) + pq1(2)*ijkCam(:,2);
      szassert(O1,[3 1]);      
      
      if isempty(obj.estimatedEpipolarZCoords)
        obj.estimatedEpipolarZCoords = cell(obj.nviews,1);
      end
      if isempty(obj.estimatedEpipolarZCoords{iView1})
        [estzmin,estzmax] = obj.estimateEpiPolarZCoords(iView1);
        obj.estimatedEpipolarZCoords{iView1} = [estzmin estzmax];
      end
      
      zEst = obj.estimatedEpipolarZCoords{iView1};
      assert(numel(zEst)==2);
      NUMZPTS = 250;
      s = linspace(zEst(1),zEst(2),NUMZPTS);
      XEPL = O1 + s.*ijkCam(:,3);
      
      uvEPL = obj.project(XEPL,iViewEpi);
      rc = uvEPL([2 1],:)';
      rcCrop = obj.cropLines(rc,roi);
      xEPL = rcCrop(:,2);
      yEPL = rcCrop(:,1);
    end

    function [u_p,v_p,w_p] = reconstruct2d(obj,x,y,iView)
      assert(isequal(size(x),size(y)));
      assert(isvector(x));
      
      uv = [x(:) y(:)]';
      pq = obj.projected2normalized(uv,iView); % [2xn]
      
      assert(false,'TODO');
%       
%       % each col of pq is a normalized pt. The World-line corresponding to
%       % pq(:,i) is the camera axis 
%       szassert(pq,[2 n]);
%       
%       
%       n = numel(x);
%       dlt = obj.getDLT(iView);      
%       u_p = nan(n,2);
%       v_p = nan(n,2);
%       w_p = nan(n,2);
%       for i=1:n
%         [u_p(i,:),v_p(i,:),w_p(i,:)] = dlt_2D_to_3D(dlt,x(i),y(i));
%       end
    end
        
    function [x,y] = project3d(obj,u,v,w,iView)
      assert(isequal(size(u),size(v),size(w)));
      
      X = [u(:)';v(:)';w(:)'];
      uv = obj.project(X,iView);
      x = uv(1,:)';
      y = uv(2,:)';
    end
    
  end
  
end

function r2vecnew = lclHelpR(r2vec,R)
Rworld2cam = vision.internal.calibration.rodriguesVectorToMatrix(r2vec);
Rworld2camNew = Rworld2cam*R';
r2vecnew = vision.internal.calibration.rodriguesMatrixToVector(Rworld2camNew);
end