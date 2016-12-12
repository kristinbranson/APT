classdef CalibratedRig2 < CalRig & matlab.mixin.Copyable
  % Romain calibrated rig

  % Coord system notes
  %
  % XL, XR, XB. These are physical 3D coords in the various camera frames. 
  % The camera frames are related to each other by the T, om matrices 
  % generating during stereo calibration. We call these as eg "physical 
  % coords in Left Camera frame" or "3D coords in Left camera frame".
  % 
  % xL, xR, xB. These are 2D projections of points onto the Camera image 
  % planes. We refer to these as eg "Left camera image points" or "Left 
  % camera projection points." The coordinate system here is as in the 
  % Caltech toolbox:
  %   * The center of the upper-left pixel is [0;0]
  %   * The center of the upper-right pixel is [nx-1;0]
  %   * The center of the lower-left pixel is [0;ny-1], etc.
  %
  % yL, yR, yB. These are the (row,col) pixel locations of points in the
  % cropped images. We refer to these as "Left camera cropped points".
  %
  % To transform from the x's to the X's, we use stereo_triangulation.
  % To transform from the X's to the x's, we use project_points2.
  % To transform from the y's to the x's, we pad (conceptually) to 'uncrop'
  %   and then we flipUD the bottom image.
  % To transform from the x's to the y's, we flipUD the bottom image and
  %   then crop using the ROIs.
  
  % DOF notes
  %
  % If you consider the Bottom cam the origin, then there are 12 extrinsic 
  % DOFs: 3 rots + 3 translations for Left cam wrt Bottom cam, 3 rots + 3
  % translations for Right cam wrt Bottom cam.
  %
  % If you consider all cams wrt a fixed frame (eg the sample/fly), then
  % there are 18 DOFs. However the cam calibration toolbox only gives us
  % one cam wrt another.
  
  
  properties (Constant)
    BIASCONFIGFILES = struct(...
      'L','bias_configCam0.json',...
      'R','bias_configCam1.json',...
      'B','bias_configCam2.json');

    YIMSIZE = 1200;
  end
    
  % Abstract in CalRig
  properties
    nviews = 3;
    viewNames = {'L' 'R' 'B'};
  end
  properties (Dependent)
    viewSizes
  end        
  
  properties
    calibFileLB; % string
    calibFileBR; % string
    
    stroInfo; % info from stereo calibs

    int; % struct with <viewNames> as fields. intrinsic params: .fc, .cc, etc
    
    % core/root calibration params;
    % X_B = R_BL*X_L + T_BL
    % When XL=0, XB=TBL => TBL is position of left cam wrt bot
    omBL;
    omBR;
    TBL;
    TBR;
            
    roi; % struct, eg roi.<viewName>.height, roi.<viewName>.offsetX, etc
    roiDir; % string
  end
  properties (Dependent,Hidden)
    % extrinsic props all derived from .omBL, .omBR, .TBL, .TBR
    RBL
    RLB
    RBR
    RRB
    RLR
    RRL
    TLB
    TRB
    TLR
    TRL
  end
  properties (Dependent)
    % for backwards compat
    om
    R
    T    
  end
  
  methods
    function v = get.viewSizes(obj)
      v = nan(obj.nviews,2);
      for iView = 1:obj.nviews
        vName = obj.viewNames{iView};
        vRoi = obj.roi.(vName);
        v(iView,:) = [vRoi.width vRoi.height];
      end
    end
    
    function R = get.RBL(obj)
      R = rodrigues(obj.omBL);
    end
    function R = get.RLB(obj)
      R = obj.RBL';
    end
    function R = get.RBR(obj)
      R = rodrigues(obj.omBR);
    end
    function R = get.RRB(obj)
      R = obj.RBR';
    end
    function R = get.RLR(obj)
      R = obj.RLB*obj.RBR;
    end
    function R = get.RRL(obj)
      R = obj.RRB*obj.RBL;
    end
    function T = get.TLB(obj)
      T = -obj.RBL'*obj.TBL;
    end
    function T = get.TRB(obj)
      T = -obj.RBR'*obj.TBR;
    end
    function T = get.TLR(obj)
      T = obj.RLB*obj.TBR + obj.TLB;
    end
    function T = get.TRL(obj)
      T = obj.RRB*obj.TBL + obj.TRB;
    end
    
    function s = get.om(obj)
      sR = obj.R;
      flds = fieldnames(sR);
      s = struct();
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        s.(f) = rodrigues(sR.(f));
      end
    end      
    function s = get.R(obj)
      s = struct(...
        'BR',obj.RBR,'RB',obj.RRB,...
        'BL',obj.RBL,'LB',obj.RLB,...
        'LR',obj.RLR,'RL',obj.RRL);
    end
    function s = get.T(obj)
      s = struct(...
        'BR',obj.TBR,'RB',obj.TRB,...
        'BL',obj.TBL,'LB',obj.TLB,...
        'LR',obj.TLR,'RL',obj.TRL);
    end
  end
    
  methods
    
    function obj = CalibratedRig2(calibResLB,calibResBR)
      % reads intrinsic/extrinsic params from calibration files
      
      obj.calibFileLB = calibResLB;
      obj.calibFileBR = calibResBR;
      
      [nameL_LB,nameR_LB,ifoLB,intLB,extLB] = CalibratedRig.loadStroCalibResults(calibResLB);
      [nameL_BR,nameR_BR,ifoBR,intBR,extBR] = CalibratedRig.loadStroCalibResults(calibResBR);
      
      assert(isequal(ifoLB,ifoBR));
      obj.stroInfo = ifoLB;
      
      LRSTR = {'l' 'r'};
      namesLR_LB = {nameL_LB;nameR_LB};
      tfCam0 = strncmp(namesLR_LB,'cam0',4);
      tfCam2 = strncmp(namesLR_LB,'cam2',4);
      assert(nnz(tfCam0)==1 && nnz(tfCam2)==1,'Unexpected left/right cam names for calibResLB.');
      RIG_LB2CAL_LR = struct();
      RIG_LB2CAL_LR.L = LRSTR{tfCam0};
      RIG_LB2CAL_LR.B = LRSTR{tfCam2};
      CAL_LR2RIG_LB = struct();
      CAL_LR2RIG_LB.(LRSTR{tfCam0}) = 'L';
      CAL_LR2RIG_LB.(LRSTR{tfCam2}) = 'B';
      
      namesLR_BR = {nameL_BR;nameR_BR};
      tfCam1 = strncmp(namesLR_BR,'cam1',4);
      tfCam2 = strncmp(namesLR_BR,'cam2',4);
      assert(nnz(tfCam1)==1 && nnz(tfCam2)==1,'Unexpected left/right cam names for calibResBR.');
      RIG_BR2CAL_LR = struct();
      RIG_BR2CAL_LR.R = LRSTR{tfCam1};
      RIG_BR2CAL_LR.B = LRSTR{tfCam2};
      CAL_LR2RIG_BR = struct();
      CAL_LR2RIG_BR.(LRSTR{tfCam1}) = 'R';
      CAL_LR2RIG_BR.(LRSTR{tfCam2}) = 'B';
      
      fprintf('Loaded: %s\n',calibResLB);
      fprintf('''left'' cam: %s. ''right'' cam: %s.\n',nameL_LB,nameR_LB);
      fprintf(' => rigL cam is ''%s'' and rigB cam is ''%s''\n',RIG_LB2CAL_LR.L,RIG_LB2CAL_LR.B);
      fprintf('Loaded: %s\n',calibResBR);
      fprintf('''left'' cam: %s. ''right'' cam: %s.\n',nameL_BR,nameR_BR);
      fprintf(' => rigB cam is ''%s'' and rigR cam is ''%s''\n',RIG_BR2CAL_LR.B,RIG_BR2CAL_LR.R);
      
      obj.int = struct();
      obj.int.L = intLB.(RIG_LB2CAL_LR.L);
      obj.int.R = intBR.(RIG_BR2CAL_LR.R);
      assert(isequal(intLB.(RIG_LB2CAL_LR.B),intBR.(RIG_BR2CAL_LR.B)));
      obj.int.B = intLB.(RIG_LB2CAL_LR.B);
      
      omm = struct();
      TT = struct();
%       RR = struct();
      
      % read off om, T, R values from stereo calib results
      rightWRTLeftLB = [CAL_LR2RIG_LB.r CAL_LR2RIG_LB.l]; % eg: 'BL'
      rightWRTLeftBR = [CAL_LR2RIG_BR.r CAL_LR2RIG_BR.l];
      omm.(rightWRTLeftLB) = extLB.om;
      TT.(rightWRTLeftLB) = extLB.T;
%       RR.(rightWRTLeftLB) = extLB.R;
      omm.(rightWRTLeftBR) = extBR.om;
      TT.(rightWRTLeftBR) = extBR.T;
%       RR.(rightWRTLeftBR) = extBR.R;
      
%       % invert to get reverse values
%       leftWRTRightLB = rightWRTLeftLB(end:-1:1);
%       leftWRTRightBR = rightWRTLeftBR(end:-1:1);
%       omm.(leftWRTRightLB) = -omm.(rightWRTLeftLB);
%       omm.(leftWRTRightBR) = -omm.(rightWRTLeftBR);
%       [RR.(leftWRTRightLB),TT.(leftWRTRightLB)] = ...
%         CalibratedRig.invertRT(RR.(rightWRTLeftLB),TT.(rightWRTLeftLB));
%       [RR.(leftWRTRightBR),TT.(leftWRTRightBR)] = ...
%         CalibratedRig.invertRT(RR.(rightWRTLeftBR),TT.(rightWRTLeftBR));
      
%       % fill out T, R matrices for all xform pairs
%       [RR.LR,TT.LR] = CalibratedRig.composeRT(RR.BR,TT.BR,RR.LB,TT.LB);
%       [RR.RL,TT.RL] = CalibratedRig.composeRT(RR.BL,TT.BL,RR.RB,TT.RB);
%       % Sanity
%       [RR_RL2,TT_RL2] = CalibratedRig.invertRT(RR.LR,TT.LR);
%       dR = abs(RR.RL-RR_RL2);
%       dT = abs(TT.RL-TT_RL2);
%       fprintf(1,'Sanity check R_RL, T_RL: %.6g %.4g\n',...
%         sum(dR(:).^2),sum(dT(:).^2));      

      obj.omBL = omm.BL;
      obj.omBR = omm.BR;
      obj.TBL = TT.BL;
      obj.TBR = TT.BR;
%       obj.om = omm;
%       obj.R = RR;
%       obj.T = TT;
    end
      
    function setROIs(obj,expdir)
      % Reads ROI info (sets .roi) from BIASCONFIGFILES
            
      s = struct();
      for cam=obj.viewNames(:)',cam=cam{1}; %#ok<FXSET>
        jsonfile = fullfile(expdir,obj.BIASCONFIGFILES.(cam));
        tmp = loadjson(jsonfile);
        s.(cam) = tmp.camera.format7Settings.roi;
      end
      
      obj.roi = s;
      obj.roiDir = expdir;      
    end
    
  end
  
  methods
    
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
      % See CalRig
      
      assert(numel(xy1)==2);
      %fprintf(1,'Cam %d: croppedcoords: %s\n',iAx,mat2str(round(pos(:)')));
      
      cam1 = obj.viewNames{iView1};
      camEpi = obj.viewNames{iViewEpi};
      
      y = [xy1(2) xy1(1)];
      xp = obj.y2x(y,cam1);
      assert(isequal(size(xp),[2 1]));
      xn1 = obj.projected2normalized(xp,cam1);
      
      % create 3D segment by projecting normalized coords into 3D space
      % (coord sys of cam1)
      Zc1 = 0:.25:100; % mm
      Xc1 = [xn1(1)*Zc1; xn1(2)*Zc1; Zc1];
      
      XcEpi = obj.camxform(Xc1,[cam1 camEpi]); % 3D seg, in frame of cam2
      xnEpi = [XcEpi(1,:)./XcEpi(3,:); XcEpi(2,:)./XcEpi(3,:)]; % normalize
      xpEpi = obj.normalized2projected(xnEpi,camEpi); % project
      
      yEpi = obj.x2y(xpEpi,camEpi);
      yEpi = obj.cropLines(yEpi,iViewEpi);
      r2 = yEpi(:,1);
      c2 = yEpi(:,2);
      xEPL = c2;
      yEPL = r2;
    end
    
    function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
      % See CalRig
      
      assert(numel(xy1)==2);
      assert(numel(xy2)==2);
      
      cam1 = obj.viewNames{iView1};
      cam2 = obj.viewNames{iView2};
      camRct = obj.viewNames{iViewRct};
      
      % get projected pts for 1 (anchor) and 2 (second)
      % AL20160624: some unfortunate semi-legacy naming here
      y1 = [xy1(2) xy1(1)];
      y2 = [xy2(2) xy2(1)];
      xp1 = obj.y2x(y1,cam1);
      xp2 = obj.y2x(y2,cam2);
      assert(isequal(size(xp1),size(xp2),[2 1]));
      
      [X1,X2,d,P,Q] = obj.stereoTriangulate(xp1,xp2,cam1,cam2);
      % X1: [3x1]. 3D coords in frame of camera1
      % X2: etc
      % d: error/discrepancy in closest approach
      % P: 3D point of closest approach on normalized ray of camera 1, in
      % frame of camera 2
      % Q: 3D point of closest approach on normalized ray of camera 2, in
      % frame of camera 2
      
      X3 = obj.camxform(X2,[cam2 camRct]);
      P3 = obj.camxform(P,[cam2 camRct]);
      Q3 = obj.camxform(Q,[cam2 camRct]);
      
      xp3 = obj.project(X3,camRct);
      pp3 = obj.project(P3,camRct);
      qp3 = obj.project(Q3,camRct);
      yx3 = obj.x2y(xp3,camRct);
      yp3 = obj.x2y(pp3,camRct);
      yq3 = obj.x2y(qp3,camRct);
      assert(isequal(size(yx3),size(yp3),size(yq3),[1 2])); % [row col]
      
      xRCT = [yp3(2) yx3(2) yq3(2)];
      yRCT = [yp3(1) yx3(1) yq3(1)];
    end
        
  end
  
  methods % coordinate conversions, projections, reconstructions
    
    function xp = y2x(obj,y,cam)
      % Transform from cropped points to image projection points.
      %
      % y: [nx2] (row,col) pixel coords of N pts.
      % cam: camera specification: 'l', 'r', or 'b'. 
      %
      % xp: [2xn]. Image projected points. See Coord Sys Notes above.
      
      assert(size(y,2)==2);
      
      % first "pad" to uncrop
      camroi = obj.roi.(cam);
      row = y(:,1);
      col = y(:,2);
      row = row + camroi.offsetY;
      col = col + camroi.offsetX;
      
      if strcmpi(cam,'b')
        % flipUD; row=1 <-> row=height
        row = obj.YIMSIZE - row + 1;
      end
      
      xp = [col-1,row-1]; 
      xp = xp';
    end
    
    function y = x2y(obj,xp,cam)
      % Transform projected points to cropped points.
      %
      % xp: [2xn] image projection pts
      % cam: etc
      %
      % y: [nx2]. (row,col) pixel cropped coords
      
      assert(size(xp,1)==2);

      col = xp(1,:)+1; % projected points are 0-based
      row = xp(2,:)+1; 

      camroi = obj.roi.(cam);

      if strcmpi(cam,'b')
        % flipUD; row=1 <-> row=height
        row = obj.YIMSIZE - row + 1;
      end
      
      row = row - camroi.offsetY;
      col = col - camroi.offsetX; 
      y = [row(:) col(:)];
    end
     
    function xp = normalized2projected(obj,xn,cam)
      intprm = obj.int.(cam);
      xp = CalibratedRig.normalized2projectedStc(xn,...
        intprm.alpha_c,intprm.cc,intprm.fc,intprm.kc);
    end
    
    function [xn,fval] = projected2normalized(obj,xp,cam,varargin)
      % Find normalized coords corresponding to projected coords.
      % This uses search/optimization to invert normalized2projected; note
      % the toolbox also has normalize().
      %
      % xp: [2x1]
      % cam: 'l', 'r', or 'b'
      % 
      % xn: [2x1]
      % fval: optimization stuff, eg final residual

      tfUseFminSearch = myparse(varargin,'tfUseFminSearch',false);
      
      assert(isequal(size(xp),[2 1]));
      
      if tfUseFminSearch
        fcn = @(xnguess) sum( (xp-obj.normalized2projected(xnguess(:),cam)).^2 );
        xn0 = [0;0];
        opts = optimset('TolX',1e-6);
        [xn,fval] = fminsearch(fcn,xn0,opts);        
      else
        intCam = obj.int.(cam);
        xn = normalize_pixel(xp,intCam.fc,intCam.cc,intCam.kc,intCam.alpha_c);
        fval = [];
      end
    end
    
    function xp = project(obj,X,cam)
      % X: [3xN] 3D coords in frame of cam
      % cam: 'l','r','b'
      %
      % xp: [2xN] projected image coords for cam
        
      assert(size(X,1)==3);
      intPrm = obj.int.(cam); 
      xp = project_points2(X,[0;0;0],[0;0;0],...
        intPrm.fc,intPrm.cc,intPrm.kc,intPrm.alpha_c);
    end
    
    function [r,c] = projectCPR(obj,X,iView)
      cam = obj.viewNames{iView};
      xp = obj.project(X,cam);
      y = obj.x2y(xp,cam);
      assert(size(y,2)==2);
      r = y(:,1);
      c = y(:,2);
    end
    
    function [XL,XB] = stereoTriangulateLB(obj,yL,yB)
      % Take cropped points for left/bot cameras and reconstruct 3D
      % positions
      %
      % yL: [Nx2]. N points, (row,col) in cropped/final Left camera image
      % yB: [Nx2]. 
      %
      % XL: [3xN]. 3d coords in Left camera frame
      % XB: [3xN].
      
      xL = obj.y2x(yL,'L');
      xB = obj.y2x(yB,'B');
      
      intL = obj.int.L;
      intB = obj.int.B;
      [XL,XB] = stereo_triangulation(xL,xB,...
        obj.om.BL,obj.T.BL,...
        intL.fc,intL.cc,intL.kc,intL.alpha_c,...
        intB.fc,intB.cc,intB.kc,intB.alpha_c);
    end
    
    function [XB,XR] = stereoTriangulateBR(obj,yB,yR)
      % see stereoTriangulateLB
      
      xB = obj.y2x(yB,'B');
      xR = obj.y2x(yR,'R');
      
      intB = obj.int.B;
      intR = obj.int.R;
      [XB,XR] = stereo_triangulation(xB,xR,...
        obj.om.RB,obj.T.RB,...
        intB.fc,intB.cc,intB.kc,intB.alpha_c,...
        intR.fc,intR.cc,intR.kc,intR.alpha_c);
    end
    
    function [X1,X2,d] = stereoTriangulateCropped(obj,y1,y2,cam1,cam2)
      % Like stereoTriangulate
      %
      % y1, y2: [nx2].
      %
      % X1, X2: [3xn].
      % d: [n]
      
      assert(isequal(size(y1),size(y2)));
      assert(size(y1,2)==2);
      n = size(y1,1);
      
      xp1 = obj.y2x(y1,cam1); % [2xn]
      xp2 = obj.y2x(y2,cam2);
      
      X1 = nan(3,n);
      X2 = nan(3,n);
      d = nan(1,n);
      for i=1:n
        [X1(:,i),X2(:,i),d(i)] = obj.stereoTriangulate(xp1(:,i),xp2(:,i),cam1,cam2);
      end      
    end
    
    function [X1,X2,d,P,Q] = stereoTriangulate(obj,xp1,xp2,cam1,cam2)
      % xp1: [2x1]. projected pixel coords, camera1
      % xp2: etc
      % cam1: 'l, 'r', or 'b'
      % cam2: etc
      %
      % X1: [3x1]. 3D coords in frame of camera1
      % X2: etc
      % d: error/discrepancy in closest approach
      % P: 3D point of closest approach on normalized ray of camera 1, in
      % frame of camera 2
      % Q: 3D point of closest approach on normalized ray of camera 2, in
      % frame of camera 2
            
      xn1 = obj.projected2normalized(xp1,cam1);
      xn2 = obj.projected2normalized(xp2,cam2);
      xn1 = [xn1;1];
      xn2 = [xn2;1];
      
      % get P0,u,Q0,v in frame of cam2
      rtype = upper([cam2 cam1]);
      RR = obj.R.(rtype);
      O1 = obj.camxform([0;0;0],[cam1 cam2]); % camera1 origin in camera2 frame
      n1 = RR*xn1; % pt1 normalized ray in camera2 frame
      O2 = [0;0;0]; % camera2 origin in camera2 frame
      n2 = xn2; % pt2 normalized ray in camera2 frame
      
      [P,Q,d] = CalibratedRig.stereoTriangulateRays(O1,n1,O2,n2);
      
      X2 = (P+Q)/2;
      X1 = obj.camxform(X2,[cam2 cam1]);      
    end
    
    function [X1,X2] = stereoTriangulateFast(obj,xp1,xp2,cam1,cam2)
      int1 = obj.int.(cam1);
      int2 = obj.int.(cam2);
      cam21 = [cam2 cam1];
      [X1,X2] = stereo_triangulation(xp1,xp2,...
        obj.om.(cam21),obj.T.(cam21),...
        int1.fc,int1.cc,int1.kc,int1.alpha_c,...
        int2.fc,int2.cc,int2.kc,int2.alpha_c);
    end
        
    function Xc2 = camxform(obj,Xc,type)
      % Extrinsic coord transformation: from camera1 coord sys to camera2
      %
      % Xc: [3xN], 3D coords in camera1 coord sys
      % type: 'lr', 'rl, 'bl', 'lb', 'br', or 'rb'. 'lr' means "transform
      % from left camera frame to right camera frame", ie Xc/Xc2 are in the
      % left/right camera frames resp.
      %
      % Xc2: [3xN], 3D coords in camera2coord sys
      
      [d,N] = size(Xc);
      assert(d==3);
      
      type = upper(type);
      switch type
        case {'LL' 'RR' 'BB'}
          Xc2 = Xc;
          return;
        case {'LR' 'RL' 'BL' 'LB' 'BR' 'RB'}
          % none
        otherwise
          error('CalibratedRig2:type','Unsupported type: ''%s''.',type);
      end
      type = type([2 1]); % convention for specification of T/R
      
      RR = obj.R.(type);
      TT = obj.T.(type);
      Xc2 = RR*Xc + repmat(TT,1,N);
    end
    
    function X2 = viewXformCPR(obj,X1,iView1,iView2)
      vNames = obj.viewNames;
      type = [vNames{iView1} vNames{iView2}];
      X2 = obj.camxform(X1,type);     
    end
    
  end
  
  methods (Static)
    
    function [P,Q,d,sc,tc] = stereoTriangulateRays(P0,u,Q0,v)
      % "Closest approach" analysis of two rays
      %
      % P0, Q0: [3x1], 3D "origin" pts
      % u, v: [3x1], vector emanating from P0, Q0, resp.
      %
      % P: [3x1]. 3D pt lying on P0+s*u that comes closest to Q ray
      % Q: [3x1]. etc.
      % d: Euclidean distance between P and Q
      % sc: value of s which minimimizes distance, ie P = P0 + sc*u
      % tc: etc
      
      assert(isequal(size(P0),[3 1]));
      assert(isequal(size(u),[3 1]));
      assert(isequal(size(Q0),[3 1]));
      assert(isequal(size(v),[3 1]));
      
      a = dot(u,u);
      b = dot(u,v);
      c = dot(v,v);
      w0 = P0-Q0;
      d = dot(u,w0);
      e = dot(v,w0);
      sc = (b*e-c*d)/(a*c-b*b);
      tc = (a*e-b*d)/(a*c-b*b);
      P = P0 + sc*u;
      Q = Q0 + tc*v;
      d = norm(P-Q,2);
    end
      
    function xp = normalized2projectedStc(xn,alpha_c,cc,fc,kc)
      % xn: [2xn], normalized coords
      %
      % xp: [2xn], projected coords
      
      [d,n] = size(xn);
      assert(d==2);
      
      r2 = sum(xn.^2,1); % [1xn], r-squared for each pt
      radlDistortFac = 1 + kc(1)*r2 + kc(2)*r2.^2 + kc(5)*r2.^3; % [1xn]
      dx = [... % tangential distortion
        2*kc(3)*xn(1,:).*xn(2,:) + kc(4)*(r2+2*xn(1,:).^2); ...
        kc(3)*(r2+2*xn(2,:).^2) + 2*kc(4)*xn(1,:).*xn(2,:)]; % [2xn]
      xd = repmat(radlDistortFac,2,1).*xn + dx; % [2xn]
      
      KK = [fc(1) alpha_c*fc(1) cc(1); ...
            0     fc(2)         cc(2)];
      xp = KK * [xd;ones(1,n)];
    end
    
    function [nameL,nameR,info,intrinsic,extrinsic] = loadStroCalibResults(file)
      cr = load(file);
      
      nameL = cr.calib_name_left;
      nameR = cr.calib_name_right;
      
      assert(cr.recompute_intrinsic_left==0);
      assert(cr.recompute_intrinsic_right==0);      
      info = struct();
      info.dX = cr.dX;
      info.nx = cr.nx;
      info.ny = cr.ny;
      info.normT = 2*cr.dX;
      
%       ccDefault = [(cr.nx-1)/2; (cr.ny-1)/2];
      fprintf(1,'[cc_left cc_left_error]:\n');
      disp([round(cr.cc_left) cr.cc_left_error]);
      fprintf(1,'[cc_right cc_right_error]:\n');
      disp([round(cr.cc_right) cr.cc_right_error]);
%       fprintf(1,'Using ccDefault:\n');
%       disp(ccDefault);
      intrinsic = struct();
%       intrinsic.l.cc = ccDefault;
%       intrinsic.r.cc = ccDefault;
      PRMS = {'alpha_c' 'cc' 'fc' 'kc'}; 
      for p=PRMS,p=p{1}; %#ok<FXSET>
        intrinsic.l.(p) = cr.([p '_left']);
        intrinsic.r.(p) = cr.([p '_right']);
      end

      extrinsic = struct();
      extrinsic.om = cr.om; % om, T, R are for coords-in-Right-frame wrt coords-in-Left-frame
      extrinsic.T = cr.T;
      extrinsic.R = rodrigues(cr.om); % X_Right = R*X_Left + T
    end
    
    function [R1,T1] = invertRT(R,T)
      R1 = R';
      T1 = -R'*T;
    end    
    function [R31,T31] = composeRT(R21,T21,R32,T32)
      % RMN, TMN:
      % X_M = RMN * X_N + TMN

      R31 = R32*R21;
      T31 = R32*T21+T32;      
    end    
    
  end
    
  methods
    function [yLre,yRre,yBre,errL,errR,errB,errLfull,errRfull,errBfull,...
        XL,XR,XB,...
        errL_reLR,errR_reLR,errB_reLR,...
        XLlr,XRlr,XBlr] = calibRoundTripFull(obj,yL,yR,yB)
      % Use each pair of views to recon the 3rd
      %
      % yL/yR/yB: [Nx2]  (row,col) in cropped coords
      %
      % yLre: [Nx2] reconstructed yL from yR, yB
      % yRre: etc
      % errL: nanmean(errLfull)
      % errLfull: [Nx1] L2 err for each pt
      % XL: [3xN]: 3d recon (best guess), in camL frame
      % errL_reLR: like errL, but computed using stereo triangularion of L/R
      % views
      
      [XLlb,XBlb] = obj.stereoTriangulateLB(yL,yB);
      [XBbr,XRbr] = obj.stereoTriangulateBR(yB,yR);
      
      % best guess: average lb, br recons
      XBbest = (XBlb+XBbr)/2;
      XRbest = obj.camxform(XBbest,'br');
      XLbest = obj.camxform(XBbest,'bl');
      
      xpR_re = obj.project(XRbest,'R');
      xpL_re = obj.project(XLbest,'L');
      xpB_re = obj.project(XBbest,'B');
      yRre = obj.x2y(xpR_re,'R');
      yLre = obj.x2y(xpL_re,'L');
      yBre = obj.x2y(xpB_re,'B');
      
      errLfull = sqrt(sum((yL-yLre).^2,2));
      errRfull = sqrt(sum((yR-yRre).^2,2));
      errBfull = sqrt(sum((yB-yBre).^2,2));
      errL = nanmean(errLfull); % mean L2 distance (in px) across all GT pts
      errR = nanmean(errRfull);
      errB = nanmean(errBfull);
      
      XL = XLbest;
      XR = XRbest;
      XB = XBbest;
      
      [XLlr,XRlr] = obj.stereoTriangulateCropped(yL,yR,'L','R');
      XBlr = obj.camxform(XLlr,'lb');
      xpR_reLR = obj.project(XRlr,'R');
      xpL_reLR = obj.project(XLlr,'L');
      xpB_reLR = obj.project(XBlr,'B');
      yR_reLR = obj.x2y(xpR_reLR,'R');
      yL_reLR = obj.x2y(xpL_reLR,'L');
      yB_reLR = obj.x2y(xpB_reLR,'B');
      
      errLfull_reLR = sqrt(sum((yL-yL_reLR).^2,2));
      errRfull_reLR = sqrt(sum((yR-yR_reLR).^2,2));
      errBfull_reLR = sqrt(sum((yB-yB_reLR).^2,2));
      errL_reLR = nanmean(errLfull_reLR); % mean L2 distance (in px) across all GT pts
      errR_reLR = nanmean(errRfull_reLR);
      errB_reLR = nanmean(errBfull_reLR);
    end
  end
  
  
% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
% [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
% [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  
  
end