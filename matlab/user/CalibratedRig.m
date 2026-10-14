classdef CalibratedRig < CalRig & matlab.mixin.Copyable
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
    
    

    om; % 
    T; % For om, T and R, eg X_bot = R.BL * X_left + T.BL
    R; % 
        
    roi; % struct, eg roi.<viewName>.height, roi.<viewName>.offsetX, etc
    roiDir; % string
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
  end
    
  methods
    
    function obj = CalibratedRig(calibResLB,calibResBR)
      % reads intrinsic/extrinsic params from calibration files
      
      obj.calibFileLB = calibResLB;
      obj.calibFileBR = calibResBR;
      
      [nameL_LB,nameR_LB,ifoLB,intLB,extLB] = loadStroCalibResults(calibResLB);
      [nameL_BR,nameR_BR,ifoBR,intBR,extBR] = loadStroCalibResults(calibResBR);
      
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
      RR = struct();
      
      % read off om, T, R values from stereo calib results
      rightWRTLeftLB = [CAL_LR2RIG_LB.r CAL_LR2RIG_LB.l]; % eg: 'BL'
      rightWRTLeftBR = [CAL_LR2RIG_BR.r CAL_LR2RIG_BR.l];
      omm.(rightWRTLeftLB) = extLB.om;
      TT.(rightWRTLeftLB) = extLB.T;
      RR.(rightWRTLeftLB) = extLB.R;
      omm.(rightWRTLeftBR) = extBR.om;
      TT.(rightWRTLeftBR) = extBR.T;
      RR.(rightWRTLeftBR) = extBR.R;
      
      % invert to get reverse values
      leftWRTRightLB = rightWRTLeftLB(end:-1:1);
      leftWRTRightBR = rightWRTLeftBR(end:-1:1);
      omm.(leftWRTRightLB) = -omm.(rightWRTLeftLB);
      omm.(leftWRTRightBR) = -omm.(rightWRTLeftBR);
      [RR.(leftWRTRightLB),TT.(leftWRTRightLB)] = ...
        invertRT(RR.(rightWRTLeftLB),TT.(rightWRTLeftLB));
      [RR.(leftWRTRightBR),TT.(leftWRTRightBR)] = ...
        invertRT(RR.(rightWRTLeftBR),TT.(rightWRTLeftBR));
      
      % fill out T, R matrices for all xform pairs
      [RR.LR,TT.LR] = composeRT(RR.BR,TT.BR,RR.LB,TT.LB);
      [RR.RL,TT.RL] = composeRT(RR.BL,TT.BL,RR.RB,TT.RB);
      % Sanity
      [RR_RL2,TT_RL2] = invertRT(RR.LR,TT.LR);
      dR = abs(RR.RL-RR_RL2);
      dT = abs(TT.RL-TT_RL2);
      fprintf(1,'Sanity check R_RL, T_RL: %.6g %.4g\n',...
        sum(dR(:).^2),sum(dT(:).^2));      

      obj.om = omm;
      obj.R = RR;
      obj.T = TT;
    end
      
    function setROIs(obj,expdir)
      % Reads ROI info (sets .roi) from BIASCONFIGFILES
            
      s = struct();
      for cam=obj.viewNames(:)',cam=cam{1}; %#ok<FXSET>
        jsonfile = fullfile(expdir,obj.BIASCONFIGFILES.(cam));
        tmp = jsondecode(fileread(jsonfile));
        s.(cam) = tmp.camera.format7Settings.roi;
      end
      
      obj.roi = s;
      obj.roiDir = expdir;      
    end
    
  end
  
  methods 
    
    function verify(obj,varargin)
      % check self-consistency of .om, .R, .T. 
      % TODO: just have either om or R, not both
      
      silent = myparse(varargin,...
        'silent',false);
      
      
      fldsOm = fieldnames(obj.om);
      fldsR = fieldnames(obj.R);
      fldsT = fieldnames(obj.T);
      fldsOmR = intersect(fldsOm,fldsR);
      if ~silent
        fprintf('Checking om/R fields: %s\n',...
          String.cellstr2CommaSepList(fldsOmR));
      end
      for f = fldsOmR(:)',f=f{1}; %#ok<FXSET>
        d = obj.om.(f) - rodrigues(obj.R.(f));
        assert(max(abs(d(:))) <= eps(10));
      end
      fldsOmUnchecked = setdiff(fldsOm,fldsOmR);
      fldsRUnchecked = setdiff(fldsR,fldsOmR);
      if ~isempty(fldsOmUnchecked) && ~silent
        fprintf('Not checking .om fields (om/R): %s\n',...
          String.cellstr2CommaSepList(fldsOmUnchecked));
      end
      if ~isempty(fldsRUnchecked) && ~silent
        fprintf('Not checking .R fields (om/R): %s\n',...
          String.cellstr2CommaSepList(fldsRUnchecked));
      end
      
      fldsRT = intersect(fldsR,fldsT);
      fldsRUnchecked = setdiff(fldsR,fldsRT);
      fldsTUnchecked = setdiff(fldsT,fldsRT);
      if ~silent
        fprintf('Checking R/T fields: %s\n',...
          String.cellstr2CommaSepList(fldsRT));
      end
      for f = fldsRT(:)',f=f{1}; %#ok<FXSET>
        fflip = f(end:-1:1);
        d = obj.T.(f) - ( -obj.R.(fflip)'*obj.T.(fflip) );
        assert(max(abs(d(:))) <= 1e-13);
      end
      if ~isempty(fldsRUnchecked) && ~silent
        fprintf('Not checking .R fields (R/T): %s\n',...
          String.cellstr2CommaSepList(fldsRUnchecked));
      end
      if ~isempty(fldsTUnchecked) && ~silent
        fprintf('Not checking .T fields (R/T): %s\n',...
          String.cellstr2CommaSepList(fldsTUnchecked));
      end      
    end
    
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
      %yEpi = obj.cropLines(yEpi,iViewEpi);
      yEpi = obj.getLineWithinAxes(yEpi,iViewEpi);
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
      xp = normalized2projectedCoords(xn,...
        intprm.alpha_c,intprm.cc,intprm.fc,intprm.kc);
    end
    
    function [xn,fval] = projected2normalized(obj,xp,cam)
      % Find normalized coords corresponding to projected coords.
      % This uses search/optimization to invert normalized2projected; note
      % the toolbox also has normalize().
      %
      % xp: [2x1]
      % cam: 'l', 'r', or 'b'
      % 
      % xn: [2x1]
      % fval: optimization stuff, eg final residual

      assert(isequal(size(xp),[2 1]));
      
      fcn = @(xnguess) sum( (xp-obj.normalized2projected(xnguess(:),cam)).^2 );
      xn0 = [0;0];
      opts = optimset('TolX',1e-6);
      [xn,fval] = fminsearch(fcn,xn0,opts);
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
      
      [P,Q,d] = stereoTriangulateRays(O1,n1,O2,n2);
      
      X2 = (P+Q)/2;
      X1 = obj.camxform(X2,[cam2 cam1]);      
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
      assert(ismember(type,{'LR' 'RL' 'BL' 'LB' 'BR' 'RB'}));      
      type = type([2 1]); % convention for specification of T/R
      
      RR = obj.R.(type);
      TT = obj.T.(type);
      Xc2 = RR*Xc + repmat(TT,1,N);
    end
    
  end
  
% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
% [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
% [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  
  
end