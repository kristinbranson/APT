classdef CalRig2CamCaltech < CalRig & matlab.mixin.Copyable

  % Coord system notes
  %
  % XL, XR. These are physical 3D coords in the various camera frames. 
  % The camera frames are related to each other by the T, om matrices 
  % generating during stereo calibration. We call these as eg "physical 
  % coords in Left Camera frame" or "3D coords in Left camera frame".
  % 
  % xL, xR. These are 2D projections of points onto the Camera image 
  % planes. We refer to these as eg "Left camera image points" or "Left 
  % camera projection points." The coordinate system here is as in the 
  % Caltech toolbox:
  %   * The center of the upper-left pixel is [0;0]
  %   * The center of the upper-right pixel is [nx-1;0]
  %   * The center of the lower-left pixel is [0;ny-1], etc.
  %
  % yL, yR. These are the (row,col) pixel locations of points in the
  % cropped images. We refer to these as "Left camera cropped points".
  %
  % To transform from the x's to the X's, we use stereo_triangulation.
  % To transform from the X's to the x's, we use project_points2.
  % To transform from the y's to the x's, we pad (conceptually) to 'uncrop'
  %   and then we flipUD the bottom image.
  % To transform from the x's to the y's, we flipUD the bottom image and
  %   then crop using the ROIs.
    
  % Abstract in CalRig
  properties
    nviews = 2;
    viewNames = {'' ''}; % cam0 cam1
    viewSizes = [nan nan; nan nan];
  end
  properties
    % Min/Max meaningful values of normalized coords xn. For view iView,
    % viewXNlimits(iView,:) gives
    % 
    % [min-xn max-xn min-yn max-yn].
    %
    % This is used by normalized2projected(). The point is that, depending
    % on rig geometry, there are only certain regimes of normalized coords
    % that remotely make sense. When normalized coords far outside this
    % meaningful regime are used in normalized2projected(), the result can
    % be counterintuitive/incorrect/non-meaningful projected coords due to
    % high nonlinearities. To avoid these spurious artifacts,
    % normalized2projected() crops normalized points outside the
    % viewXNlimits.
    viewXNLimits = [-.35 .35 -.35 .35;-.35 .35 -.35 .35];
  end
  
  properties
    calibFile; % string
    stroInfo; % info from stereo calibs

    int; % struct with <viewNames> as fields. intrinsic params: .fc, .cc, etc
    
    % core/root calibration params;
    % X_R = R_RL*X_L + T_RL
    % When X_L=0, X_R=T_RL => T_RL is position of left cam wrt right
    omRL;
    TRL;
  end
  properties (Dependent,Hidden)
    % extrinsic props all derived from .om_RL, .T_RL
    RLR
    RRL
    TLR
  end
  properties (Dependent)
    R
    T
  end
  
  methods
    function R = get.RLR(obj)
      R = obj.RRL';
    end
    function R = get.RRL(obj)
      R = rodrigues(obj.omRL);
    end
    function T = get.TLR(obj)
      T = -obj.RRL'*obj.TRL;
    end
    function s = get.R(obj)
      s = struct(...
        'LR',obj.RLR,'RL',obj.RRL);
    end
    function s = get.T(obj)
      s = struct(...
        'LR',obj.TLR,'RL',obj.TRL);
    end
  end
    
  methods
    
    function obj = CalRig2CamCaltech(calibRes)
      % calibRes: full path to calibration file from Caltech toolbox
      %
      % Reads intrinsic/extrinsic params from calibration files
      
      obj.calibFile = calibRes;
      
      [nameL,nameR,ifo,int,ext] = CalibratedRig.loadStroCalibResults(calibRes,false);
      %assert(strncmp(nameL,'cam0',4),'Unexpected left/right cam names for stereo calibration results.');
      %assert(strncmp(nameR,'cam1',4),'Unexpected left/right cam names for stereo calibration results.');
      fprintf('Loaded: %s\n',calibRes);
      fprintf('''left'' cam: %s. ''right'' cam: %s.\n',nameL,nameR);
      
      obj.viewNames = {'L' 'R'}; % For now all props expect these viewNames
      obj.stroInfo = ifo;
      obj.int = struct('L',int.l,'R',int.r);
      obj.omRL = ext.om;
      obj.TRL = ext.T;
    end
    
  end
  
  methods
    
    %#OK
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,roiEpi)
      % See CalRig
      
      assert(numel(xy1)==2);
      %fprintf(1,'Cam %d: croppedcoords: %s\n',iAx,mat2str(round(pos(:)')));

      if iView1 == 1
        view_R = rodrigues(obj.omRL);
        view_T = obj.TRL;
        view_fc = obj.int.R.fc;
        view_cc = obj.int.R.cc;
        view_kc = obj.int.R.kc;
        view_alpha_c = obj.int.R.alpha_c;

        target_fc = obj.int.L.fc;
        target_cc = obj.int.L.cc;
        target_kc = obj.int.L.kc;
        target_alpha_c = obj.int.L.alpha_c;
      else
        % view_R = inv(obj.org_R);
        view_R = rodrigues(-obj.omRL);
        view_T = -view_R * obj.TRL;
        view_fc = obj.int.L.fc;
        view_cc = obj.int.L.cc;
        view_kc = obj.int.L.kc;
        view_alpha_c = obj.int.L.alpha_c;

        target_fc = obj.int.R.fc;
        target_cc = obj.int.R.cc;
        target_kc = obj.int.R.kc;
        target_alpha_c = obj.int.R.alpha_c;
      end

      xLp = [xy1(1), xy1(2)] - 1;
      xLp = xLp';
      
      % hack... we'll add 1000 to the last argument of compute_epipole
      % what this does is to compute 1000 points on the epipolar line. This
      % isn't great, the correct solution would probably to figure out the
      % the bounds of the image space before computing points.
      epipole = compute_epipole(xLp, view_R, view_T, view_fc, view_cc, view_kc, view_alpha_c, target_fc, target_cc, target_kc, target_alpha_c, 1000);
      xEPL = epipole(1, :) + 1;
      yEPL = epipole(2, :) + 1;
      
      % next lets limit the extents of the epipolar line.
      yEPL(xEPL < 1) = [];
      xEPL(xEPL < 1) = [];

      xEPL(yEPL < 1) = [];
      yEPL(yEPL < 1) = [];
      
      yEPL(xEPL > obj.stroInfo.nx) = [];
      xEPL(xEPL > obj.stroInfo.nx) = [];
      
      xEPL(yEPL > obj.stroInfo.ny) = [];
      yEPL(yEPL > obj.stroInfo.ny) = [];
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
    
    %#OK
    function xp = y2x(obj,y,cam)
      % Transform from cropped points to image projection points.
      %
      % y: [nx2] (row,col) pixel coords of N pts.
      % cam: camera specification
      %
      % xp: [2xn]. Image projected points. See Coord Sys Notes above.
      
      assert(size(y,2)==2);
      
%       % first "pad" to uncrop
%       camroi = obj.roi.(cam);
      row = y(:,1);
      col = y(:,2);
%       row = row + camroi.offsetY;
%       col = col + camroi.offsetX;
%       
%       if strcmpi(cam,'b')
%         % flipUD; row=1 <-> row=height
%         row = obj.YIMSIZE - row + 1;
%       end
      
      xp = [col-1,row-1]; 
      xp = xp';
    end
    
    %#OK
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

%       camroi = obj.roi.(cam);
% 
%       if strcmpi(cam,'b')
%         % flipUD; row=1 <-> row=height
%         row = obj.YIMSIZE - row + 1;
%       end
%       
%       row = row - camroi.offsetY;
%       col = col - camroi.offsetX; 
      y = [row(:) col(:)];
    end
     
    %#OK
    function xp = normalized2projected(obj,xn,cam)
      % 
      
      % crop xn
      iView = find(strcmp(cam,obj.viewNames));
      assert(isscalar(iView));
      xnLims = obj.viewXNLimits(iView,:);
      assert(size(xn,1)==2);
      tfXInBounds = xnLims(1) <= xn(1,:) & xn(1,:) <= xnLims(2);
      tfYInBounds = xnLims(3) <= xn(2,:) & xn(2,:) <= xnLims(4);
      tf = tfXInBounds & tfYInBounds;
      xnCropped = xn(:,tf);

      intprm = obj.int.(cam);
      xpCropped = CalibratedRig.normalized2projectedStc(xnCropped,...
        intprm.alpha_c,intprm.cc,intprm.fc,intprm.kc);
      xp = nan(size(xn));
      xp(:,tf) = xpCropped;
    end
    
    %#OK
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
    
    function X2 = viewXform(obj,X1,iView1,iView2)
      assert(false);
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
    
    function [X,xprp,rpe] = triangulate(obj,xp)
      % CalRig impl. Forward to stereoTriangulate
      % 
      % xp: final/image coords. (like y, but without row/col switch)
      
      [d,n,nvw] = size(xp);
      assert(nvw==obj.nviews);
            
      yL = xp([2 1],:,1)';
      yR = xp([2 1],:,2)';
      X = obj.stereoTriangulateLR(yL,yR);
      szassert(X,[3 n]);

      if nargout>1
        xprp = nan(d,n,nvw);
        rpe = nan(n,nvw);
      
        X2 = obj.camxform(X,'LR');
        xp1 = obj.project(X,'L');
        xp2 = obj.project(X2,'R');
        
        xprp(:,:,1) = xp1 + 1; % see .x2y
        xprp(:,:,2) = xp2 + 1;

        rpe = sqrt(sum((xp-xprp).^2,1));
        rpe = reshape(rpe,[n nvw]);
      end
    end
    
    function [XL,XR] = stereoTriangulateLR(obj,yL,yR)
      % Take cropped points for left/bot cameras and reconstruct 3D
      % positions
      %
      % yL: [Nx2]. N points, (row,col) in cropped/final Left camera image
      % yB: [Nx2]. 
      %
      % XL: [3xN]. 3d coords in Left camera frame
      % XB: [3xN].
      
      xL = obj.y2x(yL,'L');
      xB = obj.y2x(yR,'R');
      
      intL = obj.int.L;
      intR = obj.int.R;
      [XL,XR] = stereo_triangulation(xL,xB,...
        obj.omRL,obj.TRL,...
        intL.fc,intL.cc,intL.kc,intL.alpha_c,...
        intR.fc,intR.cc,intR.kc,intR.alpha_c);
    end
    
    function [XL,XB] = stereoTriangulateLB(obj,yL,yB)
      % Take cropped points for left/bot cameras and reconstruct 3D
      % positionsd
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
    
    function [X1,X2,d] = stereoTriangulateCropped(obj,y1,y2,cam1,cam2,...
        varargin)
      % Like stereoTriangulate
      %
      % y1, y2: [nx2].
      %
      % X1, X2: [3xn].
      % d: [n]
      
      wbObj = myparse(varargin,...
        'wbObj',[]);
      tfWB = ~isempty(wbObj);
      
      assert(isequal(size(y1),size(y2)));
      assert(size(y1,2)==2);
      n = size(y1,1);
      
      xp1 = obj.y2x(y1,cam1); % [2xn]
      xp2 = obj.y2x(y2,cam2);
      
      if tfWB
        wbObj.startPeriod('Triangulating','shownumden',true,'denominator',n);
      end
      
      X1 = nan(3,n);
      X2 = nan(3,n);
      d = nan(1,n);
      for i=1:n
        if tfWB
          tfCancel = wbObj.updateFracWithNumDen(i);
          if tfCancel
            return;
          end
        end
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
        
    %#OK
    function Xc2 = camxform(obj,Xc,type)
      % Extrinsic coord transformation: from camera1 coord sys to camera2
      %
      % Xc: [3xN], 3D coords in camera1 coord sys
      % type: 'lr', 'rl'. 'lr' means "transform from left camera frame to 
      % right camera frame", ie Xc/Xc2 are in the left/right camera frames 
      % resp.
      %
      % Xc2: [3xN], 3D  coords in camera2coord sys
      
      [d,N] = size(Xc);
      assert(d==3);
      
      type = upper(type);
      assert(ismember(type,{'LR' 'RL'}));
      type = type([2 1]); % convention for specification of T/R
      
      RR = obj.R.(type);
      TT = obj.T.(type);
      Xc2 = RR*Xc + repmat(TT,1,N);
    end
    
    function t = summarizeIntrinsics(obj)
      camnames = fieldnames(obj.int);
      sall = [];
      for cam=camnames(:)',cam=cam{1};
        s = obj.int.(cam);
        s.cc = s.cc.';
        s.fc = s.fc.';
        s.kc = s.kc.';
        sall = [sall; s];
      end
      t = struct2table(sall,'rownames',camnames,'asarray',1);
    end
        
  end
      
end