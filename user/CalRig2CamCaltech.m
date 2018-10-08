classdef CalRig2CamCaltech < CalRigZhang2CamBase & matlab.mixin.Copyable

% XXX 20181005. WIP subclassing from CalRigZhang2CamBase. This class is 
% probably currently not functional. There are minor differences in 
% conventions etc that need to be resolved. If we need this, it may be 
% better to revert this commit.
  
  % Coord system notes
  %
  % XL, XR. These are physical 3D coords in the various camera frames. 
  % The camera frames are related to each other by the T, om matrices 
  % generating during stereo calibration. We call these as eg "physical 
  % coords in Left Camera frame" or "3D coords in Left camera frame".
  %
  % xLn, xRn. Normalized coords in a camera frame are computed by dividing
  % by the Z-coord in that frame, eg xLn(1:2) = XL(1:2)/XL(3). Points lying
  % along a 3D ray emanating from the camera all share the same normalized
  % coords; xLn = (tan(theta_x),tan(theta_y)).
  % 
  % xLp, xRp. Projected coords. These are 2D projections of points onto the 
  % Camera image planes. We refer to these as eg "Left camera image points" 
  % or "Left camera projection points." The coordinate system here is as in
  % the Caltech toolbox:
  %   * The center of the upper-left pixel is [0;0]
  %   * The center of the upper-right pixel is [nx-1;0]
  %   * The center of the lower-left pixel is [0;ny-1], etc.
  %
  % yL, yR. These are the (row,col) pixel locations of points in the
  % cropped images. We refer to these as "Left camera cropped points".
  %
  % To transform from the x's to the X's, we use stereo_triangulation.
  % To transform from the X's to the xp's, we use project_points2.
  % - alternately, it is easy to go from the X's to the xn's, then apply i)
  %   distortions, ii) magnification&skew, and iii) offset (principal
  %   point) to reach the xp's. See normalized2projected.
  % To transform from the y's to the x's, we pad (conceptually) to 'uncrop'
  %   and then we flipUD the bottom image.
  % To transform from the x's to the y's, we flipUD the bottom image and
  %   then crop using the ROIs.
  
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
    
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,roiEpi)
      assert(false,'TODO: call computeEpiPolarLine@basecls');
      % wrap call in .y2x and .x2y
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
            
  end
      
end