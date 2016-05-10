classdef CalibratedRig < handle

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
  
  properties
    stroInfo; % info from stereo calibs

    int; % struct with fields 'l', 'r', 'b'. intrinsic params: .fc, .cc, etc

    om; % struct with fields like om.LB indicating "Bottom wrt Left" 
    T;
    R;
        
    roi; % struct, eg roi.l.height, roi.l.offsetX, etc
  end
    
  methods
    
    function obj = CalibratedRig(calibResLB,calibResBR)
      % reads intrinsic/extrinsic params from calibration files
      
      [nameL,nameR,ifoLB,intLB,extLB] = CalibratedRig.loadStroCalibResults(calibResLB);
      assert(strcmp(nameL,'cam0-')); %left
      assert(strcmp(nameR,'cam2-')); %bot
      [nameL,nameR,ifoBR,intBR,extBR] = CalibratedRig.loadStroCalibResults(calibResBR);
      assert(strcmp(nameL,'cam2-')); %bot
      assert(strcmp(nameR,'cam1-')); %right
      
      assert(isequal(ifoLB,ifoBR));
      obj.stroInfo = ifoLB;
      
      obj.int = struct();
      obj.int.l = intLB.l;
      obj.int.r = intBR.r;
      assert(isequal(intLB.r,intBR.l));
      obj.int.b = intLB.r;
      
      obj.om = struct();
      obj.T = struct();
      obj.R = struct();
      obj.om.LB = extLB.om;
      obj.T.LB = extLB.T;
      obj.R.LB = extLB.R;
      obj.om.BR = extBR.om;
      obj.T.BR = extBR.T;
      obj.R.BR = extBR.R;
    end
    
    function setROIs(obj,expdir)
      % Reads ROI info (sets .roi) from BIAS json files
      
      FILES = struct();
      FILES.l = 'bias_configCam0.json';
      FILES.r = 'bias_configCam1.json';
      FILES.b = 'bias_configCam2.json';
      
      s = struct();
      for cam={'l' 'r' 'b'},cam=cam{1}; %#ok<FXSET>
        tmp = loadjson(fullfile(expdir,FILES.(cam)));
        s.(cam) = tmp.camera.format7Settings.roi;        
      end
      
      obj.roi = s;
    end
    
  end
  
  methods % coordinate conversions, projections, reconstructions
    
    function x = y2x(obj,y,cam)
      % Transform from cropped points to image projection points.
      %
      % y: [Nx2] (row,col) pixel coords of N pts.
      % cam: camera specification: 'l', 'r', or 'b'.
      %
      % x: [Nx2]. Image points. See Coord Sys Notes above.
      
      assert(size(y,2)==2);
      
      % first "pad" to uncrop
      camroi = obj.roi.(cam);
      row = y(:,1);
      col = y(:,2);
      row = row + camroi.offsetY;
      col = col + camroi.offsetX;
      
      if strcmp(cam,'b')
        % flipUD; row=1 <-> row=height
        row = camroi.height - row + 1;
      end
      
      x = [col-1,row-1]; 
    end    
    
    function [XL,XB] = stereoTriangulateLB(obj,yL,yB)
      % Take cropped points for left/bot cameras and reconstruct 3D
      % positions
      %
      % yL: [2xN]. N points, (row,col) in cropped/final Left camera image
      % yB: [2xN]. 
      %
      % XL: [3xN]. 3d coords in Left camera frame
      % CB: [3xB].
      
      xL = obj.y2x(yL,'l');
      xB = obj.y2x(yB,'b');
      
      intL = obj.int.l;
      intB = obj.int.b;
      [XL,XB] = stereo_triangulation(xL',xB',...
        obj.om.LB,obj.T.LB,...
        intL.fc,intL.cc,intL.kc,intL.alpha_c,...
        intB.fc,intB.cc,intB.kc,intB.alpha_c);
    end
    
    function [XB,XR] = stereoTriangulateBR(obj,yB,yR)
      % see stereoTriangulateLB
      
      xB = obj.y2x(yB,'b');
      xR = obj.y2x(yR,'r');
      
      intB = obj.int.b;
      intR = obj.int.r;
      [XB,XR] = stereo_triangulation(xB',xR',...
        obj.om.BR,obj.T.BR,...
        intB.fc,intB.cc,intB.kc,intB.alpha_c,...
        intR.fc,intR.cc,intR.kc,intR.alpha_c);
    end
    
    function xp = normalized2projected(obj,xn,cam)
      intprm = obj.int.(cam);
      xp = CalibratedRig.normalized2projectedStc(xn,...
        intprm.alpha_c,intprm.cc,intprm.fc,intprm.kc);
    end
    
  end
  
  methods (Static)
    
    function xp = normalized2projectedStc(xn,alpha_c,cc,fc,kc)
      assert(isequal(size(xn),[2 1]));
      
      r2 = sum(xn.^2);
      radlDistortFac = 1 + kc(1)*r2 + kc(2)*r2^2 + kc(5)*r2^3;
      dx = [... % tangential distortion
        2*kc(3)*xn(1)*xn(2) + kc(4)*(r2+2*xn(1)^2); ...
        kc(3)*(r2+2*xn(2)^2) + 2*kc(4)*xn(1)*xn(2)];
      xd = radlDistortFac*xn + dx;
      
      KK = [fc(1) alpha_c*fc(1) cc(1); ...
            0     fc(2)         cc(2)];
      xp = KK * [xd;1];      
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
    
  end
    
% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
% [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
% [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  
  
end