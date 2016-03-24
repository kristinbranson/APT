classdef CalibratedRig

%   properties (Constant)
%     INTRINSIC_PRMS = struct('fc',[],'cc',[],'kc',[],'alphac',[],'fcerr',[],'ccErr',[],'kcErr',[],'alphacErr',[]);
%   end
  properties    
    intL;
    intR;
    intB_L;
    intB_R;
    
    stroBL;
    stroBR;
    
    omBL
    omBR
    TBL
    TBR
  end
  properties (Dependent)
    camL % camera matrix
    camR
    camB
  end
  
  methods 
    function m = get.camL(obj)
      
    end
    function m = get.camR(obj)
    end
    function m = get.camB(obj)
    end
  end
  
  methods
    
    function obj = CalibratedRig(cdirBL,cdirBR)
      if exist('cdirBR','var')==0
        cdirBR = 'F:\DropBoxNEW\Dropbox\MultiViewFlyLegTracking\CamerasCalibration\Bottom-Right';
      end
      if exist('cdirBL','var')==0
        cdirBL = 'F:\DropBoxNEW\Dropbox\MultiViewFlyLegTracking\CamerasCalibration\Left-Bottom';
      end

      cfileL = fullfile(cdirBL,'Calib_Results_Left.mat');
      cfileR = fullfile(cdirBR,'Calib_Results_Right.mat');
      cfileB_L = fullfile(cdirBL,'Calib_Results_Bottom.mat');
      cfileB_R = fullfile(cdirBR,'Calib_Results_Bottom.mat');
      obj.intL = load(cfileL);
      obj.intR = load(cfileR);
      obj.intB_L = load(cfileB_L);
      obj.intB_R = load(cfileB_R);
      
      strofileBL = fullfile(cdirBL,'Calib_Results_stereo.mat');
      strofileBR = fullfile(cdirBR,'Calib_Results_stereo.mat');
      %obj.stroBL = load(strofileBL);
      obj.stroBR = load(strofileBR);
      
      
      
    end
    
  end
    

    
% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
% [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
% [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  
  
end