classdef CalRigZhang2CamBase < CalRig

% Technique taken from
% Z. Zhang, "A flexible new technique for camera calibration," in IEEE Transactions on
% Pattern Analysis and Machine Intelligence, vol. 22, no. 11, pp. 1330-1334, Nov. 2000,
% doi: 10.1109/34.888718.
  
  properties 
    % Abstract in CalRig
    nviews = 2;
    viewNames = {'cam1' 'cam2'};

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
    viewXNLimits = [ 
      -.35 .35 -.35 .35
      -.35 .35 -.35 .35
      ];

    % eplineZrange{ivw} gives range of z-coords (in coord sys of cam ivw) 
    % that should be used when projecting EP lines into other vw.
    % See autoCalibrate* methods
    eplineZrange = {0:.1:100 0:.1:100};
    
    % tolerance for projected2normalized nonlinear inversion. See
    % autoCalibrate* methods
    proj2NormFuncTol = 1e-8;
  end

  % Extrinsics
  properties (Abstract,Dependent,Hidden)
    RLR
    RRL
    TLR
    TRL
  end
  properties (Dependent)
    R
    T
  end  
  methods
    
    function obj = CalRigZhang2CamBase(s)
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

    function Xc2 = camxform(obj,Xc,type)
      % Extrinsic coord transformation: from camera1 coord sys to camera2
      %
      % Xc: [3xN], 3D coords in camera1 coord sys
      % type: either one of {'lr','rl'} or one of {[1 2],[2 1]}. 
      %   'lr' ([1 2]) means "transform from left camera frame to
      %   right camera frame", ie Xc/Xc2 are in the left/right camera 
      %   frames resp.
      %
      % Xc2: [3xN], 3D coords in camera2coord sys
      
      [d,N] = size(Xc);
      assert(d==3);
      
      TYPES = 'LR';
      if isnumeric(type)
        type = TYPES(type);
      else
        type = upper(type);
      end
      assert(ismember(type,{'LR' 'RL'}));
      % Warning: this is in CalRig2CamCaltech for some convention
      % type = type([2 1]); % convention for specification of T/R
      
      RR = obj.R.(type);
      TT = obj.T.(type);
      Xc2 = RR*Xc + repmat(TT,1,N);
    end
    
    function xp = normalized2projected(obj,xn,cam)
      %
      % xn: [2xn]
      % cam: an index (numeric) or value (char) of .viewNames
      %
      % xp: [2xn]
      
      [iView,camName] = obj.camArgHelper(cam);
      
      % crop xn      
      xnLims = obj.viewXNLimits(iView,:);
      assert(size(xn,1)==2);
      tfXInBounds = xnLims(1) <= xn(1,:) & xn(1,:) <= xnLims(2);
      tfYInBounds = xnLims(3) <= xn(2,:) & xn(2,:) <= xnLims(4);
      tf = tfXInBounds & tfYInBounds;
      xnCropped = xn(:,tf);
      
      intprm = obj.int.(camName);
      xpCropped = CalRigZhang2CamBase.normalized2projectedStc(xnCropped,...
        intprm.alpha_c,intprm.cc,intprm.fc,intprm.kc);
      xp = nan(size(xn));
      xp(:,tf) = xpCropped;
    end
    
    function [xn,fval] = projected2normalized(obj,xp,cam,varargin)
      % Find normalized coords corresponding to projected coords.
      % This uses search/optimization to invert normalized2projected; note
      % the toolbox also has normalize().
      %
      % xp: [2x1]
      % cam: an index (numeric) or value (char) of .viewNames
      % 
      % xn: [2x1]
      % fval: optimization stuff, eg final residual

      % TODO: vectorize me
      
      assert(isequal(size(xp),[2 1]));
      
      functol = myparse(varargin,...
        'functol',obj.proj2NormFuncTol...
        );
      
      fcn = @(xnguess) sum( (xp-obj.normalized2projected(xnguess(:),cam)).^2 );
      xn0 = [0;0];
      
      % Old 
      % opts = optimset('TolX',1e-6);
      % [xn,fval] = fminsearch(fcn,xn0,opts);

      % These worked better for RF
      opts = optimoptions('lsqnonlin',...
        'Algorithm','levenberg-marquardt',...
        'FunctionTolerance',functol,...
        'Display','none');
      [xn,fval] = lsqnonlin(fcn,xn0,[],[],opts);
    end
    
    function [X1,X2,d,P,Q] = stereoTriangulateBase(obj,xp1,xp2,cam1,cam2)
      % xp1: [2x1]. projected pixel coords, camera1
      % xp2: etc
      % cam1: one of .viewNames
      % cam2: etc
      %
      % X1: [3x1]. 3D coords in frame of camera1
      % X2: etc
      % d: error/discrepancy in closest approach
      % P: 3D point of closest approach on normalized ray of camera 1, in
      % frame of camera 2
      % Q: 3D point of closest approach on normalized ray of camera 2, in
      % frame of camera 2
      %
      % TODO: vectorize me
            
      icam1 = obj.camArgHelper(cam1);
      icam2 = obj.camArgHelper(cam2);
      if icam1==1 && icam2==2
        rtype = 'LR';
      elseif icam1==2 && icam2==1
        rtype = 'RL';
      else
        assert(false);
      end
        
      xn1 = obj.projected2normalized(xp1,cam1);
      xn2 = obj.projected2normalized(xp2,cam2);
      xn1 = [xn1;1];
      xn2 = [xn2;1];
      
      % get P0,u,Q0,v in frame of cam2
      %rtype = upper([cam2 cam1]);
      RR = obj.R.(rtype);
      O1 = obj.camxform([0;0;0],rtype); % camera1 origin in camera2 frame
      n1 = RR*xn1; % pt1 normalized ray in camera2 frame
      O2 = [0;0;0]; % camera2 origin in camera2 frame
      n2 = xn2; % pt2 normalized ray in camera2 frame
      
      [P,Q,d] = CalRigZhang2CamBase.stereoTriangulateRays(O1,n1,O2,n2);
      
      X2 = (P+Q)/2;
      X1 = obj.camxform(X2,rtype([2 1]));
    end
    
    function [iView,viewName] = camArgHelper(obj,cam)
      % This seems pretty dumb, but may have been semi-useful at some point
      
      if isnumeric(cam)
        iView = cam;
        viewName = obj.viewNames{iView};
      else
        iView = find(strcmp(cam,obj.viewNames));
        viewName = cam;
      end
    end

  end
  
  methods (Static)
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
  end
end
