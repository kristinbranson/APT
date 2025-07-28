classdef CalRigSH < CalRig
  
  properties
    nviews = 2;
    viewNames = {'side' 'front'};
%    viewSizes = [1024 1024;1024 1024]; % [nviews x 2]. viewSizes(iView,:) gives [nc nr] or [width height]
  end
  
  properties (SetAccess=private)
    kineData; % .kineData.cal.coeff.DLT_1, ...DLT2
    kineDataFile; % string 
  end
    
  properties
    % AL20160923. Increasing this to a high number to workaround apparent
    % MATLAB low-level graphics issue where EPline does not display. See
    % LabelCoreMultiViewCalibrated2.
    epLineNPts = 1e4; 
  end
  
  methods
    
    function obj = CalRigSH(s)
      if ~exist('s','var'),
        obj.setKineData('');
        return;
      end
      propscopy = {'kineData','kineDataFile','kineData'};
      for i = 1:numel(propscopy),
        fn = propscopy{i};
        if isfield(s,fn),
          obj.(fn) = s.(fn);
        end
      end
    end
    
    function setKineData(obj,kdfile)
      if isempty(kdfile)
        obj.kineData = [];
        obj.kineDataFile = '';
      else
        %try
        kd = load(kdfile,'data','DLT_1','DLT_2');
        %catch ME
         % warning('CalRigSH:kine','Error caught loading kineData file: %s: %s\n',...
         %   kdfile,ME.message);
        %end
        if isfield(kd,'DLT_1') && isfield(kd,'DLT_2')
          dlt1 = kd.DLT_1;
          dlt2 = kd.DLT_2;
        elseif isfield(kd,'data') && isfield(kd.data,'cal')
          dlt1 = kd.data.cal.coeff.DLT_1;
          dlt2 = kd.data.cal.coeff.DLT_2;
        else
          error('CalRigSH:kine',...
            'Do not recognize contents of calibration MAT-file: ''%s''. Expect two variables ''DLT_1'' and ''DLT_2''.',kdfile);
        end
        tmp = struct;
        tmp.cal.coeff.DLT_1 = dlt1; % legacy structure
        tmp.cal.coeff.DLT_2 = dlt2;
        obj.kineData = tmp;
        obj.kineDataFile = kdfile;
      end
    end
    
    % iView1: view index for anchor point
    % xy1: [2]. [x y] vector, cropped coords in iView1
    % iViewEpi: view index for target view (where EpiLine will be drawn)
    %
    % xEPL,yEPL: epipolar line, cropped coords, iViewEpi
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,roiEpi)
      kdata = obj.kineData;
      dlt_side = kdata.cal.coeff.DLT_1;
      dlt_front = kdata.cal.coeff.DLT_2;
%      vsz = obj.viewSizes(iViewEpi,:);
      if iView1==1 && iViewEpi==2 %view1==side; viewEpi==front
        dlt1 = dlt_side;
        dlt2 = dlt_front;
      elseif iView1==2 && iViewEpi==1
        dlt1 = dlt_front;
        dlt2 = dlt_side;
      else
        assert(false);
      end
      [xEPL,yEPL] = im_pt_2_im_line(xy1(1),xy1(2),dlt1,dlt2,roiEpi,...
        obj.epLineNPts);
      %rc = obj.cropLines([yEPL(:) xEPL(:)],iViewEpi);
      rc = obj.getLineWithinAxes([yEPL(:) xEPL(:)],roiEpi);
      xEPL = rc(:,2);
      yEPL = rc(:,1);
    end
    
    function dlt = getDLT(obj,iView)
      kdata = obj.kineData;
      switch iView
        case 1
          dlt = kdata.cal.coeff.DLT_1;
        case 2
          dlt = kdata.cal.coeff.DLT_2;
        otherwise
          assert(false);
      end
    end
    
    function [X,xyrp,rpe] = triangulate(obj,xy,varargin)
      % CalRig impl
      %
      % X: 3D coord sys is fixed/specified by dlt coefficients.
      
      [d,n,nvw] = size(xy);
      assert(nvw==obj.nviews);
      
      dlt1 = obj.getDLT(1);
      dlt2 = obj.getDLT(2);
      
      [X,~,~,~,xyrp(:,:,1),xyrp(:,:,2)] = ...
        dlt_2D_to_3D_point_vectorized(dlt1,dlt2,xy(:,:,1),xy(:,:,2),varargin{:});
      if nargout>2
        rpe = sqrt(sum((xy-xyrp).^2,1));
        rpe = reshape(rpe,[n nvw]);
      end
    end
    
    function [u_p,v_p,w_p] = reconstruct2d(obj,x,y,iView)
      assert(isequal(size(x),size(y)));
      assert(isvector(x));
      
      n = numel(x);
      dlt = obj.getDLT(iView);      
      u_p = nan(n,2);
      v_p = nan(n,2);
      w_p = nan(n,2);
      for i=1:n
        [u_p(i,:),v_p(i,:),w_p(i,:)] = dlt_2D_to_3D(dlt,x(i),y(i));
      end
    end
        
    function [x,y] = project3d(obj,u,v,w,iView)
      assert(isequal(size(u),size(v),size(w)));
      
      dlt = obj.getDLT(iView);
      [x,y] = arrayfun(@(uu,vv,ww)dlt_3D_to_2D(dlt,uu,vv,ww),u,v,w);
    end
    
  end

end
