classdef CalRigSH < CalRig
  
  properties
    nviews = 2;
    viewNames = {'side' 'front'};
    viewSizes = [1024 1024;1024 1024]; % [nviews x 2]. viewSizes(iView,:) gives [nc nr] or [width height]
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
    
    function obj = CalRigSH
      obj.setKineData('');
    end
    
    function setKineData(obj,kdfile)
      if isempty(kdfile)
        obj.kineData = [];
        obj.kineDataFile = '';
      else
        kd = load(kdfile);
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
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
      kdata = obj.kineData;
      dlt_side = kdata.cal.coeff.DLT_1;
      dlt_front = kdata.cal.coeff.DLT_2;
      vsz = obj.viewSizes(iViewEpi,:);
      if iView1==1 && iViewEpi==2 %view1==side; viewEpi==front
        dlt1 = dlt_side;
        dlt2 = dlt_front;
      elseif iView1==2 && iViewEpi==1
        dlt1 = dlt_front;
        dlt2 = dlt_side;
      else
        assert(false);
      end
      [xEPL,yEPL] = im_pt_2_im_line(xy1(1),xy1(2),dlt1,dlt2,...
        [1 vsz(1) 1 vsz(2)],obj.epLineNPts);
      rc = obj.cropLines([yEPL(:) xEPL(:)],iViewEpi);
      xEPL = rc(:,2);
      yEPL = rc(:,1);
    end
    
    function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
      assert(false,'Unsupported');
    end
    
  end

end
