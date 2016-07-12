classdef CalRigSH < CalRig
  
  properties
    nviews = 2;
    viewNames = {'side' 'front'};
    viewSizes = [1024 1024;1024 1024]; % [nviews x 2]. viewSizes(iView,:) gives [nc nr] or [width height]
  end
  
  properties (SetAccess=private)
    kineData; % scalar struct; kinemat calibration data
    kineDataFile; % string    
  end
    
  properties
    epLineNPts = 250;
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
        kd = kd.data;
        if ~all(isfield(kd,{'types' 'cal' 'kine'}))
          error('CalRigSH:kine',...
            'Unexpected contents in Kinemat calibration file: ''%s''',kdfile);
        end
        obj.kineData = kd;
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
