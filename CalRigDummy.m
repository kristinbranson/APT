classdef CalRigDummy < CalRig
% Generic "No calibration" Calrig

  properties
    nviews
    viewNames % [nviews]. cellstr viewnames
    viewSizes % [nviews x 2]. viewSizes(iView,:) gives [nc nr] or [width height]
  end
  
  methods
    function obj = CalRigDummy(numVw,vwNames,vwSizes)
      obj.nviews = numVw;
      obj.viewNames = vwNames;
      obj.viewSizes = vwSizes;
    end
  end
  
  methods
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
      xEPL = nan;
      yEPL = nan;
    end    
    function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
      xRCT = nan(3,1);
      yRCT = nan(3,1);
    end
    function [u_p,v_p,w_p] = reconstruct2d(obj,x,y,iView)
      npt = numel(x);
      u_p = nan(npt,2);
      v_p = u_p;
      w_p = u_p;
    end    
    function [x,y] = project3d(obj,u,v,w,iView)
      x = nan(size(u));
      y = x;
    end
  end
end