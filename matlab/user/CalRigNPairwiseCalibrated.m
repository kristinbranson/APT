classdef CalRigNPairwiseCalibrated < CalRig & matlab.mixin.Copyable
  % N-camera rig, with pairwise Caltech calibs  
      
  %CalRig
  properties
    nviews;
    viewNames;
  end
  properties (Dependent)
    ncams % same as nviews
  end  
  methods
    function v = get.ncams(obj)
      v = obj.nviews;
    end
  end
  
  properties
    % [ncams,ncams] cell array of pairwise calrig objs
    % This is a (strict) upper-triangular mat with an empty diagonal. 
    crigStros     
  end
      
  methods
    
    function obj = CalRigNPairwiseCalibrated(varargin)
      if nargin==1 
        if isstruct(varargin{1})
          s = varargin{1};
        end
      end
        
      ncam = s.nviews;
      obj.nviews = ncam;
      obj.crigStros = cell(ncam);
      crigs = s.calibrations;
      c = 1;
      % ordering of stereo crigs assumed
      for icam=1:ncam
      for jcam=icam+1:ncam
        obj.crigStros{icam,jcam} = crigs{c};
        c = c+1;
      end
      end
      
      assert(c==numel(crigs)+1);
      
      obj.viewNames = arrayfun(@(x)sprintf('view%d',x),(1:ncam)','uni',0);
    end     
    
  end
  
  methods

    % Helper. Get stereo crig and map "absolute" cam indices to left/right
    function [crigstro,ivw1stro,ivw2stro] = getStroCalRig(obj,iView1,iView2)
      % iView1/2: view indices
      %
      % crigstro: calrig object for iview1,2 pair
      % ivw1stro: 1 if iView1 maps to "left", 2 if "right"
      % ivw2stro: "    iView2  "
      
      if iView1<iView2
        crigstro = obj.crigStros{iView1,iView2};
        ivw1stro = 1;
        ivw2stro = 2;
      elseif iView2<iView1
        crigstro = obj.crigStros{iView2,iView1};
        ivw1stro = 2;
        ivw2stro = 1;
      else
        assert(false);
      end      
    end
        
    %CalRig
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,imroi)
      [crigstro,ivw1stro,ivw2stro] = obj.getStroCalRig(iView1,iViewEpi);
      [xEPL,yEPL] = crigstro.computeEpiPolarLine(ivw1stro,xy1,ivw2stro,imroi);      
    end
    
    %CalRig
    function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
      xRCT = nan;
      yRCT = nan;
    end
        
  end
      
  
end