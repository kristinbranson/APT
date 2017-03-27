classdef CPRBlurPreProc < handle
  properties
    name % char desc
    timestamp
    
    sig1 % [n1] vector (positive reals) of initial blur SDs 
    sig2 % [n2] etc
    iChan % vector into [S(:) SGS(:) SLS(:)] (total number of elements: n1+n1*n2+n1*n2)
          % of preprocessed channels to use
    
    sRescale % scalar logical, if true perform rescaling of S (blur images)
    sRescaleFacs % [n1] rescaling factors for S (blur images)
    sgsRescale % scalar logical, if true perform rescaling of SGS (blur-gradmag-blur)
    sgsRescaleFacs % [n1xn2] rescaling factors for SGS
    slsRescale % scalar logical, if true perform rescaling of SLS (blur-laplace-blur)
    slsRescaleFacs % [n1xn2], etc
  end
  properties (Dependent)
    n1
    n2
  end
  
  methods
    function v = get.n1(obj)
      v = numel(obj.sig1);
    end
    function v = get.n2(obj)
      v = numel(obj.sig2);
    end
    function set.sRescaleFacs(obj,v)
      assert(numel(v)==obj.n1); %#ok<MCSUP>
      obj.sRescaleFacs = v(:);
    end
    function set.sgsRescaleFacs(obj,v)
      szassert(v,[obj.n1 obj.n2]); %#ok<MCSUP>
      obj.sgsRescaleFacs = v;
    end
    function set.slsRescaleFacs(obj,v)
      szassert(v,[obj.n1 obj.n2]); %#ok<MCSUP>
      obj.slsRescaleFacs = v;
    end
  end
    
  methods
    function obj = CPRBlurPreProc(nm,s1,s2)
      obj.name = nm;
      obj.timestamp = now();
      
      obj.sig1 = s1;
      obj.sig2 = s2;
      
      obj.sRescale = false;
      obj.sRescaleFacs = nan(obj.n1,1);
      obj.sgsRescale = false;
      obj.sgsRescaleFacs = nan(obj.n1,obj.n2);
      obj.slsRescale = false;
      obj.slsRescaleFacs = nan(obj.n1,obj.n2);
    end
  end
  
end