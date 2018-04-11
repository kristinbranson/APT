classdef DeepTracker < LabelTracker
  
  properties
    sPrm % new-style DT params
  end
  
  methods    
    function setParams(obj,sPrm)
      obj.sPrm = sPrm;
    end
    function sPrm = getParams(obj)
      sPrm = obj.sPrm;
    end
  end
  
  methods
    function obj = DeepTracker(lObj)
      obj@LabelTracker(lObj);
    end
  end
  
  methods
    function s = getSaveToken(obj)
      s = struct();
      s.sPrm = obj.sPrm;
    end
    function loadSaveToken(obj,s)
      obj.sPrm = s.sPrm;
    end
  end
end