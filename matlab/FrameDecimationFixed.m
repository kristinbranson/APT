classdef FrameDecimationFixed < FrameDecimation
  properties
    decVal
  end
  methods
    function obj = FrameDecimationFixed(dval)
      obj.decVal = dval;
    end
    function dec = getDecimation(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;                  
      dec = obj.decVal;
    end
  end
    
  properties (Constant) % canned/enumerated vals
    EveryFrame = FrameDecimationFixed(1);
  end
end