classdef FrameDecimationFixed < FrameDecimation
  properties
    decVal
    id % id for testing for special cases
  end
  methods
    function obj = FrameDecimationFixed(dval,id)
      obj.decVal = dval;
      if nargin < 2,
        obj.id = 'custom';
      else
        obj.id = id;
      end
    end
    function dec = getDecimation(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;                  
      dec = obj.decVal;
    end
  end
    
  properties (Constant) % canned/enumerated vals
    EveryFrame = FrameDecimationFixed(1,'every');
  end
end