classdef FrameDecimationVariable < FrameDecimation
  properties
    getDecimationHookFcn % function with sig dec = getDecimationHookFcn(lObj)
  end
  methods
    function obj = FrameDecimationVariable(fcn)
      obj.getDecimationHookFcn = fcn;
    end
    function dec = getDecimation(obj,lObj)
      dec = obj.getDecimationHookFcn(lObj);
    end
  end
  
  properties (Constant) % canned/enumerated vals
    EveryNFrameLarge = FrameDecimationVariable(@(lo)lo.trackNFramesLarge);
    EveryNFrameSmall = FrameDecimationVariable(@(lo)lo.trackNFramesSmall);
  end  
end