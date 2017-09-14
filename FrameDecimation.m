classdef FrameDecimation < handle
  methods (Abstract)
    dec = getDecimation(obj,labelerObj)
  end
  methods
    function [str,decval] = getPrettyString(obj,labelerObj)
      decval = obj.getDecimation(labelerObj);
      if decval==1
        str = 'Every frame';
      else
        str = sprintf('Every %d frames',decval);
      end
    end
  end
end