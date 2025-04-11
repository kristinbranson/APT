classdef FrameDecimation < handle
  methods (Abstract)
    dec = getDecimation(obj,labelerObj)
  end
  methods
    function [str,decval] = getPrettyString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      decval = obj.getDecimation(labelerObj);
      if decval==1
        str = 'Every frame';
      else
        str = sprintf('Every %d frames',decval);
      end
    end
    function [str,decval] = getPrettyCompactString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      decval = obj.getDecimation(labelerObj);
      if decval==1
        str = '';
      else
        str = sprintf('Every %d fr',decval);
      end
    end
  end
end