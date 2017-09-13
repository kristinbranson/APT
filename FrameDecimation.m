classdef FrameDecimation < handle
  
  enumeration
    EveryFrame([],1)
    EveryNFrameLarge('trackNFramesLarge',[])
    EveryNFrameSmall('trackNFramesSmall',[])
    
    % Weird, single (mutable) custom decimation across MATLAB session.
    % FrameDecimation.Custom is not short-lived like FrameSet.Custom, but
    % it might also be ok to have a single custom value representing "your
    % preferred nonstandard decimation"
    Custom([],1) 
  end
  
  properties
    labelerProp
    customValue % positive int. If present, overrides labelerProp.
  end
  
  methods
    
    function obj = FrameDecimation(lprop,val)
      obj.labelerProp = lprop;
      obj.customValue = val;
    end
    
    function decval = getDecimation(obj,labelerObj)
      lprop = obj.labelerProp;
      val = obj.customValue;
      if ~isempty(val)
        decval = val;
      elseif ~isempty(lprop)
        decval = labelerObj.(lprop);
      else
        decval = nan;
      end
    end
    
    function [v,decval] = getPrettyString(obj,labelerObj)
      decval = obj.getDecimation(labelerObj);
      if decval==1
        v = 'Every frame';
      else
        v = sprintf('Every %d frames',decval);
      end
    end
    
  end
end