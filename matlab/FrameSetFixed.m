classdef FrameSetFixed < FrameSet
  properties
    frames
  end
  methods
    function obj = FrameSetFixed(frms)
      assert(isvector(frms));
      obj.frames = frms(:);
    end
    function str = getPrettyString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      n = numel(obj.frames);
      str = sprintf('%d frames from %d to %d',n,obj.frames(1),obj.frames(end));
    end    
    function frms = getFrames(obj,labelerObj,iMov,iTgt,decFac)
      frms = obj.frames;
      frms = frms(1:decFac:numel(frms));
    end
  end
end