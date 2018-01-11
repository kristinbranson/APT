classdef FrameSet < handle
  methods (Abstract)
    str = getPrettyString(obj,labelerObj)
    frms = getFrames(obj,labelerObj,iMov,iTgt,decFac)
  end
end