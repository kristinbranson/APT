classdef TargetSet < handle
  methods (Abstract)    
    str = getPrettyString(obj,labelerObj)
    iTgts = getTargetIndices(obj,labelerObj,iMov)
  end
end