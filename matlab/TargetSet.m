classdef TargetSet < handle
  methods (Abstract)    
    str = getPrettyString(obj,labelerObj)
    iTgts = getTargetIndices(obj,labelerObj,mIdx)
  end
end