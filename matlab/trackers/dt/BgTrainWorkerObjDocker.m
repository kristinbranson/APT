classdef BgTrainWorkerObjDocker < BgWorkerObjDocker & BgTrainWorkerObj  
  
  methods
    
    function obj = BgTrainWorkerObjDocker(varargin)
      obj@BgWorkerObjDocker(varargin{:});
    end
    
  end
    
end
