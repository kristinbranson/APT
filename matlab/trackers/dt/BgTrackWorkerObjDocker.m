classdef BgTrackWorkerObjDocker < BgWorkerObjDocker & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjDocker(varargin)
      obj@BgTrackWorkerObj(varargin{:});
    end    
  end
end