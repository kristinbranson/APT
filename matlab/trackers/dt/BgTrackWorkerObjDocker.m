classdef BgTrackWorkerObjDocker < BgWorkerObjDocker & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjDocker(varargin)
      obj@BgWorkerObjDocker(varargin{:});
    end    
  end
end