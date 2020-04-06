classdef BgTrackWorkerObjBsub < BgWorkerObjBsub & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjBsub(varargin)
      obj@BgTrackWorkerObj(varargin{:});
    end    
  end
end