classdef BgTrackWorkerObjBsub < BgWorkerObjBsub & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjBsub(nviews,varargin)
      obj@BgTrackWorkerObj(varargin{:});
      obj.nviews = nviews;
    end    
  end
end