classdef BgTrackWorkerObjDocker < BgWorkerObjDocker & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjDocker(nviews,varargin)
      obj@BgWorkerObjDocker(varargin{:});
      obj.nviews = nviews;
    end    
  end
end