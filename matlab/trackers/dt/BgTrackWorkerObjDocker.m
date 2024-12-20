classdef BgTrackWorkerObjDocker < BgWorkerObjDocker & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjDocker(nviews,track_type,varargin)
      obj@BgWorkerObjDocker(varargin{:});
      obj.nviews = nviews;
      obj.track_type = track_type ;
    end    
  end
end