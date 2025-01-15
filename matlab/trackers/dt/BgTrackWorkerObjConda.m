classdef BgTrackWorkerObjConda < BgWorkerObjConda & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjConda(varargin)
      obj@BgTrackWorkerObj(varargin{:});
      %obj.nviews = nviews;
      %obj.track_type = track_type ;
    end    
  end
end
