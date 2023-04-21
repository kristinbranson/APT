classdef BgTrackWorkerObjConda < BgWorkerObjConda & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjConda(nviews,varargin)
      obj@BgTrackWorkerObj(varargin{:});
      obj.nviews = nviews;
    end    
  end
end