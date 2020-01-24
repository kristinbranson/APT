classdef BgTrackWorkerObjConda < BgWorkerObjConda & BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjConda(varargin)
      obj@BgTrackWorkerObj(varargin{:});
    end    
  end
end