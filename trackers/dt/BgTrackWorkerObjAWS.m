classdef BgTrackWorkerObjAWS < BgWorkerObjAWS & BgTrackWorkerObj 
  methods    
    function obj = BgTrackWorkerObjAWS(varargin)
      obj@BgWorkerObjAWS(varargin{:});
    end    
  end
end