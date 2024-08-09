classdef BgTrainSplitWorkerObjBsub < BgTrainWorkerObjBsub
    
  methods
    
    function obj = BgTrainSplitWorkerObjBsub(varargin)
      obj@BgTrainWorkerObjBsub(varargin{:});
    end
    
    function f = getTrainCompleteArtifacts(obj)
      % f: [nview x 1] cellstr of artifacts whose presence indicates train
      % complete
      %
      % overloaded to look for val/test results
      f = obj.dmcs.valresultsLnx(:);
    end

  end
  
end
