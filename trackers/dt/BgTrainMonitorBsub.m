classdef BgTrainMonitorBsub < BgTrainMonitor
  
  
  methods
    
    function obj = BgTrainMonitorBsub()
      obj@BgTrainMonitor();
    end
    
    function prepareHook(obj,trnMonVizObj,bgWorkerObj)
      % none
    end
    
  end
  
end
