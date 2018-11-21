classdef BgTrainMonitorAWS < BgTrainMonitor
  
  properties
    awsEc2 % scalar handle AWSec2
  end
  
  methods
    
    function obj = BgTrainMonitorAWS
      obj@BgTrainMonitor();
    end
    
    function prepareHook(obj,trnMonVizObj,bgWorkerObj)
      obj.awsEc2 = bgWorkerObj.awsEc2;
    end
       
  end
  
end