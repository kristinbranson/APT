classdef BgTrainMonitorAWS < BgTrainMonitor
  
  properties
    awsEc2 % scalar handle AWSec2
    remotePID % view1 remote PID
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