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
       
    function killRemoteProcess(obj)
      if isempty(obj.remotePID)
        error('Unknown PID for remote process.');
      end
      
      cmdremote = sprintf('kill %d',obj.remotePID);
      [tfsucc,res] = obj.awsEc2.cmdInstance(cmdremote,'dispcmd',true);
      if tfsucc
        fprintf('Kill command sent.\n\n');
      else
        error('Kill command failed.');
      end

    end
    
  end
  
end