classdef BgTrainMonitorAWS < BgTrainMonitor
  
  properties
    awsEc2 % scalar handle AWSec2
    remotePID % view1 remote PID. Currently unused, awsEc2 handles it
  end
  
  methods
    
    function obj = BgTrainMonitorAWS
      obj@BgTrainMonitor();
    end
    
    function prepareHook(obj,trnMonVizObj,bgWorkerObj)
      obj.awsEc2 = bgWorkerObj.awsEc2;
    end
    
    function killProcess(obj)
      if ~obj.isRunning
        error('Training is not in progress.');
      end
      aws = obj.awsEc2;
      if isempty(aws)
        error('AWSEC2 backend object is unset.');
      end
      
      dmc = obj.bgWorkerObj.dmcs;
      assert(isscalar(dmc)); % single-view atm
      killfiles = {dmc.killTokenLnx};

      aws.killRemoteProcess();

      % expect command to fail; fail -> py proc killed
      pollCbk = @()~aws.cmdInstance('pgrep python','dispcmd',true,'failbehavior','silent');
      iterWaitTime = 1;
      maxWaitTime = 10;
      tfsucc = waitforPoll(pollCbk,iterWaitTime,maxWaitTime);
      
      if ~tfsucc
        warningNoTrace('Could not confirm that remote process was killed.');
      else
        % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to 
        % pick up
        for i=1:numel(killfiles)
          kfile = killfiles{i};
          cmd = sprintf('touch %s',kfile);
          tfsucc = aws.cmdInstance(cmd,'dispcmd',false); 
          if ~tfsucc
            warningNoTrace('Failed to create remote KILLED token: %s',kfile);
          else
            fprintf('Created remote KILLED token: %s. Please wait for your training monitor to acknowledge the kill!\n',kfile);
          end
        end
        
        % bgTrnMonitorAWS should pick up KILL tokens and stop bg trn monitoring
      end
    end
       
  end
  
end