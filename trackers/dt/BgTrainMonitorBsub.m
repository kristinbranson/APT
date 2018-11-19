classdef BgTrainMonitorBsub < BgTrainMonitor
  
  properties
    jobID % bsub jobID
  end
  
  methods
    
    function obj = BgTrainMonitorBsub()
      obj@BgTrainMonitor();
    end
    
    function prepareHook(obj,trnMonVizObj,bgWorkerObj)
      % none
    end
    
    function killProcess(obj)
      if ~obj.isRunning
        error('Training is not in progress.');
      end
      if isempty(obj.jobID) || isnan(obj.jobID)
         error('Bsub jobID is unset.');
      end
      
      killfiles = obj.bgWorkerObj.artfctKills;
      
      bkillcmd = sprintf('bkill %d',obj.jobID);
      bkillcmd = DeepTracker.codeGenSSHGeneral(bkillcmd);
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Bkill command failed: %s',res);
        return;
      end

      
      
      
      % xxx stopped 
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