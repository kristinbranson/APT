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
      
      dmcs = obj.bgWorkerObj.dmcs;
      killfiles = {dmcs.killTokenLnx};
      
      bkillcmd = sprintf('bkill %d',obj.jobID);
      bkillcmd = DeepTracker.codeGenSSHGeneral(bkillcmd,'bg',false);
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Bkill command failed: %s',res);
        return;
      end

      % expect command to fail; fail -> py proc killed
      fcn = makeBsubJobKilledPollFcn(obj.jobID);
      iterWaitTime = 1;
      maxWaitTime = 12;
      tfsucc = waitforPoll(fcn,iterWaitTime,maxWaitTime);
      
      if ~tfsucc
        warningNoTrace('Could not confirm that bsub job was killed.');
      else
        % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to 
        % pick up
        for i=1:numel(killfiles)
          kfile = killfiles{i};
          touchcmd = sprintf('touch %s',kfile);
          touchcmd = DeepTracker.codeGenSSHGeneral(touchcmd,'bg',false);
          [st,res] = system(touchcmd);
          if st~=0
            warningNoTrace('Failed to create KILLED token: %s',kfile);
          else
            fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge the kill!\n',kfile);
          end
        end
        
        % bgTrnMonitor should pick up KILL tokens and stop bg trn monitoring
      end
    end    
  end
  
end

function fcn = makeBsubJobKilledPollFcn(jobID)

pollcmd = sprintf('bjobs -o stat -noheader %d',jobID);
pollcmd = DeepTracker.codeGenSSHGeneral(pollcmd,'bg',false);
 
fcn = @lcl;

  function tf = lcl
    % returns true when jobID is killed
    %disp(pollcmd);
    [st,res] = system(pollcmd);
    if st==0
      tf = isempty(regexp(res,'RUN','once'));      
    else
      tf = false;
    end
  end
end