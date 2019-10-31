classdef BgWorkerObjAWS < BgWorkerObj
  
  properties
    awsEc2 % Instance of AWSec2
  end
  
  methods

    function obj = BgWorkerObjAWS(nviews,dmcs,awsec2,varargin)
      obj@BgWorkerObj(nviews,dmcs);
      obj.awsEc2 = awsec2;
    end
    
    function tf = fileExists(obj,f)
      tf = obj.awsEc2.remoteFileExists(f,'dispcmd',true);
    end
    
    function tf = errFileExistsNonZeroSize(obj,errFile)
      tf = obj.awsEc2.remoteFileExists(errFile,'reqnonempty',true,'dispcmd',true);
    end    
    
    function s = fileContents(obj,f)
      s = obj.awsEc2.remoteFileContents(f,'dispcmd',true);
    end
    
    function dispModelChainDir(obj)
      aws = obj.awsEc2;
      for ivw=1:obj.nviews
        dmc = obj.dmcs(ivw);
        cmd = sprintf('ls -al %s',dmc.dirModelChainLnx);
        fprintf('### View %d:\n',ivw);
        [tfsucc,res] = aws.cmdInstance(cmd,'dispcmd',false); 
        if tfsucc
          disp(res);
        else
          warningNoTrace('Failed to access training directory %s: %s',...
            dmc.dirModelChainLnx,res);
        end
        fprintf('\n');
      end
    end
    
    function [tfsucc,warnings] = killProcess(obj)
      warnings = {};
%       if ~obj.isRunning
%         error('Training is not in progress.');
%       end
      aws = obj.awsEc2;
      if isempty(aws)
        error('AWSEC2 backend object is unset.');
      end
      
      if ~aws.canKillRemoteProcess()
        tfpid = aws.getRemotePythonPID();
        if ~tfpid
          error('Could not ascertain remote process ID in AWSEC2 instance %s.',aws.instanceID);
        end
      end
      
      killfile = obj.getKillFiles();
      killfile = unique(killfile);
      assert(isscalar(killfile)); % for now
      killfile = killfile{1};

      aws.killRemoteProcess();

      % expect command to fail; fail -> py proc killed
      pollCbk = @()~aws.cmdInstance('pgrep python','dispcmd',true,'failbehavior','silent');
      iterWaitTime = 1;
      maxWaitTime = 10;
      tfsucc = waitforPoll(pollCbk,iterWaitTime,maxWaitTime);
      
      if ~tfsucc
        warningNoTrace('Could not confirm that remote process was killed.');
        warnings{end+1} = 'Could not confirm that remote process was killed.';
      else
        % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to 
        % pick up
        cmd = sprintf('touch ''%s''',killfile); % use single-quotes; cmdInstance will use double-quotes
        tfsucc = aws.cmdInstance(cmd,'dispcmd',false);
        if ~tfsucc
          warningNoTrace('Failed to create remote KILLED token: %s',killfile);
          warnings{end+1} = sprintf('Failed to create remote KILLED token: %s',killfile);
        else
          fprintf('Created remote KILLED token: %s. Please wait for your monitor to acknowledge the kill!\n',killfile);
        end
        % bgTrnMonitorAWS should pick up KILL tokens and stop bg trn monitoring
      end
    end
    
  end
    
end