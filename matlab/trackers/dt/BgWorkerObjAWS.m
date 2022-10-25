classdef BgWorkerObjAWS < BgWorkerObj & matlab.mixin.Copyable
  
  properties
    awsEc2 % Instance of AWSec2
  end
  
  methods (Access=protected)    
    function obj2 = copyElement(obj)
      % overload so that .awsec2 is deep-copied
      obj2 = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.dmcs)
        obj2.dmcs = copy(obj.dmcs);
      end
      if ~isempty(obj.awsEc2)
        obj2.awsEc2 = copy(obj.awsEc2);
      end
    end
  end
  methods

    function obj = BgWorkerObjAWS(nviews,dmcs,awsec2,varargin)
      obj@BgWorkerObj(nviews,dmcs);
      obj.awsEc2 = awsec2;
    end
    
    function obj2 = copyAndDetach(obj)
      % See note in BGClient/configure(). We create a new obj2 here that is
      % a deep-copy made palatable for parfeval
      
      obj2 = copy(obj); % deep-copies obj, including .awsec2 and .dmcs if appropriate

      dmcs = obj.dmcs;
      if ~isempty(dmcs)
        dmcs.prepareBg();
      end

      aws = obj.awsEc2;
      if ~isempty(aws)
        aws.clearStatusFuns();
      end      
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

    function tfsucc = lsdir(obj,dir)
      tfsucc = obj.awsEc2.remoteLs(dir);
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
      %pollCbk = @()~aws.cmdInstance('pgrep -o python','dispcmd',true,'failbehavior','silent');
      pollCbk = @()aws.getNoPyProcRunning();
      iterWaitTime = 1;
      maxWaitTime = 20;
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