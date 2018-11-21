classdef BgTrainWorkerObjAWS < BgTrainWorkerObj
  
  properties
    awsEc2 % Instance of AWSec2
  end
  
  methods

    function obj = BgTrainWorkerObjAWS(nviews,dmcs,awsec2)
      obj@BgTrainWorkerObj(nviews,dmcs);                  
      obj.awsEc2 = awsec2;
    end
            
    function sRes = compute(obj)
      % sRes: [nviewx1] struct array.
            
      aws = obj.awsEc2;
      
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      sRes = struct(...
        'pollsuccess',cell(obj.nviews,1),... % if true, remote poll cmd was successful
        'pollts',[],... % datenum time that poll cmd returned
        'jsonPath',[],... % char, full path to json trnlog being polled
        'jsonPresent',[],... % true if file exists. if false, remaining fields are indeterminate
        'lastTrnIter',[],... % (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',[],... % (only if jsonPresent==true) true if the current read represents an updated training iter.
        'contents',[],... % (only if jsonPresent==true) if tfupdate is true, this can contain all json contents.
        'trainCompletePath',[],... % char, full path to artifact indicating train complete
        'trainComplete',[],... % true if trainCompletePath exists
        'errFile',[],... % char, full path to DL err file
        'errFileExists',[],... % true of errFile exists and has size>0
        'logFile',[],... % char, full path to Bsub logfile
        'logFileErrLikely',[],... % true if Bsub logfile suggests error
        'killFile',[],... % char, full path to KILL tokfile
        'killFileExists',[]... % true if KILL tokfile found
        ); % 
      dmcs = obj.dmcs;
      for ivw=1:obj.nviews
        
        dmc = dmcs(ivw);
        json = dmc.trainDataLnx;
        finalindex = dmc.trainFinalIndexLnx;
        errFile = dmc.errfileLnx;
        logFile = dmc.trainLogLnx;
        killFile = dmc.killTokenLnx;
        
%         json = obj.artfctTrainDataJson{ivw};
%         finalindex = obj.artfctFinalIndex{ivw};
%         errFile = obj.artfctErrFile{ivw};
%         logFile = obj.artfctLogs{ivw};
%         killFile = obj.artfctKills{ivw};
        
        fspollargs = ...
          sprintf('exists %s exists %s existsNE %s existsNEerr %s exists %s contents %s',...
            json,finalindex,errFile,logFile,killFile,json);
        cmdremote = sprintf('~/APT/misc/fspoll.py %s',fspollargs);

        [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
        if tfpollsucc
          reslines = regexp(res,'\n','split');
          tfpollsucc = iscell(reslines) && numel(reslines)==6+1; % last cell is {0x0 char}
        end
          
        sRes(ivw).pollsuccess = tfpollsucc;
        sRes(ivw).pollts = now;
        sRes(ivw).jsonPath = json;
        sRes(ivw).trainCompletePath = finalindex;
        sRes(ivw).errFile = errFile;
        sRes(ivw).logFile = logFile;
        sRes(ivw).killFile = killFile;        
        
        if tfpollsucc          
          sRes(ivw).jsonPresent = strcmp(reslines{1},'y');
          sRes(ivw).trainComplete = strcmp(reslines{2},'y');
          sRes(ivw).errFileExists = strcmp(reslines{3},'y');
          sRes(ivw).logFileErrLikely = strcmp(reslines{4},'y');
          sRes(ivw).killFileExists = strcmp(reslines{5},'y');
          
          if sRes(ivw).jsonPresent
            trnLog = jsondecode(reslines{6});
            lastKnownStep = obj.trnLogLastStep(ivw);
            newStep = trnLog.step(end);
            tfupdate = newStep>lastKnownStep;
            sRes(ivw).tfUpdate = tfupdate;
            if tfupdate
              sRes(ivw).lastTrnIter = newStep;
              obj.trnLogLastStep(ivw) = newStep;
              sRes(ivw).contents = trnLog;
            else
              sRes(ivw).lastTrnIter = lastKnownStep;
            end
          end
        end
      end
    end
    
    % Since compute() is overloaded in this subclass, these Abstract 
    % methods are not needed for BgTrainWorkerObj/compute. However they are
    % called elsewhere.
    
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
    
    function killProcess(obj)
%       if ~obj.isRunning
%         error('Training is not in progress.');
%       end
      aws = obj.awsEc2;
      if isempty(aws)
        error('AWSEC2 backend object is unset.');
      end
      
      dmc = obj.dmcs;
      assert(isscalar(dmc)); % single-view atm
      killfile = dmc.killTokenLnx;

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
        cmd = sprintf('touch %s',killfile);
        tfsucc = aws.cmdInstance(cmd,'dispcmd',false);
        if ~tfsucc
          warningNoTrace('Failed to create remote KILLED token: %s',killfile);
        else
          fprintf('Created remote KILLED token: %s. Please wait for your training monitor to acknowledge the kill!\n',killfile);
        end
        % bgTrnMonitorAWS should pick up KILL tokens and stop bg trn monitoring
      end
    end
    
  end
    
end