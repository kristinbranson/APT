classdef BgTrainWorkerObjAWS < BgWorkerObjAWS & BgTrainWorkerObj
  
  properties
  end
  
  methods

    function obj = BgTrainWorkerObjAWS(varargin)
      obj@BgWorkerObjAWS(varargin{:});
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
        'tfComplete',[],...
        'lastTrnIter',[],... % (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',[],... % (only if jsonPresent==true) true if the current read represents an updated training iter.
        'contents',[],... % (only if jsonPresent==true) if tfupdate is true, this can contain all json contents.
        'trainCompletePath',[],... % char, full path to artifact indicating train complete
        'trainComplete',[],... % true if trainCompletePath exists
        'errFile',[],... % char, full path to DL err file
        'errFileExists',[],... % true of errFile exists and has size>0
        'logFile',[],... % char, full path to Bsub logfile
        'logFileExists',[],... % logical
        'logFileErrLikely',[],... % true if Bsub logfile suggests error
        'killFile',[],... % char, full path to KILL tokfile
        'killFileExists',[]... % true if KILL tokfile found
        ); % 
      dmcs = obj.dmcs;
      for ivw=1:obj.nviews
        
        % Multiview: errFile and logFile for ivw>1 currently never exist.
        % which is fine for now
        dmc = dmcs(ivw);
        json = dmc.trainDataLnx;
        finalmdl = dmc.trainFinalModelLnx;
        errFile = dmc.errfileLnx;
        logFile = dmc.trainLogLnx;
        killFile = dmc.killTokenLnx;
                
        % See AWSEC2 convenience meth
        fspollargs = ...
          sprintf('exists %s exists %s existsNE %s existsNE %s existsNEerr %s exists %s contents %s',...
            json,finalmdl,errFile,logFile,logFile,killFile,json);
        cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollargs);

        [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
        if tfpollsucc
          reslines = regexp(res,'\n','split');
          tfpollsucc = iscell(reslines) && numel(reslines)==7+1; % last cell is {0x0 char}
        end
          
        sRes(ivw).pollsuccess = tfpollsucc;
        sRes(ivw).pollts = now;
        sRes(ivw).jsonPath = json;
        sRes(ivw).trainCompletePath = finalmdl;
        sRes(ivw).errFile = errFile;
        sRes(ivw).logFile = logFile;
        sRes(ivw).killFile = killFile;        
        
        if tfpollsucc          
          sRes(ivw).jsonPresent = strcmp(reslines{1},'y');
          sRes(ivw).tfComplete = strcmp(reslines{2},'y');
          sRes(ivw).errFileExists = strcmp(reslines{3},'y');
          sRes(ivw).logFileExists = strcmp(reslines{4},'y');
          sRes(ivw).logFileErrLikely = strcmp(reslines{5},'y');
          sRes(ivw).killFileExists = strcmp(reslines{6},'y');
          
          if sRes(ivw).jsonPresent
            trnLog = jsondecode(reslines{7});
            lastKnownStep = obj.trnLogLastStep(ivw);
            newStep = trnLog.step(end);
            tfupdate = newStep>lastKnownStep;
            sRes(ivw).tfUpdate = tfupdate;
            sRes(ivw).contents = trnLog;
            if tfupdate
              sRes(ivw).lastTrnIter = newStep;
              obj.trnLogLastStep(ivw) = newStep;
            else
              sRes(ivw).lastTrnIter = lastKnownStep;
            end
          end
        else
          % Still not bulletproof here. Consider initting sRes fully above
          % before computation
          sRes(ivw).jsonPresent = false;
          sRes(ivw).tfComplete = false;
          sRes(ivw).errFileExists = false;
          sRes(ivw).logFileExists = false;
          sRes(ivw).logFileErrLikely = false;
          sRes(ivw).killFileExists = false;
        end
      end
    end

  end
    
end