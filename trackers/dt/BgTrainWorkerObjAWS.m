classdef BgTrainWorkerObjAWS < BgTrainWorkerObj
  
  properties
    awsEc2 % Instance of AWSec2
  end
  
  methods

    function obj = BgTrainWorkerObjAWS(dlLblFile,jobID,cacheRemoteRel,...
        logfilesremote,awsec2)
      % jobID used only for errfile
      
      obj@BgTrainWorkerObj(dlLblFile,jobID);
      
      obj.artfctLogs = logfilesremote;
      obj.artfctKills = BgTrainWorkerObj.killedFilesFromLogFiles(logfilesremote);
      [obj.artfctTrainDataJson,obj.artfctFinalIndex,obj.artfctErrFile] = ...
        arrayfun(@(ivw)obj.trainMonitorArtifacts(cacheRemoteRel,ivw),...
        1:obj.nviews,'uni',0);
            
      obj.awsEc2 = awsec2;
    end
        
    function [json,finalindex,errfile] = trainMonitorArtifacts(obj,...
        cacheRemoteRel,ivw)
      
      projvw = sprintf('%s_view%d',obj.projname,ivw-1); % !! cacheDirs are 0-BASED
      subdir = fullfile('/home/ubuntu',cacheRemoteRel);
      
%       json = sprintf('%s_pose_unet_traindata.json',projvw);
      json = 'traindata.json'; % AL AWS testing 20180826: what happens with multiview?
      json = fullfile(subdir,json);
      json = FSPath.standardPathChar(json);
      
      finaliter = obj.sPrm.dl_steps;
      finalindex = sprintf('%s_pose_unet-%d.index',projvw,finaliter);
      finalindex = fullfile(subdir,finalindex);
      finalindex = FSPath.standardPathChar(finalindex);
      
      errfile = DeepTracker.dlerrGetErrFile(obj.jobID,'/home/ubuntu');
      errfile = FSPath.standardPathChar(errfile);
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
      for ivw=1:obj.nviews
        json = obj.artfctTrainDataJson{ivw};
        finalindex = obj.artfctFinalIndex{ivw};
        errFile = obj.artfctErrFile{ivw};
        logFile = obj.artfctLogs{ivw};
        killFile = obj.artfctKills{ivw};
        
        fspollargs = ...
          sprintf('exists %s exists %s existsNE %s existsNEerr %s exists %s contents %s',...
            json,finalindex,errFile,logFile,killFile,json);
        cmdremote = sprintf('~/APT/misc/fspoll.py %s',fspollargs);

        [tfpollsucc,res] = aws.cmdInstance(cmdremote);
        if tfpollsucc
          reslines = regexp(res,'\n','split');
          tfpollsucc = iscell(reslines) && numel(reslines)==6;
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
    
    function tf = fileExists(obj,f)
      tf = obj.awsEc2.remoteFileExists(f,'dispcmd',true);
    end
    
    function tf = errFileExistsNonZeroSize(obj,errFile)
      tf = obj.awsEc2.remoteFileExists(errFile,'reqnonempty',true,'dispcmd',true);
    end    
    
    function s = fileContents(obj,f)
      s = obj.awsEc2.remoteFileContents(f,'dispcmd',true);
    end
        
  end
    
end