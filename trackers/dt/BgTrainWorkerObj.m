classdef BgTrainWorkerObj < handle
  % Object deep copied onto BG Training worker. To be used with
  % BGWorkerContinuous
  % 
  % Responsibilities:
  % - Poll filesystem for training updates

  % TODO: this obj should own or have a copy of the DeepModelChainOnDisk so
  % it can do things like list the current modelChanDir.
  
  properties
    nviews
    
    artfctLogs % [nview] cellstr of fullpaths to logfiles
    artfctKills % [nview] cellstr of fullpaths to KILLED toks
    artfctTrainDataJson % [nview] cellstr of fullpaths to training data jsons
    artfctFinalIndex % [nview] cellstr of fullpaths to final training .index file
    artfctErrFile % [nview] cellstr of fullpaths to DL errfile
    
    trnLogLastStep; % [nview] int. most recent last step from training json logs
  end
  
  methods (Abstract)
    tf = fileExists(obj,file)
    tf = errFileExistsNonZeroSize(obj,errFile)
    s = fileContents(obj,file)
  end
  
  methods
    
    function obj = BgTrainWorkerObj(nviews,dmcs)
      obj.nviews = nviews;
      assert(isa(dmcs,'DeepModelChainOnDisk') && numel(dmcs)==nviews);
      
      obj.artfctLogs = {dmcs.trainLogLnx}'; 
      obj.artfctKills = {dmcs.killTokenLnx}'; 
      obj.artfctTrainDataJson = {dmcs.trainDataLnx}';
      obj.artfctFinalIndex = {dmcs.trainFinalIndexLnx}'; 
      obj.artfctErrFile = {dmcs.errfileLnx}';
    
      obj.reset();
    end
    
    function sRes = compute(obj) % obj const except for .trnLogLastStep
      % sRes: [nviewx1] struct array.
            
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
        );
      for ivw=1:obj.nviews
        json = obj.artfctTrainDataJson{ivw};
        finalindex = obj.artfctFinalIndex{ivw};
        errFile = obj.artfctErrFile{ivw};
        logFile = obj.artfctLogs{ivw};
        killFile = obj.artfctKills{ivw};
        
        sRes(ivw).pollsuccess = true;
        sRes(ivw).pollts = now;
        sRes(ivw).jsonPath = json;
        sRes(ivw).jsonPresent = obj.fileExists(json);
        sRes(ivw).trainCompletePath = finalindex;
        sRes(ivw).trainComplete = obj.fileExists(finalindex);
        sRes(ivw).errFile = errFile;
        sRes(ivw).errFileExists = obj.errFileExistsNonZeroSize(errFile);
        sRes(ivw).logFile = logFile;
        sRes(ivw).logFileErrLikely = obj.logFileErrLikely(logFile);
        sRes(ivw).killFile = killFile;
        sRes(ivw).killFileExists = obj.fileExists(killFile);        
        
        if sRes(ivw).jsonPresent
          json = obj.fileContents(json);
          trnLog = jsondecode(json);
          lastKnownStep = obj.trnLogLastStep(ivw);
          newStep = trnLog.step(end);
          tfupdate = newStep>lastKnownStep;
          sRes(ivw).tfUpdate = tfupdate;
          if tfupdate
            sRes(ivw).lastTrnIter = newStep;
            obj.trnLogLastStep(ivw) = newStep;
          else
            sRes(ivw).lastTrnIter = lastKnownStep;
          end
          sRes(ivw).contents = trnLog;
        end
      end
    end
    
    function reset(obj) 
      % clears/inits .trnLogLastStep, the only mutatable prop
      obj.trnLogLastStep = repmat(-1,1,obj.nviews);
    end
   
    function printLogfiles(obj) % obj const
      logFiles = obj.artfctLogs;
      logFileContents = cellfun(@(x)obj.fileContents(x),logFiles,'uni',0);
      BgTrainWorkerObj.printLogfilesStc(logFiles,logFileContents)
    end
            
    function [tfEFE,errFile] = errFileExists(obj) % obj const
      errFile = unique(obj.artfctErrFile);
      assert(isscalar(errFile));
      errFile = errFile{1};
      tfEFE = obj.errFileExistsNonZeroSize(errFile);
    end
    
    function tfLogErrLikely = logFileErrLikely(obj,file) % obj const
      tfLogErrLikely = obj.fileExists(file);
      if tfLogErrLikely
        logContents = obj.fileContents(file);
        tfLogErrLikely = ~isempty(regexpi(logContents,'exception','once'));
      end
    end

  end
  
  methods (Static)
    
    function printLogfilesStc(logFiles,logFileContents)
      % Print training logs for all views for current/last retrain 
      
      for ivw=1:numel(logFiles)
        logfile = logFiles{ivw};
        fprintf(1,'\n### View %d:\n### %s\n\n',ivw,logfile);
        disp(logFileContents{ivw});
      end
    end
    
  end
  
end