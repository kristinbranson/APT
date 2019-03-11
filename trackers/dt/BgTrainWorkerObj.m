classdef BgTrainWorkerObj < BgWorkerObj
  % Object deep copied onto BG Training worker. To be used with
  % BGWorkerContinuous
  % 
  % Responsibilities:
  % - Poll filesystem for training updates
  % - Be able to read/parse the current state of the train/model on disk
  
  properties
    trnLogLastStep; % [nview] int. most recent last step from training json logs
    trnLogPrev = [];
  end
  
  methods
    
    function obj = BgTrainWorkerObj(varargin)
      obj@BgWorkerObj(varargin{:});
    end
    
    
    function sRes = compute(obj) % obj const except for .trnLogLastStep
      % sRes: [nviewx1] struct array.
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      dmcs = obj.dmcs;
      njobs = numel(dmcs);
      sRes = struct(...
        'pollsuccess',cell(njobs,1),... % if true, remote poll cmd was successful
        'pollts',[],... % datenum time that poll cmd returned
        'jsonPath',[],... % char, full path to json trnlog being polled
        'jsonPresent',[],... % true if file exists. if false, remaining fields are indeterminate
        'lastTrnIter',[],... % (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',[],... % (only if jsonPresent==true) true if the current read represents an updated training iter.
        'contents',[],... % (only if jsonPresent==true) if tfupdate is true, this can contain all json contents.
        'trainCompletePath',[],... % char, full path to artifact indicating train complete
        'tfComplete',[],... % true if trainCompletePath exists
        'errFile',[],... % char, full path to DL err file
        'errFileExists',[],... % true of errFile exists and has size>0
        'logFile',[],... % char, full path to Bsub logfile
        'logFileErrLikely',[],... % true if Bsub logfile suggests error
        'killFile',[],... % char, full path to KILL tokfile
        'killFileExists',[]... % true if KILL tokfile found
        );
      killFiles = obj.getKillFiles();
      for ivw=1:njobs,
        dmc = dmcs(ivw);
        json = dmc.trainDataLnx;
        finalindex = dmc.trainFinalIndexLnx;
        errFile = dmc.errfileLnx;
        logFile = dmc.trainLogLnx;
        killFile = killFiles{ivw};
        
        sRes(ivw).pollsuccess = true;
        sRes(ivw).pollts = now;
        sRes(ivw).jsonPath = json;
        sRes(ivw).jsonPresent = obj.fileExists(json);
        sRes(ivw).trainCompletePath = finalindex;
        sRes(ivw).tfComplete = obj.fileExists(finalindex);
        sRes(ivw).errFile = errFile;
        sRes(ivw).errFileExists = obj.errFileExistsNonZeroSize(errFile);
        sRes(ivw).logFile = logFile;
        sRes(ivw).logFileErrLikely = obj.logFileErrLikely(logFile);
        sRes(ivw).killFile = killFile;
        sRes(ivw).killFileExists = obj.fileExists(killFile);        
        
        if sRes(ivw).jsonPresent
          try
            json = obj.fileContents(json);
            trnLog = jsondecode(json);
          catch ME
            warning('Failed to read json file for training progress update:\n%s',getReport(ME));
            if ~isempty(obj.trnLogPrev),
              trnLog = obj.trnLogPrev;
            else
              sRes(ivw).jsonPresent = false;
              continue;
            end
          end
          newStep = trnLog.step(end);
          lastKnownStep = obj.trnLogLastStep(ivw);
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
      if ~isempty(obj.nviews),
        obj.trnLogLastStep = repmat(-1,1,obj.nviews);
      end

      if isempty(obj.dmcs),
        return;
      end
      
      killFiles = obj.getKillFiles();
      for i = 1:numel(killFiles),
        if exist(killFiles{i},'file'),
          delete(killFiles{i});
        end
      end
      
      logFiles = obj.getLogFiles();
      for i = 1:numel(logFiles),
        if exist(logFiles{i},'file'),
          delete(logFiles{i});
        end
      end

      errFiles = obj.getErrFile();
      for i = 1:numel(errFiles),
        if exist(errFiles{i},'file'),
          delete(errFiles{i});
        end
      end
      
    end
    
    function logFiles = getLogFiles(obj)
      logFiles = unique({obj.dmcs.trainLogLnx}');
    end
    
    function errFile = getErrFile(obj)
      errFile = unique({obj.dmcs.errfileLnx}');
%       assert(isscalar(errFile));
%       errFile = errFile{1};
    end

  end
  
end