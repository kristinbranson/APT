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
    
    function f = getTrainCompleteArtifacts(obj)
      % f: [nmodel x 1] cellstr of artifacts whose presence indicates train
      % complete
      f = obj.dmcs.trainFinalModelLnx()';
    end

    function sRes = initComputeResults(obj)
      nmodels = obj.dmcs.n;
      %trainCompleteFiles = obj.getTrainCompleteArtifacts();
      sRes = struct;
      sRes.identifiers = obj.dmcs.getIdentifiers();
      sRes.pollsuccess = false; % if true, remote poll cmd was successful
      sRes.pollts = now; % datenum time that poll cmd returned
      sRes.jsonPath = obj.dmcs.trainDataLnx; % cell of chars, full paths to json trn loss being polled
      sRes.jsonPresent = false(1,nmodels); % array, true if corresponding file exists. if false, remaining fields are indeterminate
      sRes.lastTrnIter = nan(1,nmodels); % array, (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
      sRes.tfUpdate = false(1,nmodels); % (only if jsonPresent==true) array, true if the current read represents an updated training iter.
      sRes.contents = cell(1,nmodels); % (only if jsonPresent==true) array, if tfupdate is true, this can contain all json contents.
      sRes.trainCompletePath = obj.dmcs.trainCompleteArtifacts(); % cell of cell of char, full paths to artifact indicating train complete
      sRes.trainFinalModel = obj.dmcs.trainFinalModelLnx();
      sRes.tfComplete = false(1,nmodels); % array, true if trainCompletePath exists
      sRes.errFile = obj.dmcs.errfileLnx; % cell of char, full path to DL err file, should be only one
      sRes.errFileExists = false(1,nmodels); % array, true of errFile exists and has size>0
      sRes.logFile = obj.dmcs.trainLogLnx; % cell of char, full path to Bsub logfile
      sRes.logFileExists = false(1,nmodels); % array logical
      sRes.logFileErrLikely = false(1,nmodels); % array, true if Bsub logfile suggests error
      sRes.killFile = obj.getKillFiles(); % char, full path to KILL tokfile
      sRes.killFileExists = false(1,nmodels); % scalar, true if KILL tokfile found
    end

    function sRes = readTrainLoss(obj,sRes,imodel,jsoncurr)
      try
        trnLog = jsondecode(jsoncurr);
      catch ME
        warning('Failed to read json file for training model %d progress update:\n%s',imodel,getReport(ME));
        if numel(obj.trnLogPrev) >= imodel && ~isempty(obj.trnLogPrev{imodel}),
          trnLog = obj.trnLogPrev{imodel};
        else
          sRes.jsonPresent(imodel) = false;
          return;
        end
      end
      newStep = trnLog.step(end);
      if numel(obj.trnLogLastStep) >= imodel,
        lastKnownStep = obj.trnLogLastStep(imodel);
      else
        lastKnownStep = -1;
      end
      tfupdate = newStep>lastKnownStep;
      sRes.tfUpdate(imodel) = tfupdate;
      if tfupdate
        sRes.lastTrnIter(imodel) = newStep;
        obj.trnLogLastStep(imodel) = newStep;
      else
        sRes.lastTrnIter(imodel) = lastKnownStep;
      end
      sRes.contents{imodel} = trnLog;

    end
    
    function sRes = compute(obj) % obj const except for .trnLogLastStep
      % sRes: [nviewx1] struct array.
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      nmodels = obj.dmcs.n;
      sRes = obj.initComputeResults();

      sRes.jsonPresent = cellfun(@obj.fileExists,sRes.jsonPath);
      for i=1:nmodels,
        sRes.tfComplete(i) = cellfun(@obj.fileExists,sRes.trainCompletePath{i});
      end
      [unique_jobs,idx1,jobidx] = unique(sRes.identifiers.jobidx);
      % one error, log, and kill file per job
      errFile = sRes.errFile(idx1);
      logFile = sRes.logFile(idx1);
      killFile = sRes.killFile(idx1);
      for ijob = 1:numel(unique_jobs),
        sRes.errFileExists(jobidx==ijob) = obj.errFileExistsNonZeroSize(errFile{ijob});
        sRes.logFileExists(jobidx==ijob) = obj.errFileExistsNonZeroSize(logFile{ijob}); % ahem good meth name
        sRes.logFileErrLikely(jobidx==ijob) = obj.logFileErrLikely(logFile{ijob});
        sRes.killFileExists(jobidx==ijob) = obj.fileExists(killFile{ijob});
      end

      % loop through all models trained in this job
      for i = 1:nmodels,
        if sRes.jsonPresent(i),
          [jsoncurr] = obj.fileContents(sRes.jsonPath{i});
          sRes = obj.readTrainLoss(sRes,i,jsoncurr);
        end
      end
      sRes.pollsuccess = true;
    end
    
    function reset(obj) 
      % clears/inits .trnLogLastStep, the only mutatable prop
      if isempty(obj.dmcs),
        obj.trnLogLastStep = [];
        return;
      end
      obj.trnLogLastStep = repmat(-1,[1,obj.dmcs.n]);
      
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
      logFiles = unique(obj.dmcs.trainLogLnx)';
    end
    
    function errFile = getErrFile(obj)
      errFile = unique(obj.dmcs.errfileLnx)';
%       assert(isscalar(errFile));
%       errFile = errFile{1};
    end
 
    function killFiles = getKillFiles(obj)
      killFiles = obj.dmcs.killTokenLnx;
    end
    
  end
  
end