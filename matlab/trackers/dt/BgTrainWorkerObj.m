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
    
    function sRes = compute(obj) % obj const except for .trnLogLastStep
      % sRes: [nviewx1] struct array.
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      nmodels = obj.dmcs.n;
      %trainCompleteFiles = obj.getTrainCompleteArtifacts();
      sRes = struct(...
        'identifiers',obj.dmcs.getIdentifiers(),...
        'pollsuccess',false,... % if true, remote poll cmd was successful
        'pollts',now,... % datenum time that poll cmd returned
        'jsonPath',obj.dmcs.trainDataLnx,... % cell of chars, full paths to json trn loss being polled
        'jsonPresent',false(1,nmodels),... % array, true if corresponding file exists. if false, remaining fields are indeterminate
        'lastTrnIter',nan(1,nmodels),... % array, (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',false(1,nmodels),... % (only if jsonPresent==true) array, true if the current read represents an updated training iter.
        'contents',cell(1,nmodels),... % (only if jsonPresent==true) array, if tfupdate is true, this can contain all json contents.
        'trainCompletePath',obj.dmcs.trainCompleteArtifacts(),... % cell of cell of char, full paths to artifact indicating train complete
        'tfComplete',false(1,nmodels),... % array, true if trainCompletePath exists
        'errFile',obj.dmcs.errfileLnx,... % cell of char, full path to DL err file, should be only one
        'errFileExists',false(1,nmodels),... % array, true of errFile exists and has size>0
        'logFile',obj.dmcs.trainLogLnx,... % cell of char, full path to Bsub logfile
        'logFileExists',false(1,nmodels),... % array logical 
        'logFileErrLikely',false(1,nmodels),... % array, true if Bsub logfile suggests error
        'killFile',obj.getKillFiles(),... % char, full path to KILL tokfile
        'killFileExists',false(1,nmodels)... % scalar, true if KILL tokfile found
        );
      sRes(iijob).jsonPresent = cellfun(@obj.fileExists,sRes.jsonPath);
      for i=1:nmodels,
        sRes.tfComplete(i) = all(cellfun(@obj.fileExists,sRes.trainCompleteFiles{i}));
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
          try
            jsoncurr = obj.fileContents(sRes.jsonPath{i});
            trnLog = jsondecode(jsoncurr);
          catch ME
            warning('Failed to read json file for training model %d progress update:\n%s',i,getReport(ME));
            if numel(obj.trnLogPrev) >= i && ~isempty(obj.trnLogPrev{i}),
              trnLog = obj.trnLogPrev{i};
            else
              sRes.jsonPresent(i) = false;
              continue;
            end
          end
          newStep = trnLog.step(end);
          if numel(obj.trnLogLastStep) >= i,
            lastKnownStep = obj.trnLogLastStep(i);
          else
            lastKnownStep = -1;
          end
          tfupdate = newStep>lastKnownStep;
          sRes.tfUpdate(i) = tfupdate;
          if tfupdate
            sRes.lastTrnIter(i) = newStep;
            obj.trnLogLastStep(i) = newStep;
          else
            sRes.lastTrnIter(i) = lastKnownStep;
          end
          sRes.contents{i} = trnLog;
        end
      end
      sRes.pollsuccess = true;
    end
    
    function reset(obj) 
      % clears/inits .trnLogLastStep, the only mutatable prop
      if isempty(obj.dmcs),
        obj.trnLogLastStep = {};
        return;
      end
      jobs = obj.dmcs.getJobs();
      [unique_jobs,~,jobidx] = unique(jobs);
      njobs = numel(unique_jobs);
      obj.trnLogLastStep = cell(1,njobs);
      for i = 1:njobs,
        obj.trnLogLastStep{i} = repmat(-1,1,nnz(jobidx==i));
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