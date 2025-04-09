classdef BgTrainPoller < BgPoller
  
  properties
    dmcs_  % [nview] DeepModelChainOnDisk array  
    backend_  % A scalar DlBackEndClass, owned by someone else
    trnLogLastStep_  % [nview] int. most recent last step from training json logs
  end  % properties
  
  methods
    function obj = BgTrainPoller(dmc, backend)
      obj.dmcs_ = dmc ;
      obj.backend_ = backend ;
    end

    function result = poll(obj, logger) 
      % Create a logger if none provided
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ; 
      end
            
      % It's lame that we need to handle the AWS backend case differently than the
      % non-AWS backend case.  The backend is supposed to hide these differences from us!
      % But for now that's just how it is.  Would be nice to fix in the future.  
      % -- ALT, 2025-01-16
      if obj.backend_.type == DLBackEnd.AWS ,
        result = obj.pollAWS_(logger) ;
      else
        result = obj.pollNonAWS_(logger) ;
      end
      assert(isstruct(result)&&isscalar(result)) ;
    end  % function

    function result = pollNonAWS_(obj, logger)  %#ok<INUSD>
      % sRes: [nviewx1] struct array.
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      result = obj.initialPollResults_() ;

      try
        result.jsonPresent = cellfun(@(fileName)(obj.backend_.fileExists(fileName)), result.jsonPath);
        nModels = obj.dmcs_.n;
        doesOutputFileExist = false(1,nModels);
        for i=1:nModels,
          doesOutputFileExist(i) = cellfun(@(fileName)(obj.backend_.fileExists(fileName)), result.trainCompletePath{i});
        end
        [unique_jobs,idx1,jobidx] = unique(result.identifiers.jobidx);
        unique_job_count = numel(unique_jobs) ;
        % one error, log, and kill file per job
        errFilePaths = result.errFile(idx1);  % cell array of paths
        logFilePaths = result.logFile(idx1);  % cell array of paths
        for ijob = 1:unique_job_count ,
          result.errFileExists(jobidx==ijob) = obj.backend_.fileExistsAndIsNonempty(errFilePaths{ijob});
          result.logFileExists(jobidx==ijob) = obj.backend_.fileExistsAndIsNonempty(logFilePaths{ijob}); % ahem good meth name
        end
          
        result.isRunning = row(obj.backend_.isAliveFromRegisteredJobIndex('train')) ;
        result.tfComplete = doesOutputFileExist & ~result.isRunning ;

        % loop through all models trained in this job
        for i = 1:nModels,
          if result.jsonPresent(i),
            jsoncurr = obj.backend_.fileContents(result.jsonPath{i});
            result = obj.readTrainLoss_(result,i,jsoncurr);
          end
        end
        pollsuccess = true ;
      catch me
        % Presumably something went wrong when probing the filesystem
        pollsuccess = false ;
      end
      result.pollsuccess = pollsuccess ;
    end  % function
    
    function result = pollAWS_(obj, logger)
      % sRes: nview x 1 struct array.
            
      result = obj.initialPollResults_() ;

      fspollargs = {};
      
      % do this for jobs
      [unique_jobs,idx1,jobidx] = unique(result.identifiers.jobidx);
      unique_job_count = numel(unique_jobs) ;
      % one error, log, and kill file per job
      errFilePaths = result.errFile(idx1);  % cell array of paths
      logFilePaths = result.logFile(idx1);  % cell array of paths
      %killFile = sRes.killFile(idx1);
      for i = 1:unique_job_count,
        % fspollargs = [fspollargs,{'existsNE',errFile{i},'existsNE',logFile{i},...
        %   'existsNEerr',logFile{i},'exists',killFile{i}}]; %#ok<AGROW> 
        fspollargsForThisJob = {'existsNE',errFilePaths{i}, ...
                                'existsNE',logFilePaths{i}} ;
        fspollargs = horzcat(fspollargs, fspollargsForThisJob) ;  %#ok<AGROW>
      end
      nlinesperjob = 2 ;  % needs to match the number of things done per job above
      nModels = obj.dmcs_.n ; 
      for i = 1:nModels,
        fspollargsForThisModel = {'exists',result.jsonPath{i}, ...
                                  'exists',result.trainFinalModel{i}, ...
                                  'contents',result.jsonPath{i}} ;
        fspollargs = horzcat(fspollargs, fspollargsForThisModel) ; %#ok<AGROW> 
      end
      nlinespermodel = 3 ;  % needs to match the number of things done per model above
      [tfpollsucc, reslines] = obj.backend_.batchPoll(fspollargs);
      logger.log('obj.backend.batchPoll(fspollargs) tfpollsucc: %d\n',tfpollsucc) ;
      logger.log('obj.backend.batchPoll(fspollargs) reslines:\n%s\n',newline_out(reslines)) ;
      if tfpollsucc
        for i = 1:unique_job_count,
          off = (i-1)*nlinesperjob;
          result.errFileExists(jobidx==i) = strcmp(reslines{off+1},'y');
          result.logFileExists(jobidx==i) = strcmp(reslines{off+2},'y');
          %sRes.logFileErrLikely(jobidx==i) = strcmp(reslines{off+3},'y');
          %sRes.killFileExists(jobidx==i) = strcmp(reslines{off+4},'y');
        end
        doesOutputFileExist = false(1,nModels);
        for i = 1:nModels,
          off = unique_job_count*nlinesperjob+(i-1)*nlinespermodel;
          result.jsonPresent(i) = strcmp(reslines{off+1},'y');
          doesOutputFileExist = strcmp(reslines{off+2},'y');          
          if result.jsonPresent(i),
            result = obj.readTrainLoss_(result,i,reslines{off+3});
          end
        end
        try
          result.isRunning = row(obj.backend_.isAliveFromRegisteredJobIndex('train')) ;
          result.tfComplete = doesOutputFileExist & ~result.isRunning ;
          result.pollsuccess = true ;
        catch me
          result.pollsuccess = false ;
        end 

      end  % if tfpollsucc
    end  % function

    function result = readTrainLoss_(obj,result,imodel,jsoncurr)
      try
        trnLog = jsondecode(jsoncurr);
      catch ME
        warning('Failed to read json file for training model %d progress update:\n%s',imodel,getReport(ME));
        result.jsonPresent(imodel) = false;
        return
      end
      newStep = trnLog.step(end);
      if numel(obj.trnLogLastStep_) >= imodel,
        lastKnownStep = obj.trnLogLastStep_(imodel);
      else
        lastKnownStep = -1;
      end
      tfupdate = newStep>lastKnownStep;
      result.tfUpdate(imodel) = tfupdate;
      if tfupdate
        result.lastTrnIter(imodel) = newStep;
        obj.trnLogLastStep_(imodel) = newStep;
      else
        result.lastTrnIter(imodel) = lastKnownStep;
      end
      result.contents{imodel} = trnLog;
    end  % function

    function result = initialPollResults_(obj)
      nModels = obj.dmcs_.n;
      result = struct();
      result.identifiers = obj.dmcs_.getIdentifiers();
      result.pollsuccess = false;  % if true, remote poll cmd was successful
      result.pollts = now() ; % datenum time that poll cmd returned
      result.jsonPath = obj.dmcs_.trainDataLnx; % cell of chars, full paths to json trn loss being polled
      result.jsonPresent = false(1,nModels); % array, true if corresponding file exists. if false, remaining fields are indeterminate
      result.lastTrnIter = nan(1,nModels); % array, (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
      result.tfUpdate = false(1,nModels); % (only if jsonPresent==true) array, true if the current read represents an updated training iter.
      result.contents = cell(1,nModels); % (only if jsonPresent==true) array, if tfupdate is true, this can contain all json contents.
      result.trainCompletePath = obj.dmcs_.trainCompleteArtifacts(); % cell of cell of char, full paths to artifact indicating train complete
      result.trainFinalModel = obj.dmcs_.trainFinalModelLnx();
      result.tfComplete = false(1,nModels); % array, true if trainCompletePath exists and training job is not running anymore
      result.errFile = obj.dmcs_.errfileLnx; % cell of char, full path to DL err file
      result.errFileExists = false(1,nModels); % array, true of errFile exists and has size>0
      result.logFile = obj.dmcs_.trainLogLnx; % cell of char, full path to Bsub logfile
      result.logFileExists = false(1,nModels); % array logical
      result.isRunning = false(1,nModels) ;
      result.isPopulated = true(1,nModels) ;
    end  % function

    function suitcase = packParfevalSuitcase(obj)
      suitcase = obj.backend_.packParfevalSuitcase() ;
    end  % function
   
    function restoreAfterParfeval(obj, suitcase)
      obj.backend_.restoreAfterParfeval(suitcase) ;
    end  % function
    
  end  % methods
    
end  % classdef