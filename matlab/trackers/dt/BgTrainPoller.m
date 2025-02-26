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

    function sRes = poll(obj, logger) 
      % Create a logger if none provided
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ; 
      end
            
      % It's lame that we need to handle the AWS backend case differently than the
      % non-AWS backend case.  The backend is supposed to hide these differences from us!
      % But for now that's just how it is.  Would be nice to fix in the future.  
      % -- ALT, 2025-01-16
      if obj.backend_.type == DLBackEnd.AWS ,
        sRes = obj.pollAWS_(logger) ;
      else
        sRes = obj.pollNonAWS_(logger) ;
      end
    end  % function

    function sRes = pollNonAWS_(obj, logger)  %#ok<INUSD>
      % sRes: [nviewx1] struct array.
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      sRes = obj.initialPollResults_() ;

      sRes.jsonPresent = cellfun(@(fileName)(obj.backend_.fileExists(fileName)), sRes.jsonPath);
      nModels = obj.dmcs_.n;
      for i=1:nModels,
        sRes.tfComplete(i) = cellfun(@(fileName)(obj.backend_.fileExists(fileName)), sRes.trainCompletePath{i});
      end
      [unique_jobs,idx1,jobidx] = unique(sRes.identifiers.jobidx);
      % one error, log, and kill file per job
      errFile = sRes.errFile(idx1);
      logFile = sRes.logFile(idx1);
      %killFile = sRes.killFile(idx1);
      for ijob = 1:numel(unique_jobs),
        sRes.errFileExists(jobidx==ijob) = obj.backend_.fileExistsAndIsNonempty(errFile{ijob});
        sRes.logFileExists(jobidx==ijob) = obj.backend_.fileExistsAndIsNonempty(logFile{ijob}); % ahem good meth name
        %sRes.logFileErrLikely(jobidx==ijob) = obj.logFileErrLikely(logFile{ijob});
        %sRes.killFileExists(jobidx==ijob) = obj.fileExists(killFile{ijob});
      end

      % loop through all models trained in this job
      for i = 1:nModels,
        if sRes.jsonPresent(i),
          jsoncurr = obj.backend_.fileContents(sRes.jsonPath{i});
          sRes = obj.readTrainLoss_(sRes,i,jsoncurr);
        end
      end
      sRes.pollsuccess = true;
    end  % function
    
    function sRes = pollAWS_(obj, logger)
      % sRes: nview x 1 struct array.
            
      sRes = obj.initialPollResults_() ;

      fspollargs = {};
      
      % do this for jobs
      [unique_jobs,idx1,jobidx] = unique(sRes.identifiers.jobidx);
      nJobs = numel(unique_jobs);
      % one error, log, and kill file per job
      errFile = sRes.errFile(idx1);
      logFile = sRes.logFile(idx1);
      %killFile = sRes.killFile(idx1);
      for i = 1:nJobs,
        % fspollargs = [fspollargs,{'existsNE',errFile{i},'existsNE',logFile{i},...
        %   'existsNEerr',logFile{i},'exists',killFile{i}}]; %#ok<AGROW> 
        fspollargsForThisJob = {'existsNE',errFile{i}, ...
                                'existsNE',logFile{i}} ;
        fspollargs = horzcat(fspollargs, fspollargsForThisJob) ;  %#ok<AGROW>
      end
      nlinesperjob = 2 ;  % needs to match the number of things done per job above
      nModels = obj.dmcs_.n ; 
      for i = 1:nModels,
        fspollargsForThisModel = {'exists',sRes.jsonPath{i}, ...
                                  'exists',sRes.trainFinalModel{i}, ...
                                  'contents',sRes.jsonPath{i}} ;
        fspollargs = horzcat(fspollargs, fspollargsForThisModel) ; %#ok<AGROW> 
      end
      nlinespermodel = 3 ;  % needs to match the number of things done per model above
      [tfpollsucc, reslines] = obj.backend_.batchPoll(fspollargs);
      logger.log('obj.backend.batchPoll(fspollargs) tfpollsucc: %d\n',tfpollsucc) ;
      logger.log('obj.backend.batchPoll(fspollargs) reslines:\n%s\n',newline_out(reslines)) ;
      if tfpollsucc
        sRes.pollsuccess = true ;
        for i = 1:nJobs,
          off = (i-1)*nlinesperjob;
          sRes.errFileExists(jobidx==i) = strcmp(reslines{off+1},'y');
          sRes.logFileExists(jobidx==i) = strcmp(reslines{off+2},'y');
          %sRes.logFileErrLikely(jobidx==i) = strcmp(reslines{off+3},'y');
          %sRes.killFileExists(jobidx==i) = strcmp(reslines{off+4},'y');
        end
        for i = 1:nModels,
          off = nJobs*nlinesperjob+(i-1)*nlinespermodel;
          sRes.jsonPresent(i) = strcmp(reslines{off+1},'y');
          sRes.tfComplete(i) = strcmp(reslines{off+2},'y');          
          if sRes.jsonPresent(i),
            sRes = obj.readTrainLoss_(sRes,i,reslines{off+3});
          end
        end
      end
    end  % function

    function sRes = readTrainLoss_(obj,sRes,imodel,jsoncurr)
      try
        trnLog = jsondecode(jsoncurr);
      catch ME
        warning('Failed to read json file for training model %d progress update:\n%s',imodel,getReport(ME));
        sRes.jsonPresent(imodel) = false;
        return
      end
      newStep = trnLog.step(end);
      if numel(obj.trnLogLastStep_) >= imodel,
        lastKnownStep = obj.trnLogLastStep_(imodel);
      else
        lastKnownStep = -1;
      end
      tfupdate = newStep>lastKnownStep;
      sRes.tfUpdate(imodel) = tfupdate;
      if tfupdate
        sRes.lastTrnIter(imodel) = newStep;
        obj.trnLogLastStep_(imodel) = newStep;
      else
        sRes.lastTrnIter(imodel) = lastKnownStep;
      end
      sRes.contents{imodel} = trnLog;
    end  % function

    function sRes = initialPollResults_(obj)
      nModels = obj.dmcs_.n;
      sRes = struct();
      sRes.identifiers = obj.dmcs_.getIdentifiers();
      sRes.pollsuccess = false; % if true, remote poll cmd was successful
      sRes.pollts = now; % datenum time that poll cmd returned
      sRes.jsonPath = obj.dmcs_.trainDataLnx; % cell of chars, full paths to json trn loss being polled
      sRes.jsonPresent = false(1,nModels); % array, true if corresponding file exists. if false, remaining fields are indeterminate
      sRes.lastTrnIter = nan(1,nModels); % array, (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
      sRes.tfUpdate = false(1,nModels); % (only if jsonPresent==true) array, true if the current read represents an updated training iter.
      sRes.contents = cell(1,nModels); % (only if jsonPresent==true) array, if tfupdate is true, this can contain all json contents.
      sRes.trainCompletePath = obj.dmcs_.trainCompleteArtifacts(); % cell of cell of char, full paths to artifact indicating train complete
      sRes.trainFinalModel = obj.dmcs_.trainFinalModelLnx();
      sRes.tfComplete = false(1,nModels); % array, true if trainCompletePath exists
      sRes.errFile = obj.dmcs_.errfileLnx; % cell of char, full path to DL err file, should be only one
      sRes.errFileExists = false(1,nModels); % array, true of errFile exists and has size>0
      sRes.logFile = obj.dmcs_.trainLogLnx; % cell of char, full path to Bsub logfile
      sRes.logFileExists = false(1,nModels); % array logical
      %sRes.logFileErrLikely = false(1,nModels); % array, true if Bsub logfile suggests error
      %sRes.killFile = obj.getKillFiles(); % char, full path to KILL tokfile
      %sRes.killFileExists = false(1,nModels); % scalar, true if KILL tokfile found
    end  % function

    function suitcase = packParfevalSuitcase(obj)
      suitcase = obj.backend_.packParfevalSuitcase() ;
    end  % function
   
    function restoreAfterParfeval(obj, suitcase)
      obj.backend_.restoreAfterParfeval(suitcase) ;
    end  % function
    
  end  % methods
    
end  % classdef