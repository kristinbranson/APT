classdef BgTrainWorkerObjAWS < BgWorkerObjAWS & BgTrainWorkerObj
  
  properties
  end
  
  methods

    function obj = BgTrainWorkerObjAWS(varargin)
      obj@BgWorkerObjAWS(varargin{:});
    end
            
    function sRes = oldcompute(obj) % obj const except for .trnLogLastStep
      % sRes: [nviewx1] struct array.
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      nmodels = obj.dmcs.n;
      sRes = obj.initComputeResults();

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
          [jsoncurr] = obj.fileContents(sRes.jsonPath{i});
          sRes = obj.readTrainLoss(sRes,i,jsoncurr);
        end
      end
      sRes.pollsuccess = true;
    end

    function sRes = compute(obj)
      % sRes: [nviewx1] struct array.
            
      aws = obj.awsEc2;
      nlinesperjob = 4;
      nlinespermodel = 3;

      sRes = obj.initComputeResults();
      nmodels = numel(sRes.jsonPath);
      
      fspollargs = {};
      
      % do this for jobs
%       sRes(iijob).jsonPresent = cellfun(@obj.fileExists,sRes.jsonPath);
%       for i=1:nmodels,
%         sRes.tfComplete(i) = all(cellfun(@obj.fileExists,sRes.trainCompleteFiles{i}));
%       end
      [unique_jobs,idx1,jobidx] = unique(sRes.identifiers.jobidx);
      njobs = numel(unique_jobs);
      % one error, log, and kill file per job
      errFile = sRes.errFile(idx1);
      logFile = sRes.logFile(idx1);
      killFile = sRes.killFile(idx1);
      for i = 1:njobs,
        fspollargs = [fspollargs,{'existsNE',errFile{i},'existsNE',logFile{i},...
          'existsNEerr',logFile{i},'exists',killFile{i}}]; %#ok<AGROW> 
      end

      for i = 1:nmodels,

        % See AWSEC2 convenience meth
        fspollargs = ...
          [fspollargs,{'exists',sRes.jsonPath{i},'exists',sRes.trainFinalModel{i},'contents',sRes.jsonPath{i}}]; %#ok<AGROW> 

      end
      [tfsucc,res] = aws.remoteCallFSPoll(fspollargs);
      if tfpollsucc
        reslines = regexp(res,'\n','split');
        sRes.tfpollsucc = iscell(reslines) && numel(reslines)==nlinesperjob*njobs+nlinespermodel*nmodels+1; % last cell is {0x0 char}
        for i = 1:njobs,
          off = (i-1)*njobs;
          sRes.errFileExists(jobidx==i) = strcmp(reslines{off+1},'y');
          sRes.logFileExists(jobidx==i) = strcmp(reslines{off+2},'y');
          sRes.logFileErrLikely(jobidx==i) = strcmp(reslines{off+3},'y');
          sRes.killFileExists(jobidx==i) = strcmp(reslines{off+4},'y');
        end
        for i = 1:nmodels,
          off = njobs*nlinesperjob+(i-1)*nlinespermodel;
          sRes.jsonPresent(i) = strcmp(reslines{off+1},'y');
          sRes.tfComplete(i) = strcmp(reslines{off+2},'y');          
          if sRes.jsonPresent(i),
            sRes = obj.readTrainLoss(sRes,i,reslines{off+3});
          end
        end
      end
    end

  end
    
end