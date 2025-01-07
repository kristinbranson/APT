classdef BgTrainWorkerObjAWS < BgWorkerObjAWS & BgTrainWorkerObj
  
  properties
  end
  
  methods

    function obj = BgTrainWorkerObjAWS(varargin)
      obj@BgWorkerObjAWS(varargin{:});
    end
            
    function sRes = work(obj, logger)
      % sRes: [nviewx1] struct array.
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end
            
      nlinesperjob = 4;
      nlinespermodel = 3;

      sRes = obj.initComputeResults();
      nmodels = numel(sRes.jsonPath);
      
      fspollargs = {};
      
      % do this for jobs
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
        fspollargs = ...
          [fspollargs,{'exists',sRes.jsonPath{i},'exists',sRes.trainFinalModel{i},'contents',sRes.jsonPath{i}}]; %#ok<AGROW> 
      end
      [tfpollsucc,reslines] = obj.backend.batchPoll(fspollargs);
      logger.log('obj.backend.batchPoll(fspollargs) tfpollsucc: %d\n',tfpollsucc) ;
      logger.log('obj.backend.batchPoll(fspollargs) reslines:\n%s\n',newline_out(reslines)) ;
      if tfpollsucc
        sRes.pollsuccess = true ;
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
    end  % function

  end  % methods
    
end  % classdef