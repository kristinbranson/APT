classdef BgTrainWorkerObj < handle
  % Object deep copied onto BG Training worker. To be used with
  % BGWorkerContinuous
  % 
  % Responsibilities:
  % - Poll filesystem for training updates

  properties
    nviews
    sPrm % parameter struct
    projname
    jobID % char
    
    artfctLogs % [nview] cellstr of fullpaths to bsub logs
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
    function obj = BgTrainWorkerObj(dlLblFile,jobID)
      lbl = load(dlLblFile,'-mat');
      obj.nviews = lbl.cfg.NumViews;
      obj.sPrm = lbl.trackerDeepData.sPrm; % .sPrm guaranteed to match dlLblFile
      obj.projname = lbl.projname;
      obj.jobID = jobID;

      obj.trnLogLastStep = repmat(-1,1,obj.nviews);

      % Concrete subclasses responsible for initting artfct* props      
    end
    
    function sRes = compute(obj)
      % sRes: [nviewx1] struct array.
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      sRes = struct(...
        'jsonPath',cell(obj.nviews,1),... % char, full path to json trnlog being polled
        'jsonPresent',[],... % true if file exists. if false, remaining fields are indeterminate
        'lastTrnIter',[],... % (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',[],... % (only if jsonPresent==true) true if the current read represents an updated training iter.
        'contents',[],... % (only if jsonPresent==true) if tfupdate is true, this can contain all json contents.
        'trainCompletePath',[],... % char, full path to artifact indicating train complete
        'trainComplete',[],... % true if trainCompletePath exists
        'errFile',[],... % char, full path to DL err file
        'errFileExists',[],... % true of errFile exists and has size>0
        'logFile',[],... % char, full path to Bsub logfile
        'logFileErrLikely',[]... % true if Bsub logfile suggests error
        ); % 
      for ivw=1:obj.nviews
        json = obj.artfctTrainDataJson{ivw};
        finalindex = obj.artfctFinalIndex{ivw};
        errFile = obj.artfctErrFile{ivw};
        logFile = obj.artfctLogs{ivw};
        
        sRes(ivw).jsonPath = json;
        sRes(ivw).jsonPresent = obj.fileExists(json);
        sRes(ivw).trainCompletePath = finalindex;
        sRes(ivw).trainComplete = obj.fileExists(finalindex);
        sRes(ivw).errFile = errFile;
        sRes(ivw).errFileExists = obj.errFileExistsNonZeroSize(errFile);
        sRes(ivw).logFile = logFile;
        sRes(ivw).logFileErrLikely = obj.logFileErrLikely(logFile);
        
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
            sRes(ivw).contents = trnLog;
          else
            sRes(ivw).lastTrnIter = lastKnownStep;
          end
        end
      end
    end
   
    function printLogfiles(obj)
      logFiles = obj.artfctLogs;
      logFileContents = cellfun(@(x)obj.fileContents(x),logFiles,'uni',0);
      BgTrainWorkerObj.printLogfilesStc(logFiles,logFileContents)
    end
    
    function [tfEFE,errFile] = errFileExists(obj)
      errFile = unique(obj.artfctErrFile);
      assert(isscalar(errFile));
      errFile = errFile{1};
      tfEFE = obj.errFileExistsNonZeroSize(errFile);
    end
    
    function tfLogErrLikely = logFileErrLikely(obj,file)
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