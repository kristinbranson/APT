classdef BgTrackMonitor < handle  % BGTrainMonitor
  %
  % A BgTrackMonitor:
  % 1. Is a BGClient/BGWorker pair comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of a track. For
  %     debugging/testing this can be done from the client machine.
  % 2. In general knows how to communicate with the bg tracking process.
  % 
  %
  % BgTrackMonitor is intended to be subclassed.
  %
  % BgTrackMonitor does NOT know how to spawn tracking jobs.
  %
  % See also prepare() method comments for related info.
  
  properties
    bgContCallInterval = 20; %secs
    
    bgClientObj
    bgWorkerObj
    
    cbkTrkComplete % fcnhandle with sig cbk(res), called when track complete
  end
  properties (Dependent)
    prepared
    isRunning
  end
  methods
    function v = get.prepared(obj)
      v = ~isempty(obj.bgClientObj);
    end
    function v = get.isRunning(obj)
      bgc = obj.bgClientObj;
      v = ~isempty(bgc) && bgc.isRunning;
    end
  end
  
  methods    
    
    function obj = BgTrackMonitor
      obj.reset();
    end
    
    function delete(obj)
      obj.reset();
    end
    
    function reset(obj)
      if ~isempty(obj.bgClientObj)
        delete(obj.bgClientObj);
      end
      obj.bgClientObj = [];
      
      if ~isempty(obj.bgWorkerObj)
        delete(obj.bgWorkerObj)
      end
      obj.bgWorkerObj = [];
      
      obj.cbkTrkComplete = [];
    end
    
    function prepare(obj,bgWorkerObj,cbkComplete)
      % prepare(obj,mIdx,nview,movfiles,outfiles,bsublogfiles)
      
      obj.reset();

      fprintf(1,'Configuring tracking background worker...\n');
      
      bgc = BGClient;
      cbkResult = @obj.bgTrkResultsReceived;
      bgc.configure(cbkResult,bgWorkerObj,'compute');
 
      obj.bgTrkMonitorClient = bgc;
      obj.bgTrkMonitorWorkerObj = bgWorkerObj;
      obj.cbkTrkComplete = cbkComplete;
    end
    
    function start(obj)
      assert(obj.prepared);
      bgc = obj.bgClientObj;
      bgc.startWorker('workerContinuous',true,...
        'continuousCallInterval',obj.bgContCallInterval);
    end    
    
    function bgTrkResultsReceived(obj,sRes)
      res = sRes.result;
      
      errOccurred = any([res.errFileExists]);
      if errOccurred
        obj.stop();
        
        fprintf(1,'Error occurred during tracking:\n');
        errFile = res(1).errFile; % currently, errFiles same for all views
        fprintf(1,'\n### %s\n\n',errFile);
        errContents = obj.bgWorkerObj.fileContents(errFile);
        disp(errContents);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        
        % bgTrkReset not called
      end
      
      for i=1:numel(res)
        if res(i).logFileErrLikely
          obj.stop();
          
          fprintf(1,'Error occurred during tracking:\n');
          errFile = res(i).logFile;
          fprintf(1,'\n### %s\n\n',errFile);
          errContents = obj.bgWorkerObj.fileContents(errFile);
          disp(errContents);
          fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
          
          % bgTrkReset not called
        end
      end
      
      tfdone = all([res.tfcomplete]);
      if tfdone
        fprintf(1,'Tracking output files detected:\n');
        arrayfun(@(x)fprintf(1,'  %s\n',x.trkfile),res);        
        obj.stop();
        obj.cbkTrkComplete(res);
      end
    end
    
    function stop(obj)
      obj.bgClientObj.stopWorker();
      % don't clear trkSysInfo for now     
    end
    
  end
end