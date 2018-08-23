classdef BgTrainMonitorAWS < handle
  % a BGTrainMonitor is
  % 1. A BGClient/BGWorker pair comprising an async pipeline betwn the
  % client and a bg worker
  % 2. A Training Monitor object that visualizes training progress sent
  % from BGWorker
  % 3. A set of actions performed when training is complete
  
  properties
    bgClientObj
    bgWorkerObj % scalar "detached" object that is deep-copied onto
    % workers. Note, this is not the BGWorker obj itself
    trnMonitorObj % object with resultsreceived() method
  end
  properties (Dependent)
    bgTrnReady
  end
  methods
    function v = get.bgTrnReady(obj)
      v = ~isempty(obj.bgClientObj);
    end
  end
  
  methods
    
    function obj = BgTrainMonitorAWS
      obj.reset();
    end
    
    function reset(obj)
      % Reset BG Train Monitor state
      %
      % - TODO Note, when you change eg params, u need to call this. etc etc.
      % Any mutation that alters PP, train/track on the BG worker...
      
      if ~isempty(obj.bgClientObj)
        delete(obj.bgClientObj);
      end
      obj.bgClientObj = [];
      
      if ~isempty(obj.bgWorkerObj)
        delete(obj.bgWorkerObj)
      end
      obj.bgWorkerObj = [];
      
      if ~isempty(obj.trnMonitorObj)
        delete(obj.trnMonitorObj);
      end
      obj.trnMonitorObj = [];
    end
    
    function prepare(obj,nview,dlLblFile,jobID,awsEc2,varargin)
      cmdlineMonitor = myparse(varargin,...
        'cmdlineMonitor',false);
      
      obj.reset();
      
      %       errFile = DeepTracker.dlerrGetErrFile(jobID);
      %       assert(exist(errFile,'file')==0,'Error file ''%s'' exists.',errFile);
      if cmdlineMonitor
        objMon = DeepTrackerTrainingMonitorCmdline;
      else
        objMon = DeepTrackerTrainingMonitor(nview);
      end
      cbkResult = @obj.bgTrnResultsReceived;
      workerObj = BGWorkerObjAws(dlLblFile,jobID,awsEc2);

      bgc = BGClient;
      fprintf(1,'Configuring background worker...\n');
      bgc.configure(cbkResult,workerObj,'compute');
      
      obj.bgClientObj = bgc;
      obj.bgWorkerObj = workerObj;
      obj.trnMonitorObj = objMon;
    end
    
    function start(obj)
      assert(obj.bgTrnReady);
      obj.bgClientObj.startWorker('workerContinuous',true,...
        'continuousCallInterval',30);
    end
    
    function bgTrnResultsReceived(obj,sRes)
      obj.trnMonitorObj.resultsReceived(sRes);
      
      errOccurred = any([sRes.result.errFileExists]);
      if errOccurred
        obj.stop();
        
        fprintf(1,'Error occurred during training:\n');
        errFile = sRes.result(1).errFile; % currently, errFiles same for all views
        fprintf(1,'\n### %s\n\n',errFile);
        type(errFile);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        
        % monitor plot stays up; reset not called etc
      end
      
      trnComplete = all([sRes.result.trainComplete]);
      if trnComplete
        obj.stop();
        % % monitor plot stays up; reset not called etc
        fprintf('Training complete at %s.\n',datestr(now));
        fprintf('COPY/DO STUFF!!\n');
      end
    end
    
    function stop(obj)
      obj.bgClientObj.stopWorker();
    end
    
  end
end