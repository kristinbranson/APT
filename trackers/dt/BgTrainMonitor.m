classdef BgTrainMonitor < handle
  % BGTrainMonitor
  %
  % A BGTrainMonitor is:
  % 1. A BGClient/BGWorker pair comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of a train. For
  %     debugging/testing this can be done from the client machine.
  % 2. A client-side TrainingMonitorViz object that visualizes training 
  % progress sent back from the BGWorker
  % 3. Custom actions performed when training is complete
  %
  % BGTrainMonitor is intended to be subclassed.
  %
  % BGTrainMonitor does NOT know how to spawn training jobs but will know
  % how to (attempt to) kill them. For debugging, you can manually spawn 
  % jobs and monitor them with BgTrainMonitor.
  %
  % See also prepare() method comments for related info.
  
  properties
    bgContCallInterval = 30; %secs
    
    bgClientObj
    bgWorkerObj % scalar "detached" object that is deep-copied onto
    % workers. Note, this is not the BGWorker obj itself
    trnMonitorObj % object with resultsreceived() method
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
    
    function obj = BgTrainMonitor
      obj.reset();
    end
    
    function delete(obj)
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
    
    function prepare(obj,trnMonVizObj,bgWorkerObj,varargin)
      % bgWorkerObj knows how to poll the state of the train. 
      % trnMonVizObj knows how to vizualize this state. 
      % bgTrnResultsReceivedHook performs custom actions after receiving
      % an update from bgWorkerObj. 
      %
      % bgWorkerObj/trnMonVizObj should be mix+matchable as bgWorkerObj 
      % should send a core set of 'standard' metrics that trnMonVizObj can
      % use.
      %
      % bgWorkerObj matches 1-1 with the concrete BgTrainMonitor and its 
      % bgTrnResultsReceivedHook method. These work in concert and the 
      % custom actions taken by bgTrnResultsReceivedHook depends on custom 
      % info supplied by bgWorkerObj.
      
      obj.reset();
      
      [tfEFE,errFile] = bgWorkerObj.errFileExists;
      if tfEFE
        error('Error file ''%s'' exists.',errFile);
      end
      
      cbkResult = @obj.bgTrnResultsReceived;

      bgc = BGClient;
      fprintf(1,'Configuring background worker...\n');
      bgc.configure(cbkResult,bgWorkerObj,'compute');
      
      obj.bgClientObj = bgc;
      obj.bgWorkerObj = bgWorkerObj;
      obj.trnMonitorObj = trnMonVizObj;
      
      obj.prepareHook(trnMonVizObj,bgWorkerObj);
    end
    
    function start(obj)
      assert(obj.prepared);
      bgc = obj.bgClientObj;
      bgc.startWorker('workerContinuous',true,...
        'continuousCallInterval',obj.bgContCallInterval);
    end
    
    function bgTrnResultsReceived(obj,sRes)
      obj.trnMonitorObj.resultsReceived(sRes);
      obj.bgTrnResultsReceivedHook(sRes);
    end
    
    function bgTrnResultsReceivedHook(obj,sRes)
      errOccurred = any([sRes.result.errFileExists]);
      if errOccurred
        obj.stop();
        
        fprintf(1,'Error occurred during training:\n');
        errFile = sRes.result(1).errFile; % currently, errFiles same for all views
        fprintf(1,'\n### %s\n\n',errFile);
        errContents = obj.bgWorkerObj.fileContents(errFile);
        disp(errContents);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        
        % monitor plot stays up; reset not called etc
      end
      
      for i=1:numel(sRes.result)
        if sRes.result(i).logFileErrLikely
          obj.stop();
          
          fprintf(1,'Error occurred during training:\n');
          errFile = sRes.result(i).logFile;
          fprintf(1,'\n### %s\n\n',errFile);
          errContents = obj.bgWorkerObj.fileContents(errFile);
          disp(errContents);
          fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
          
          % monitor plot stays up; bgTrnReset not called etc
        end
      end
      
      trnComplete = all([sRes.result.trainComplete]);
      if trnComplete
        obj.stop();
        % % monitor plot stays up; reset not called etc
        fprintf('Training complete at %s.\n',datestr(now));
        
        % May need a custom hook here
      end
    end
    
    function stop(obj)
      obj.bgClientObj.stopWorker();
    end
    
  end
  
  methods (Abstract)
    prepareHook(obj,trnMonVizObj,bgWorkerObj)
  end
end