classdef BgMonitor < handle
  % BGMonitor
  %
  % A BGMonitor is:
  % 1. A BgClient/BgRunner pair comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of the process. For
  %     debugging/testing this can be done from the client machine.
  % 2. A client-side MonitorViz object that visualizes 
  % progress sent back from the BgRunner
  % 3. Custom actions performed when process is complete
  %
  % BGMonitor does NOT know how to spawn process jobs but will know
  % how to (attempt to) kill them. For debugging, you can manually spawn 
  % jobs and monitor them with BgMonitor.
  %
  % BGMonitor does NOT know how to probe the detailed state of the
  % process eg on disk. That is BgRunnerObj's domain.
  %
  % So BGMonitor is a connector/manager obj that runs the worker 
  % (knows how to poll the filesystem in detail) in the background and 
  % connects it with a Monitor.
  %
  % See also prepare() method comments for related info.
  
  properties
    bgContCallInterval  % scalar double, in secs    
    bgClientObj  % the BgClient (the BgRunner is created in the .startRunner() method of the BgClient)
    bgWorkerObj  % scalar "detached" object (not sure this is still true about it being 
                 % detached --ALT, 2024-06-28) that is deep-copied onto
                 % workers. Note, this is not the BgRunner obj itself.
    monitorObj  % object with resultsreceived() method, typically a "monitor visualizer"
    cbkComplete  % fcnhandle with sig cbk(res), called when operation complete
    processName
  end

  properties (Dependent)
    prepared
    isRunning
  end
    
  events
    bgStart
    bgEnd    
  end
  
  methods
    function obj = BgMonitor(type_string, monVizObj, bgWorkerObj, cbkComplete, varargin)      
      % Is obj for monitoring training or tracking?
      if strcmp(type_string, 'train') ,
        obj.processName = 'train' ;
        obj.bgContCallInterval = 30 ;  % secs
      elseif strcmp(type_string, 'track') ,
        obj.processName = 'track' ;
        obj.bgContCallInterval = 20 ;  % secs
      else
        error('Internal error: BgMonitor() argument must be ''train'' or ''track''') ;
      end

      % bgWorkerObj knows how to poll the state of the process. 
      % monVizObj knows how to vizualize this state. 
      % bgResultsReceivedHook performs custom actions after receiving
      % an update from bgWorkerObj. 
      %
      % bgWorkerObj/monVizObj should be mix+matchable as bgWorkerObj 
      % should send a core set of 'standard' metrics that monVizObj can
      % use.
      %
      % bgWorkerObj matches 1-1 with the concrete BgMonitor and its 
      % bgResultsReceivedHook method. These work in concert and the 
      % custom actions taken by bgResultsReceivedHook depends on custom 
      % info supplied by bgWorkerObj.
      [track_type] = myparse(varargin,'track_type','movie');
      if strcmp(track_type,'movie')
        compute_fcn_name = 'compute';
      else
        compute_fcn_name = 'computeList';
      end

      %obj.reset_();  % Not needed
      
      [tfEFE,errFile] = bgWorkerObj.errFileExists;
      if tfEFE
        error('Error file ''%s'' exists.',errFile);
      end
      
      cbkResult = @obj.bgResultsReceived;

      bgc = BgClient() ;
      fprintf(1,'Configuring background worker...\n');
      bgc.configure(cbkResult, bgWorkerObj, compute_fcn_name) ;
      
      obj.bgClientObj = bgc;
      obj.bgWorkerObj = bgWorkerObj;
      obj.monitorObj = monVizObj;
      if exist('cbkComplete','var'),
        obj.cbkComplete = cbkComplete;
      end
    end  % constructor
    
    function delete(obj)
      if obj.isRunning
        obj.notify('bgEnd');
      end
      
      % IMHO, it's a code smell that we explicitly delete() all these things in a
      % delete() method...  -- ALT, 2024-06-28
      if ~isempty(obj.bgClientObj)
        delete(obj.bgClientObj);
      end
      obj.bgClientObj = [];
      
      if ~isempty(obj.bgWorkerObj)
        delete(obj.bgWorkerObj)
      end
      obj.bgWorkerObj = [];
      
      obj.cbkComplete = [];
      
      if ~isempty(obj.monitorObj)
        delete(obj.monitorObj);
      end
      obj.monitorObj = [];
    end  % delete() method
    
    function v = get.prepared(obj)
      v = ~isempty(obj.bgClientObj);
    end

    function v = get.isRunning(obj)
      bgc = obj.bgClientObj;
      v = ~isempty(bgc) && bgc.isRunning;
    end
    
    function start(obj)
      assert(obj.prepared);
      bgc = obj.bgClientObj;
      bgc.startRunner('runnerContinuous',true,...
                      'continuousCallInterval',obj.bgContCallInterval);
      obj.notify('bgStart');
    end
    
    function stop(obj)
      bgc = obj.bgClientObj;
      bgc.stopWorkerHard();
      obj.notify('bgEnd');
    end
    
    function bgResultsReceived(obj,sRes)
      % current pattern is, this meth only handles things which stop the
      % process. everything else handled by monitor

	    % tfSucc = false when bgMonitor should be stopped because resultsReceived found an issue
      [tfSucc,msg] = obj.monitorObj.resultsReceived(sRes);
      
      BgMonitor.debugfprintf('bgResultsReceived: tfSucc = %d\n',tfSucc);
      
      tfpollsucc = BgMonitor.getPollSuccess(sRes);
      
      killOccurred = any(tfpollsucc & BgMonitor.getKillOccurred(sRes));
      if killOccurred
        obj.stop();        
        fprintf(1,'Process killed!\n');
        return;
        % monitor plot stays up; reset not called etc
      end
      
      errOccurred = any(tfpollsucc & BgMonitor.getErrOccurred(sRes));
      if errOccurred
        obj.stop();

        fprintf(1,'Error occurred during %s:\n',obj.processName);
        errFile = BgMonitor.getErrFile(sRes); % currently, errFiles same for all views
        if iscell(errFile) ,
          if isscalar(errFile) ,
            errFile = errFile{1} ;
          else
            error('errFile is a non-scalar cell array')
          end
        end        
        fprintf(1,'\n### %s\n\n',errFile);
        errContents = obj.bgWorkerObj.fileContents(errFile);
        disp(errContents);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        return;
        
        % monitor plot stays up; reset not called etc
      end
      
      logFileErrLikely = BgMonitor.getLogFileErrLikely(sRes);
      for i=1:numel(sRes.result)
        if tfpollsucc(i) && logFileErrLikely(i),
          obj.stop();
          
          fprintf(1,'Error occurred during %s:\n',obj.processName);
          logFiles = BgMonitor.getLogFile(sRes,i);  % This is a cell array of char arrays, at least sometimes
          displayFileOrFiles(logFiles, obj.bgWorkerObj) ;
          fprintf(1,'\n\n. You may need to manually kill any running %s process.\n',obj.processName);
          return;
          
          % monitor plot stays up; bgReset not called etc
        end
      end
            
      tfComplete = all(tfpollsucc & BgMonitor.isComplete(sRes));
      if tfComplete
        obj.stop();
        % % monitor plot stays up; reset not called etc
        fprintf(1,'%s complete at %s.\n',obj.processName,datestr(now));
        
        if ~isempty(obj.cbkComplete),
          obj.cbkComplete(sRes.result);
        end
        return
      end
      
      % KB: check if resultsReceived found a reason to stop 
      if ~tfSucc,
        if isempty(msg),
          fprintf('resultsReceived did not return success. Stopping.\n');
        else
          fprintf('%s - Stopping.\n',msg);
        end
        obj.stop();
        return
      end      
    end  % function bgResultsReceived
  end  % methods
  
  methods (Static)
    function tfpollsucc = getPollSuccess(sRes)
      if isfield(sRes.result,'pollsuccess'),
        tfpollsucc = [sRes.result.pollsuccess];
      else
        tfpollsucc = true(1,numel(sRes.result));
      end
    end

    function killOccurred = getKillOccurred(sRes)
      killOccurred = [sRes.result.killFileExists];
    end

    function errOccurred = getErrOccurred(sRes)
      errOccurred = [sRes.result.errFileExists];
    end
    
    function errFile = getErrFile(sRes)
      errFile = sRes.result(1).errFile;
    end

    function logFile = getLogFile(sRes,i)
      logFile = sRes.result(i).logFile;
    end

    function logFileErrLikely = getLogFileErrLikely(sRes)
      logFileErrLikely = [sRes.result.logFileErrLikely];
    end

    function tfComplete = isComplete(sRes)
      tfComplete = [sRes.result.tfComplete];
    end

    function debugfprintf(varargin)
      DEBUG = false ;
      if DEBUG ,
        fprintf(varargin{:});
      end
    end  % function    
  end  % methods (Static)
end  % classdef
