classdef BgMonitor < handle
  % BGMonitor
  %
  % A BGMonitor is:
  % 1. A BgClient comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of the process. For
  %     debugging/testing this can be done from the client machine.
  % 2. A client-side MonitorViz object that visualizes 
  % progress sent back from runPollongLoop()
  % 3. Custom actions performed when process is complete
  %
  % BGMonitor does NOT know how to spawn process jobs but will know
  % how to (attempt to) kill them. For debugging, you can manually spawn 
  % jobs and monitor them with BgMonitor.
  %
  % BGMonitor does NOT know how to probe the detailed state of the
  % process eg on disk. That is BgWorkerObj's domain.
  %
  % So BGMonitor is a connector/manager obj that runs the worker 
  % (knows how to poll the filesystem in detail) in the background and 
  % connects it with a Monitor.
  %
  % See also prepare() method comments for related info.
  
  properties
    bgContCallInterval  % scalar double, in secs    
    bgClientObj  % the BgClient
    bgWorkerObj  % scalar "detached" object (not sure this is still true about it being 
                 % detached --ALT, 2024-06-28) that is deep-copied onto
                 % workers.
    monitorVizObj  % object with resultsreceived() method, typically a "monitor visualizer"
    cbkComplete  % empty, or fcnhandle with sig cbk(res), called when operation complete
    processName  % 'train' or 'track'
    % It seems like the bgClientObj and monitorObj are owned by obj, but
    % bgWorkerObj is not (hence that ref should be treated as soft).  The
    % effective parent of obj is the DeepTracker.  cbkComplete seems to usually
    % (always?) call a method of the parent DeepTracker.  Would probably make
    % sense to add a "parent" field that contains a ref to the DeepTracker, and to
    % call a "didCompleteBgMonitor" method of the parent when complete.  The
    % bgWorkerObj is owned by the parent DeepTracker.  -- ALT, 2024-06-28
    parent_  % the DeepTracker object that created obj
    projTempDirMaybe_
    tfComplete_  
      % Initialized to false in start() method, set to true when completion detected.
      % Used to prevent post-completion code from running twice.
  end

  properties (Dependent)
    prepared
    isRunning
  end
    
%   events
%     bgStart
%     bgEnd    
%   end
  
  methods
    function obj = BgMonitor(parent, type_string, monVizObj, bgWorkerObj, cbkComplete, varargin)      
      % Is obj for monitoring training or tracking?
      obj.parent_ = parent ;
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
      [track_type, projTempDir] = myparse(varargin, ...
                                          'track_type', 'movie', ...
                                          'projTempDir', []);
      if strcmp(track_type,'movie')
        compute_fcn_name = 'compute';
      else
        compute_fcn_name = 'computeList';
      end
      if isempty(projTempDir) ,
        obj.projTempDirMaybe_ = {} ;
      else
        obj.projTempDirMaybe_ = { projTempDir } ;
      end

      %obj.reset_();  % Not needed
      
      [tfEFE,errFile] = bgWorkerObj.errFileExists;
      if tfEFE
        error('Error file ''%s'' exists.',errFile);
      end
      
      cbkResult = @obj.bgResultsReceived;

      fprintf(1,'Configuring background worker...\n');
      bgc = BgClient(cbkResult, bgWorkerObj, compute_fcn_name, 'projTempDirMaybe', obj.projTempDirMaybe_) ;
      
      obj.bgClientObj = bgc;
      obj.bgWorkerObj = bgWorkerObj;
      obj.monitorVizObj = monVizObj;
      if exist('cbkComplete','var'),
        obj.cbkComplete = cbkComplete;
      end
    end  % constructor
    
    function delete(obj)
      if obj.isRunning ,
        %obj.notify('bgEnd');
        if ~isempty(obj.parent_)  && isvalid(obj.parent_) ,
          obj.parent_.didStopBgMonitor(obj.processName) ;
        end
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
      
      if ~isempty(obj.monitorVizObj)
        delete(obj.monitorVizObj);
      end
      obj.monitorVizObj = [];
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
      obj.tfComplete_ = false ;
      bgc = obj.bgClientObj;
      bgc.startRunner('continuousCallInterval',obj.bgContCallInterval) ;
      obj.parent_.didStartBgMonitor(obj.processName) ;
    end
    
    function stop(obj)
      bgc = obj.bgClientObj;
      bgc.stopRunnerHard();
      obj.parent_.didStopBgMonitor(obj.processName) ;
    end
    
    function waitForJobsToExit(obj)
      obj.parent_.waitForJobsToExit(obj.processName) ;
    end
    
    function bgResultsReceived(obj,sRes)
      % current pattern is, this meth only handles things which stop the
      % process. everything else handled by obj.monitorVizObj

	    % tfSucc = false when bgMonitor should be stopped because resultsReceived found an issue
      [tfSucc,msg] = obj.monitorVizObj.resultsReceived(sRes);
      
      BgMonitor.debugfprintf('bgResultsReceived: tfSucc = %d\n',tfSucc);
      
      tfpollsucc = BgMonitor.getPollSuccess(sRes);
      
      killOccurred = any(tfpollsucc & BgMonitor.getKillOccurred(sRes));
      if killOccurred
        obj.stop();        
        fprintf(1,'Process killed!\n');
        return
        % monitor plot stays up; reset not called etc
      end
      
      errOccurred = any(tfpollsucc & BgMonitor.getErrOccurred(sRes));
      if errOccurred
        obj.stop();

        fprintf('Error occurred during %s:\n',obj.processName);
        errFile = BgMonitor.getErrFile(sRes); % currently, errFiles same for all views
        if iscell(errFile) ,
          if isscalar(errFile) ,
            errFile = errFile{1} ;
          else
            error('errFile is a non-scalar cell array')
          end
        end        
        fprintf('\n### %s\n\n',errFile);
        errContents = obj.bgWorkerObj.fileContents(errFile);
        disp(errContents);
        % We've taked steps to kill any running DL processes -- ALT, 2024-10-10
        %fprintf('\n\nYou may need to manually kill any running DeepLearning process.\n');
        return
        
        % monitor plot stays up; reset not called etc
      end
      
      logFileErrLikely = BgMonitor.getLogFileErrLikely(sRes);
      for i=1:numel(sRes.result)
        if tfpollsucc(i) && logFileErrLikely(i),
          obj.stop();
          
          fprintf(1,'Error occurred during %s:\n',obj.processName);
          logFiles = BgMonitor.getLogFile(sRes,i);  % This is a cell array of char arrays, at least sometimes
          displayFileOrFiles(logFiles, obj.bgWorkerObj) ;
          % We've taked steps to kill any running DL processes -- ALT, 2024-10-10
          %fprintf('\n\nYou may need to manually kill any running %s process.\n',obj.processName);
          return
          
          % monitor plot stays up; bgReset not called etc
        end
      end
            
      if ~obj.tfComplete_  % If we've already done the post-completion stuff, don't want to do it again
        obj.tfComplete_ = all(tfpollsucc & BgMonitor.isComplete(sRes));
        if obj.tfComplete_
          %obj.bgClientObj.stopRunnerHard();  % Stop the runner immediately, so we don't handle completion twice
          obj.waitForJobsToExit() ;  
            % Right now, tfComplete is true as soon as the output files *exist*.
            % This can lead to issues if they're not done being written to, so we wait for
            % the job(s) to exit before proceeding.
          obj.stop();
          % % monitor plot stays up; reset not called etc
          fprintf('%s complete at %s.\n',obj.processName,datestr(now()));
          
          if ~isempty(obj.cbkComplete),
            obj.cbkComplete(sRes.result);
          end
          return
        end
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
