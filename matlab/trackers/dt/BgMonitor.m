classdef BgMonitor < handle
  % The BGMonitor is responsible for receiving polling results from the
  % BgClient (which gets them from the BgPoller running in a separate process),
  % and notifying the parent DeepTracker as needed.  It also is the DeepTracker's
  % point of contact for controlling the monitoring.  E.g. monitoring starts
  % when the BgMonitor gets the start() message from the DeepTracker.
   
  properties
    pollInterval  % scalar double, in secs    
    bgClientObj  % the BgClient
    %poller  % a BgTrainPoller or BgTrackPoller object
    %monitorVizObj  % object with resultsreceived() method, typically a "monitor visualizer"
    processName  % 'train' or 'track'
    parent_  % the (typically) DeepTracker object that created this BgMonitor
    projTempDirMaybe_
    tfComplete_  
      % Initialized to false in start() method, set to true when completion detected.
      % Used to prevent post-completion code from running twice.
  end

  properties (Dependent)
    isRunning
  end

  properties (Transient)
    sRes
  end

  methods
    function obj = BgMonitor(parent, type_string, poller, varargin)      
      % Is obj for monitoring training or tracking?
      obj.parent_ = parent ;
      if strcmp(type_string, 'train') ,
        obj.processName = 'train' ;
        obj.pollInterval = 30 ;  % secs
      elseif strcmp(type_string, 'track') ,
        obj.processName = 'track' ;
        obj.pollInterval = 20 ;  % secs
      else
        error('Internal error: BgMonitor() argument must be ''train'' or ''track''') ;
      end

      % poller knows how to poll the state of the process. 
      % monVizObj knows how to vizualize this state. 
      % didReceivePollResults performs custom actions after receiving
      % an update from poller. 
      %
      % poller/monVizObj should be mix+matchable as poller 
      % should send a core set of 'standard' metrics that monVizObj can
      % use.
      %
      % poller matches 1-1 with the concrete BgMonitor and its 
      % didReceivePollResults method. These work in concert and the 
      % custom actions taken by didReceivePollResults depends on custom 
      % info supplied by poller.
      projTempDir = myparse(varargin, ...
                            'projTempDir', []);
      if isempty(projTempDir) ,
        obj.projTempDirMaybe_ = {} ;
      else
        obj.projTempDirMaybe_ = { projTempDir } ;
      end

      %obj.reset_();  % Not needed
      
      % [tfEFE,errFile] = poller.errFileExists;
      % if tfEFE
      %   error('Error file ''%s'' exists.',errFile);
      % end
      
      fprintf('Configuring background poller client...\n');
      bgc = BgClient(obj, poller, 'projTempDirMaybe', obj.projTempDirMaybe_) ;
      
      obj.bgClientObj = bgc;
      %obj.poller = poller;
      %obj.monitorVizObj = monVizObj;
    end  % constructor
    
    function delete(obj)
      if obj.isRunning ,
        obj.stop() ;
      end
      
      % IMHO, it's a code smell that we explicitly delete() all these things in a
      % delete() method...  -- ALT, 2024-06-28
      if ~isempty(obj.bgClientObj)
        delete(obj.bgClientObj);
      end
      obj.bgClientObj = [];
      
      % if ~isempty(obj.poller)
      %   delete(obj.poller)
      % end
      % obj.poller = [];
    end  % delete() method
    
    function v = get.isRunning(obj)
      bgc = obj.bgClientObj;
      v = ~isempty(bgc) && bgc.isRunning;
    end
    
    function start(obj)
      obj.tfComplete_ = false ;
      bgc = obj.bgClientObj;
      bgc.startPollingLoop(obj.pollInterval) ;
      obj.parent_.didStartBgMonitor(obj.processName) ;
    end
    
    function stop(obj)
      % This can be called from the delete() method, so we are extra careful about
      % making sure the message targets are valid.
      sendMaybe(obj.bgClientObj, 'stopPollingLoopHard') ;
      sendMaybe(obj.parent_, 'didStopBgMonitor', obj.processName) ;
    end
    
    function waitForJobsToExit(obj)
      obj.parent_.waitForJobsToExit(obj.processName) ;
    end
    
    function didReceivePollResults(obj, sRes)
      % current pattern is, this meth only handles things which stop the
      % process. everything else handled by obj.monitorVizObj

	    % tfSucc = false when bgMonitor should be stopped because resultsReceived found an issue
      obj.sRes = sRes ;  % Stash so to controllers/views have access to it.
      if strcmp(obj.processName, 'track') 
        obj.parent_.didReceiveTrackingPollResults_() ;
      elseif strcmp(obj.processName, 'train') 
        obj.parent_.didReceiveTrainingPollResults_() ;
      else
        error('Internal error.  Tell Adam he''s fired.') ;
      end
      %[tfSucc,msg] = obj.monitorVizObj.resultsReceived(sRes);
      tfSucc = true ;
      
      BgMonitor.debugfprintf('BgMonitor.didReceivePollResults: tfSucc = %d\n',tfSucc);
      
      tfpollsucc = BgMonitor.getPollSuccess(sRes);
      
      % killOccurred = any(tfpollsucc & BgMonitor.getKillOccurred(sRes));
      % if killOccurred
      %   obj.stop();        
      %   fprintf(1,'Process killed!\n');
      %   return
      %   % monitor plot stays up; reset not called etc
      % end
      
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
        errContents = obj.parent_.backend.fileContents(errFile) ;
        disp(errContents);
        % We've taked steps to kill any running DL processes -- ALT, 2024-10-10
        %fprintf('\n\nYou may need to manually kill any running DeepLearning process.\n');
        return
        
        % monitor plot stays up; reset not called etc
      end
                  
      if ~obj.tfComplete_  % If we've already done the post-completion stuff, don't want to do it again
        obj.tfComplete_ = all(tfpollsucc & BgMonitor.isComplete(sRes));
        if obj.tfComplete_
          obj.waitForJobsToExit() ;  
            % Right now, tfComplete is true as soon as the output files *exist*.
            % This can lead to issues if they're not done being written to, so we wait for
            % the job(s) to exit before proceeding.
          obj.stop();
          % % monitor plot stays up; reset not called etc
          fprintf('%s complete at %s.\n',obj.processName,datestr(now()));
          
          % Signal to parent object, typically a DeepTracker, that tracking/training
          % has completed.
          if strcmp(obj.processName, 'track') ,
            obj.parent_.didCompleteTracking(sRes.result) ;
          elseif strcmp(obj.processName, 'train') ,
            obj.parent_.didCompleteTraining(sRes.result) ;
          else
            error('Internal error: Unknown processName %s', obj.processName) ;
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
    end  % function didReceivePollResults
  end  % methods
  
  methods (Static)
    function tfpollsucc = getPollSuccess(sRes)
      if isfield(sRes.result,'pollsuccess'),
        tfpollsucc = [sRes.result.pollsuccess];
      else
        tfpollsucc = true(1,numel(sRes.result));
      end
    end

    % function killOccurred = getKillOccurred(sRes)
    %   killOccurred = [sRes.result.killFileExists];
    % end

    function errOccurred = getErrOccurred(sRes)
      errOccurred = [sRes.result.errFileExists];
    end
    
    function errFile = getErrFile(sRes)
      errFile = sRes.result(1).errFile;
    end

    function logFile = getLogFile(sRes,i)
      logFile = sRes.result(i).logFile;
    end

    function result = getLogFileErrLikely(sRes)
      result = false(size(sRes.result)) ;
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
