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
    pollingResult
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
    end
    
    function stop(obj)
      % Stop polling for training/tracking results.

      % This can be called from the delete() method, so we are extra careful about
      % making sure the message targets are valid.
      sendMaybe(obj.bgClientObj, 'stopPollingLoop') ;
    end
    
    function didReceivePollResults(obj, pollingResult)
      % Called by the BgClient when a polling result is received.  Checks for error
      % or completion and notifies the parent DeepTracker accordingly.

      % % DEBUG
      % pollingResult  %#ok<NOPRT>

      % Cause views/controllers to be updated with the latest poll results
      obj.pollingResult = pollingResult ;  % Stash so to controllers/views have access to it.
      obj.parent_.didReceivePollResults(obj.processName) ;
        % This call causes (through a child-to-parent call chain) the labeler to
        % notify() views/controllers that there's a training/tracking result, and that they should
        % update themselves accordingly.  But that's it. Determining that training/tracking is
        % complete is done below.
      
      % Determine whether the polling itself was successful or not
      didPollingItselfSucceed = pollingResult.pollsuccess ;  % logical scalar
      if ~didPollingItselfSucceed
        % Signal to parent object, typically a DeepTracker, that tracking/training
        % has errored.
        obj.parent_.didErrorDuringTrainingOrTracking(obj.processName, pollingResult) ;

        % If we get here, we're done dealing with the current polling result        
        return
      end
      
      % Check for errors.      
      errFileExists = pollingResult.errFileExists ;  % could be njobs x 1, or nmovies x nviews x nstages
      tfComplete = pollingResult.tfComplete ;  % could be njobs x 1, or nmovies x nviews x nstages
      isRunning = pollingResult.isRunning ;  % could be njobs x 1, or nmovies x nviews x nstages
      isPopulated = pollingResult.isPopulated ;  % could be njobs x 1, or nmovies x nviews x nstages
        % However shaped, the four vars above should have the *same* shape.
        % isPopulated indicates which elements of the other three correspond to
        % actual jobs, rather than just being set to a default value.
      didErrorOccur = any(isPopulated & (errFileExists | (~tfComplete & ~isRunning)), 'all') ;
      if didErrorOccur
        % Signal to parent object, typically a DeepTracker, that tracking/training
        % has errored.
        obj.parent_.didErrorDuringTrainingOrTracking(obj.processName, pollingResult) ;

        % If we get here, we're done dealing with the current polling result        
        return
      end
                  
      % Check for completion.
      if ~obj.tfComplete_  % If we've already done the post-completion stuff, don't want to do it again
        obj.tfComplete_ = all(~isPopulated | tfComplete, 'all') ;
        if obj.tfComplete_
          % Send message to console
          fprintf('%s complete at %s.\n',obj.processName,datestr(now()));
          
          % Signal to parent object, typically a DeepTracker, that tracking/training
          % has completed.
          obj.parent_.didCompleteTrainingOrTracking(obj.processName, pollingResult) ;

          % If we get here, we're done dealing with the current polling result
          return
        end
      end
    end  % function didReceivePollResults
  end  % methods
  
  % methods (Static)
  %   % function tfpollsucc = getPollSuccess(pollingResult)
  %   %   if isfield(pollingResult,'pollsuccess'),
  %   %     tfpollsucc = [pollingResult.pollsuccess] ;
  %   %   else
  %   %     tfpollsucc = true(1,numel(pollingResult)) ;
  %   %   end
  %   % end
  % 
  %   % function killOccurred = getKillOccurred(pollingResult)
  %   %   killOccurred = [pollingResult.killFileExists];
  %   % end
  % 
  %   % function errOccurred = getErrOccurred(pollingResult)
  %   %   errOccurred = [pollingResult.errFileExists] ;
  %   % end
  % 
  %   % function errFile = getErrFile(pollingResult)
  %   %   errFile = pollingResult.errFile{1} ;
  %   % end
  % 
  %   % function logFile = getLogFile(pollingResult,i)
  %   %   logFile = pollingResult.logFile{i} ;
  %   % end
  % 
  %   % function result = getLogFileErrLikely(pollingResult)
  %   %   result = false(size(pollingResult)) ;
  %   % end
  % 
  %   % function tfComplete = isComplete(pollingResult)
  %   %   tfComplete = [pollingResult.tfComplete] ;
  %   % end
  % 
  %   % function debugfprintf(varargin)
  %   %   DEBUG = false ;
  %   %   if DEBUG ,
  %   %     fprintf(varargin{:});
  %   %   end
  %   % end  % function    
  % end  % methods (Static)
end  % classdef
