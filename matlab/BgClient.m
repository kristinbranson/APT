classdef BgClient < handle
  % The BgClient is responsible for launching the polling loop via parfeval(), and
  % communicating with the resulting process.  But when it receives a message
  % from the background process about a new poll result, it simply passes it on
  % to its parent BgMonitor.

  properties
    poller  % Object with method .poll()

    qPoller2Me % matlab.pool.DataQueue for receiving data from poller (interrupts)
    % qMe2Poller % matlab.pool.PollableDataQueue for sending data to poller (polled)    
    fevalFuture % FevalFuture output from parfeval
    idPool % scalar uint for cmd ids
    idTics % [numIDsSent] uint64 col vec of start times for each command id sent 
    idTocs % [numIDsReceived] col vec of compute elapsed times, set when response to each command id is received
    
    printlog = false; % if true, logging messages are displayed
    
    parpoolIdleTimeout = 100*60; % bump gcp IdleTimeout to at least this value every time a poller is started
    projTempDirMaybe_ = {}
    parent_  % The parent object, typically a BgMonitor
  end

  properties (Dependent)
    isRunning  % true iff the polling loop process is running
  end
  
  methods 
    function v = get.isRunning(obj)
      v = ~isempty(obj.fevalFuture) && strcmp(obj.fevalFuture.State,'running');
      %fprintf('In BgClient::get.isRunning(), isRunning is %d\n', v) ;
    end
  end

  methods 
    function obj = BgClient(parent, poller, varargin)
      % parent is typically a BgMonitor
      tfPre2017a = verLessThan('matlab','9.2.0');
      if tfPre2017a
        error('BG:ver','Background processing requires Matlab 2017a or later.');
      end
      projTempDirMaybe = myparse(varargin, 'projTempDirMaybe', {}) ;

      % Configure poller object
      assert(ismethod(parent,'didReceivePollResultsRetrograde'));      
      obj.parent_ = parent ;
      obj.poller = poller ; % will be parfeval-copied into background process
      
      obj.projTempDirMaybe_ = projTempDirMaybe ;
    end

    function delete(obj)
      if ~isempty(obj.qPoller2Me)
        delete(obj.qPoller2Me);
        obj.qPoller2Me = [];
      end
      % if ~isempty(obj.qMe2Poller)
      %   delete(obj.qMe2Poller);
      %   obj.qMe2Poller = [];
      % end
      if ~isempty(obj.fevalFuture)
        obj.fevalFuture.cancel();
        delete(obj.fevalFuture);
        obj.fevalFuture = [];
      end
    end
  end
  
  methods    
    function startPollingLoop(obj, pollInterval)
      % Start runPollingLoop() on new thread
      
      fromPollingLoopDataQueue = parallel.pool.DataQueue() ;
      fromPollingLoopDataQueue.afterEach(@(dat)obj.didReceivePollResultsFromPollingLoop(dat));
      obj.qPoller2Me = fromPollingLoopDataQueue;
      
      p = gcp() ;
      if obj.parpoolIdleTimeout > p.IdleTimeout 
        warningNoTrace('Increasing current parpool IdleTimeout to %d minutes.',obj.parpoolIdleTimeout);
        p.IdleTimeout = obj.parpoolIdleTimeout;
      end
      
      % poller is saved a la saveobj() and then loaded a la loadobj() into polling loop process
      % We pack a suitcase so we can restore Transient properties on the other side.
      poller = obj.poller ;
      parfevalSuitcase = poller.packParfevalSuitcase() ;

      % Start production code
      obj.fevalFuture = ...
        parfeval(@runPollingLoop, 0, fromPollingLoopDataQueue, poller, parfevalSuitcase, pollInterval, obj.projTempDirMaybe_) ;
      % End production code

      % % Start debug code
      % tempfilename = tempname() ;
      % saveAnonymous(tempfilename, poller) ;  % simulate poller as it will be on the other side of the parfeval boundary
      % cleaner = onCleanup(@()(delete(tempfilename))) ;
      % poller = loadAnonymous(tempfilename) ;
      % feval(@runPollingLoop, fromPollingLoopDataQueue, poller, parfevalSuitcase, pollInterval, obj.projTempDirMaybe_) ;  %#ok<FVAL>
      % %   The feval() (not parfeval) line above is sometimes useful when debugging.
      % % End debug code

      obj.idPool = uint32(1);
      obj.idTics = uint64(0);
      obj.idTocs = nan;
    end
    
    % function sendCommand(obj,sCmd)
    %   % Send command to poller; runPollingLoop() must have been called
    %   % 
    %   % sCmd: struct with fields {'action' 'data'}
    % 
    %   if ~obj.isRunning
    %     error('BgClient:run','Runner is not running.');
    %   end      
    % 
    %   assert(isstruct(sCmd) && all(isfield(sCmd,{'action' 'data'})));
    %   sCmd.id = obj.idPool;
    %   obj.idPool = obj.idPool + 1;
    % 
    %   q = obj.qMe2Poller;
    %   if isempty(q)
    %     warningNoTrace('BgClient:queue','Send queue not configured.');
    %   else
    %     obj.idTics(sCmd.id) = tic();
    %     q.send(sCmd);
    %     obj.log('Sent command id %d',sCmd.id);
    %   end
    % end
    
    function stopPollingLoop(obj)
      % Stop polling loop by cancel fevalFuture      
      if obj.isRunning
        obj.fevalFuture.cancel();
      end
    end  % function    
    
  end  % methods
  
  methods (Access=private)
    
    function log(obj,varargin)
      if obj.printlog
        str = sprintf(varargin{:});
        fprintf(1,'BgClient (%s): %s\n',datestr(now(),'yyyymmddTHHMMSS'),str);
      else
        % for now don't do anything
      end
    end
    
    % function afterEach(obj,dat)
    %   if isa(dat,'parallel.pool.PollableDataQueue')
    %     obj.qMe2Poller = dat;
    %     obj.log('Received pollableDataQueue from poller.');
    %   else
    %     obj.log('Received results id %d',dat.id);
    %     obj.idTocs(dat.id) = toc(obj.idTics(dat.id));
    %     obj.cbkResult(dat);
    %   end
    % end    
    
    function didReceivePollResultsFromPollingLoop(obj, data)
      % Pass poll results on to the parent BgMonitor
      obj.parent_.didReceivePollResultsRetrograde(data) ;
    end  % function
    
  end
  
end
