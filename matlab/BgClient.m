classdef BgClient < handle
  properties
    cbkResult % function handle called when a new result is computed. 
              % Signature:  cbkResult(s) where s has fields .id, .action, .result
    computeObj % Object with method .(computeObjMeth)(s) where s has fields .id, .action, .data
    computeObjMeth % compute method name for computeObj

    qWorker2Me % matlab.pool.DataQueue for receiving data from worker (interrupts)
    qMe2Worker % matlab.pool.PollableDataQueue for sending data to Worker (polled)    
    fevalFuture % FevalFuture output from parfeval
    isContinuous = false % scalar. if true, worker is a continuous worker
    idPool % scalar uint for cmd ids
    idTics % [numIDsSent] uint64 col vec of start times for each command id sent 
    idTocs % [numIDsReceived] col vec of compute elapsed times, set when response to each command id is received
    
    printlog = false; % if true, logging messages are displayed
    
    parpoolIdleTimeout = 100*60; % bump gcp IdleTimeout to at least this value every time a worker is started
  end
  properties (Dependent)
    isConfigured
    isRunning 
  end
  methods 
    function v = get.isConfigured(obj)
      v = ~isempty(obj.cbkResult) && ~isempty(obj.computeObj) && ~isempty(obj.computeObjMeth);
    end    
    function v = get.isRunning(obj)
      v = ~isempty(obj.fevalFuture) && strcmp(obj.fevalFuture.State,'running');
      %fprintf('In BgClient::get.isRunning(), isRunning is %d\n', v) ;
    end
  end

  methods 
    function obj = BgClient()
      tfPre2017a = verLessThan('matlab','9.2.0');
      if tfPre2017a
        error('BG:ver','Background processing requires Matlab 2017a or later.');
      end
    end
    function delete(obj)
      if ~isempty(obj.qWorker2Me)
        delete(obj.qWorker2Me);
        obj.qWorker2Me = [];
      end
      if ~isempty(obj.qMe2Worker)
        delete(obj.qMe2Worker);
        obj.qMe2Worker = [];
      end
      if ~isempty(obj.fevalFuture)
        obj.fevalFuture.cancel();
        delete(obj.fevalFuture);
        obj.fevalFuture = [];
      end
    end
  end
  
  methods
    
    function configure(obj,resultCallback,computeObj,computeObjMeth)
      % Configure compute object and results callback
      
      assert(isa(resultCallback,'function_handle'));
      
      if ismethod(computeObj,'copyAndDetach')
        % AL20191218
        % Some computeObjs have properties that don't deep-copy well over
        % to the background worker via parfeval. Examples might be large UI
        % objects containing java handles etc.
        %
        % To deal with this, computeObjs may optionally copy and mutate
        % themselves to make themselves palatable for transmission through
        % parfeval and subsequent computation in the bg.
        %
        % Note computeObj doesn't do anything in this class besides get 
        % transmitted over via parfeval.
        computeObj = computeObj.copyAndDetach();
      end
      obj.cbkResult = resultCallback;
      obj.computeObj = computeObj; % will be deep-copied onto worker
      obj.computeObjMeth = computeObjMeth;
    end
    
    function startRunner(obj,varargin)
      % Start BgRunner on new thread
      
      [runnerContinuous,continuousCallInterval] = myparse(varargin,...
        'runnerContinuous',false,...
        'continuousCallInterval',nan);
      
      if ~obj.isConfigured
        error('BgClient:config',...
          'Object unconfigured; call configure() before starting worker.');
      end
      
      queue = parallel.pool.DataQueue;
      if runnerContinuous
      	queue.afterEach(@(dat)obj.afterEachContinuous(dat));
      else
        queue.afterEach(@(dat)obj.afterEach(dat));
      end		
      obj.qWorker2Me = queue;
      
      p = gcp() ;
      if obj.parpoolIdleTimeout > p.IdleTimeout 
        warningNoTrace('Increasing current parpool IdleTimeout to %d minutes.',obj.parpoolIdleTimeout);
        p.IdleTimeout = obj.parpoolIdleTimeout;
      end
      
      fprintf('obj.computeObj.awsEc2.sshCmd: %s\n', obj.computeObj.awsEc2.sshCmd) ;
      if runnerContinuous
        runnerObj = BgRunnerContinuous() ;
        % computeObj deep-copied onto worker
        obj.fevalFuture = ...
          parfeval(@run, 1, runnerObj, queue, obj.computeObj, obj.computeObjMeth, continuousCallInterval) ;
%         foo = feval(@start, workerObj, queue, obj.computeObj, obj.computeObjMeth, continuousCallInterval) ; 
        % ALTTODO: Make sure this is the parfeval version
      else      
        runnerObj = BgRunner() ;
        % computeObj deep-copied onto worker
        obj.fevalFuture = ...
          parfeval(@run, 1, runnerObj, queue, obj.computeObj, obj.computeObjMeth) ; 
      end
      
      obj.isContinuous = runnerContinuous;
      obj.idPool = uint32(1);
      obj.idTics = uint64(0);
      obj.idTocs = nan;
    end
        
    function sendCommand(obj,sCmd)
      % Send command to worker; startWorker() must have been called
      % 
      % sCmd: struct with fields {'action' 'data'}
            
      if ~obj.isRunning
        error('BgClient:run','Worker is not running.');
      end      
      
      assert(isstruct(sCmd) && all(isfield(sCmd,{'action' 'data'})));
      sCmd.id = obj.idPool;
      obj.idPool = obj.idPool + 1;
      
      q = obj.qMe2Worker;
      if isempty(q)
        warningNoTrace('BgClient:queue','Send queue not configured.');
      else
        obj.idTics(sCmd.id) = tic;
        q.send(sCmd);
        obj.log('Sent command id %d',sCmd.id);
      end
    end
    
    function stopWorker(obj)
      % "Proper" stop; STOP message is sent to BgRunner obj; BgRunner reads
      % STOP message and breaks from polling loop
      
      if ~obj.isRunning
        warningNoTrace('BgClient:run','Worker is not running.');
      else
        sCmd = struct('action',BgRunner.STOPACTION,'data',[]);
        obj.sendCommand(sCmd);
      end
    end
    
    function stopWorkerHard(obj)
      % Harder stop, cancel fevalFuture
      
      if ~obj.isRunning
        warningNoTrace('BgClient:run','Worker is not running.');
      else
        obj.fevalFuture.cancel();
      end
    end    
    
  end
  
  methods (Access=private)
    
    function log(obj,varargin)
      if obj.printlog
        str = sprintf(varargin{:});
        fprintf(1,'BgClient (%s): %s\n',datestr(now,'yyyymmddTHHMMSS'),str);
      else
        % for now don't do anything
      end
    end
    
    function afterEach(obj,dat)
      if isa(dat,'parallel.pool.PollableDataQueue')
        obj.qMe2Worker = dat;
        obj.log('Received pollableDataQueue from worker.');
      else
        obj.log('Received results id %d',dat.id);
        obj.idTocs(dat.id) = toc(obj.idTics(dat.id));
        obj.cbkResult(dat);
      end
    end    

    function afterEachContinuous(obj,dat)
      if isa(dat,'parallel.pool.PollableDataQueue')
        obj.qMe2Worker = dat;
        obj.log('Received pollableDataQueue from worker.');
      else
        obj.cbkResult(dat);
      end
    end    
    
  end
  
end
