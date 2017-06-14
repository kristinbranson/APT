classdef BGClient < handle
  properties
    qWorker2Me % matlab.pool.DataQueue for receiving data from worker (interrupts)
    qMe2Worker % matlab.pool.PollableDataQueue for sending data to Client (polled)
    
    fevalFuture % FevalFuture output from parfeval
    clientObj % object that work results are sent to; clientObj.newResult(s) where s
              % has fields .id and .result
    idPool % scalar uint for cmd ids
  end
  methods 
    function obj = BGClient
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
        delete(obj.fevalFuture);
        obj.fevalFuture = [];
      end
    end
  end
  
  methods
    function log(obj,varargin) %#ok<INUSL>
      str = sprintf(varargin{:});
      fprintf(1,'BGClient (%s): %s\n',datestr(now,'yyyymmddTHHMMSS'),str);
    end
    function startWorker(obj,clientObj,computeObj)
      obj.clientObj = clientObj;
      
      queue = parallel.pool.DataQueue;
      queue.afterEach(@(dat)obj.afterEach(dat));
      obj.qWorker2Me = queue;
      
      workerObj = BGWorker;
      % computeObj deep-copied onto worker
      obj.fevalFuture = parfeval('start',1,workerObj,queue,computeObj); 
      
      obj.idPool = uint32(1);
    end
    function afterEach(obj,dat)
      if isa(dat,'parallel.pool.PollableDataQueue')
        obj.qMe2Worker = dat;
        obj.log('Received pollableDataQueue from worker.');
      else
        obj.log('Received results id %d',dat.id);
        obj.clientObj.newResult(dat);
      end
    end
    function sendCommand(obj,sCmd)
      % Send command to worker; startWorker() must have been called
      % 
      % sCmd: struct with fields {'action' 'data'}
      
      assert(isstruct(sCmd) && all(isfield(sCmd,{'action' 'data'})));
      sCmd.id = obj.idPool;
      obj.idPool = obj.idPool + 1;
      
      q = obj.qMe2Worker;
      if isempty(q)
        warningNoTrace('BGClient:queue','Send queue not configured yet.');
      else
        q.send(sCmd);
        obj.log('Sent command id %d',sCmd.id);
      end
    end
    function stopWorker(obj)      
      sCmd = struct('action',BGWorker.STOPACTION,'data',[]);
      obj.sendCommand(sCmd);
    end
  end
end
