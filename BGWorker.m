classdef BGWorker < handle
  properties (Constant)
    STOPACTION = 'STOP';
  end
  properties
    qMe2Client % matlab.pool.DataQueue for sending data to Client (interrupts Client)
    qClient2Me % matlab.pool.PollableDataQueue for receiving data from Client (polled)
    
    computeObj % object that knows how to perform desired computations
  end
  methods
    function obj = BGWorker
    end
    function log(obj,varargin) %#ok<INUSL>
      str = sprintf(varargin{:});
      fprintf(1,'BGWorker (%s): %s\n',datestr(now,'yyyymmddTHHMMSS'),str);
    end
    function status = start(obj,dataQueue,cObj)
      % Spin up worker; call via parfeval
      % 
      % dataQueue: parallel.pool.DataQueue created by Client
      % cObj: object with method .compute(action,data) where action is a
      %   str
      
      assert(isa(dataQueue,'parallel.pool.DataQueue'));
      
      obj.qMe2Client = dataQueue;
      pdQueue = parallel.pool.PollableDataQueue;
      dataQueue.send(pdQueue);
      obj.qClient2Me = pdQueue;
      obj.log('Done configuring queues');
      
      obj.computeObj = cObj;
      
      while true
        [data,ok] = pdQueue.poll();
        if ok
          assert(isstruct(data) && all(isfield(data,{'action' 'data' 'id'})));
          action = data.action;          
          obj.log('Received %s',action);
          if strcmp(action,BGWorker.STOPACTION)
            break;
          else
            result = cObj.compute(data);
            dataQueue.send(struct('id',data.id,'result',result));
          end
        end
      end
      
      status = 1;
    end
  end
end
  