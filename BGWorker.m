classdef BGWorker < handle
  % BGWorker is a worker that idles in the background. When you send it a
  % command, it runs it and sends the result back and goes back to idling.
  % It runs one compute call per sent command.
  
  properties (Constant)
    STOPACTION = 'STOP';
    STATACTION = 'STAT';
  end
  
  properties
    computeTimes = zeros(0,1); % vector of tic/toc compute time elapsed for each compute command received
  end
  
  methods    
    function obj = BGWorker
      tfPre2017a = verLessThan('matlab','9.2.0');
      if tfPre2017a
        error('BG:ver','Background processing requires Matlab 2017a or later.');
      end
    end    
  end
  
  methods     
    
    function status = start(obj,dataQueue,cObj,cObjMeth)
      % Spin up worker; call via parfeval
      % 
      % dataQueue: parallel.pool.DataQueue created by Client
      % cObj: object with method .compute(action,data) where action is a
      %   str
      
      assert(isa(dataQueue,'parallel.pool.DataQueue'));
      pdQueue = parallel.pool.PollableDataQueue;
      dataQueue.send(pdQueue);
      obj.log('Done configuring queues');
            
      while true
        [data,ok] = pdQueue.poll();
        if ok
          assert(isstruct(data) && all(isfield(data,{'action' 'data' 'id'})));
          action = data.action;          
          obj.log('Received %s',action);
          switch action
            case BGWorker.STOPACTION
              break;
            case BGWorker.STATACTION
              sResp = struct('id',data.id,'action',action,'result',obj.computeTimes);
              dataQueue.send(sResp);
            otherwise
              tic;
              result = cObj.(cObjMeth)(data);
              obj.computeTimes(end+1,1) = toc;
              dataQueue.send(struct('id',data.id,'action',action,'result',result));
          end
        end
      end
      
      status = 1;
    end
    
  end
  
  methods (Access=private)    
    function log(obj,varargin) %#ok<INUSL>
      str = sprintf(varargin{:});
      fprintf(1,'BGWorker (%s): %s\n',datestr(now,'yyyymmddTHHMMSS'),str);
    end    
  end
  
end
  