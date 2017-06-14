classdef BGWorker < handle
  properties (Constant)
    STOPACTION = 'STOP';
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
          if strcmp(action,BGWorker.STOPACTION)
            break;
          else
            result = cObj.(cObjMeth)(data);
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
  