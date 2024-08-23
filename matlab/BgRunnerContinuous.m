classdef BgRunnerContinuous < handle
  % BgRunnerContinuous is a worker that runs a repeated computation in the
  % background at regular intervals. Each time it runs its computation, it
  % sends a message back. After starting, all you can do is tell it to 
  % stop.  Oh, and also to report its status.
  
  properties (Constant)
    STOPACTION = 'STOP';
    STATACTION = 'STAT';
  end
  
  properties
    computeTimes = zeros(0,1)  % vector of tic/toc compute time elapsed for each compute command received
  end
  
  methods    
    function obj = BgRunnerContinuous()
      tfPre2017a = verLessThan('matlab','9.2.0');
      if tfPre2017a
        error('Background processing requires Matlab 2017a or later.');
      end
    end    

    function delete(obj)  %#ok<INUSD> 
      fprintf('A BgRunnerContinuous is being deleted.\n') ;
    end
    
    function status = run(obj,dataQueue,cObj,cObjMeth,callInterval)
      % Spin up worker; call via parfeval
      % 
      % dataQueue: parallel.pool.DataQueue created by Client
      % cObj: object with method cObjMeth
      % callInterval: time in seconds to wait between calls to
      %   cObj.(cObjMeth)
      
      logger = FileLogger('BgRunnerContinuous.log', 'BgRunnerContinuous') ;

      logger.log('Inside BgRunnerContinuous::run()\n') ;

      logger.log('cObj:\n') ;
      logger.log(formattedDisplayText(cObj)) ;
      logger.log('\n') ;

      if isa(cObj, 'BgTrackWorkerObjAWS') ,
        logger.log('cObj.awsEc2:\n') ;
        logger.log(formattedDisplayText(cObj.awsEc2)) ;
        logger.log('\n') ;
      end      

      assert(isa(dataQueue,'parallel.pool.DataQueue'));
      pdQueue = parallel.pool.PollableDataQueue;
      dataQueue.send(pdQueue);
      logger.log('Done configuring queues');
            
      iterations_completed = 0 ;
      while true        
        tic_id = tic() ;
        
        [data,ok] = pdQueue.poll();
        if ok
          assert(isstruct(data) && all(isfield(data,{'action' 'data' 'id'})));
          action = data.action;          
          logger.log('Received %s',action);
          switch action
            case BgRunner.STOPACTION
              break
            case BgRunner.STATACTION
              sResp = struct('id',data.id,'action',action,'result',obj.computeTimes);
              dataQueue.send(sResp);
            otherwise
              error('Unrecognized action: %s',action);
          end
        else
          % continue
        end
        
        result = cObj.(cObjMeth)(logger);
        result.iterations_completed = iterations_completed ;
        obj.computeTimes(end+1,1) = toc(tic_id) ;
        dataQueue.send(struct('id',0,'action','','result',{result}));
%         dataQueue.send(struct('id',data.id,'action',action,'result',result));

        logger.log('Pausing...');
        pause(callInterval);
        logger.log('Done pausing...');
        iterations_completed = iterations_completed + 1 ;
        logger.log('iterations_completed: %d\n', iterations_completed) ;
      end
      
      status = 1;
      logger.log('About to exit BgRunnerContinuous::run()\n') ;
    end  % function
    
  end  % methods

end  % classdef
  