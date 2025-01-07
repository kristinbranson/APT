function status = runPollingLoop(toClientDataQueue, worker, backendSuitcase, pollInterval, projTempDirMaybe)
% Run the polling loop; typically called via parfeval
% 
% toClientDataQueue: parallel.pool.DataQueue created by BgClient
% worker: object with method cObjMeth
% callInterval: time in seconds to wait between calls to
%   worker.(cObjMeth)

% Unpack the backend suitcase
worker.backend.restoreAfterParfeval(backendSuitcase) ;

% Set up the log file, if called for
if isempty(projTempDirMaybe) ,
  logger = FileLogger() ;  % silently ignores .log() calls
else
  projTempDir = projTempDirMaybe{1} ;
  logFilePath = fullfile(projTempDir, 'runPollingLoop.log') ;
  logger = FileLogger(logFilePath, 'runPollingLoop') ;
end

logger.log('Inside runPollingLoop()\n') ;

logger.log('cObj:\n') ;
logger.log(formattedDisplayText(worker)) ;
logger.log('\n') ;

assert(isa(toClientDataQueue,'parallel.pool.DataQueue'));
fromClientDataQueue = parallel.pool.PollableDataQueue;
toClientDataQueue.send(fromClientDataQueue);
logger.log('Done configuring queues');

% Initialize vector of tic/toc compute time elapsed for each compute command received
computeTimes = zeros(0,1) ;

iterations_completed = 0 ;
while true        
  tic_id = tic() ;
  
  [data,ok] = fromClientDataQueue.poll();
  if ok
    assert(isstruct(data) && all(isfield(data,{'action' 'data' 'id'})));
    action = data.action;          
    logger.log('Received %s',action);
    switch action
      case 'STOP'
        break
      case 'STAT'
        sResp = struct('id',data.id,'action',action,'result',computeTimes);
        toClientDataQueue.send(sResp);
      otherwise
        error('Unrecognized action: %s',action);
    end
  else
    % continue
  end
  
  result = worker.work(logger);  % this is a row struct, with length equal to the number of views
  view_count = numel(result) ;
  for view_index = 1 : view_count ,
    result(view_index).iterations_completed = iterations_completed ;
  end
  computeTimes(end+1,1) = toc(tic_id) ;  %#ok<AGROW>
  toClientDataQueue.send(struct('id',0,'action','','result',{result}));

  logger.log('Pausing...');
  pause(pollInterval);
  logger.log('Done pausing...');
  iterations_completed = iterations_completed + 1 ;
  logger.log('iterations_completed: %d\n', iterations_completed) ;
end  % while true

status = 1;
logger.log('About to exit runPollingLoop()\n') ;

end  % function
    