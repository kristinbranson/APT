function runPollingLoop(toClientDataQueue, poller, suitcase, pollInterval, projTempDirMaybe)
% Run the polling loop; typically called via parfeval
% 
% toClientDataQueue: parallel.pool.DataQueue created by BgClient
% worker: object with method cObjMeth
% callInterval: time in seconds to wait between calls to
%   worker.(cObjMeth)

% Asserts to check input types
assert(isa(toClientDataQueue,'parallel.pool.DataQueue'));
assert(isa(poller,'BgPoller'));

% Unpack the backend suitcase
poller.restoreAfterParfeval(suitcase) ;

% Set up the log file, if called for
if isempty(projTempDirMaybe) ,
  logger = FileLogger() ;  % silently ignores .log() calls
else
  projTempDir = projTempDirMaybe{1} ;
  logFilePath = fullfile(projTempDir, 'runPollingLoop.log') ;
  logger = FileLogger(logFilePath, 'runPollingLoop') ;
end

logger.log('Inside runPollingLoop()') ;

logger.log('cObj:') ;
logger.log(formattedDisplayText(poller)) ;
logger.log('') ;

% fromClientDataQueue = parallel.pool.PollableDataQueue;
% toClientDataQueue.send(fromClientDataQueue);
%logger.log('Done configuring queue');

iterations_completed = 0 ;
while true          
  % [data,ok] = fromClientDataQueue.poll() ;
  % if ok
  %   assert(isstruct(data) && all(isfield(data,{'action' 'data' 'id'})));
  %   action = data.action;          
  %   logger.log('Received %s',action);
  %   switch action
  %     case 'STOP'
  %       break
  %     case 'STAT'
  %       sResp = struct('id',data.id,'action',action,'result',0);
  %       toClientDataQueue.send(sResp);
  %     otherwise
  %       error('Unrecognized action: %s',action);
  %   end
  % else
  %   % continue
  % end
  
  % Poll to get the status of the spawned jobs
  pollingResult = poller.poll(logger);  % this is a row struct
  % count = numel(pollingResult) ;
  % for index = 1 : count ,
  %   pollingResult(index).iterations_completed = iterations_completed ;
  % end
  toClientDataQueue.send(pollingResult);

  % Wait a bit to not poll too much
  logger.log('Pausing...\n');  % want an extra newline here
  if iterations_completed < 50
    pause(pollInterval);
  else
    pause(pollInterval*3);
  end
  logger.log('Done pausing...');

  % Log the number of iterations completed
  iterations_completed = iterations_completed + 1 ;
  logger.log('iterations_completed: %d', iterations_completed) ;
end  % while true

end  % function
    
