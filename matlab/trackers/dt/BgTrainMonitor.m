classdef BgTrainMonitor < BgMonitor
  % BGTrainMonitor
  %
  % A BGTrainMonitor is:
  % 1. A BGClient/BGWorker pair comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of a train. For
  %     debugging/testing this can be done from the client machine.
  % 2. A client-side TrainingMonitorViz object that visualizes training 
  % progress sent back from the BGWorker
  % 3. Custom actions performed when training is complete
  %
  % BGTrainMonitor does NOT know how to spawn training jobs but will know
  % how to (attempt to) kill them. For debugging, you can manually spawn 
  % jobs and monitor them with BgTrainMonitor.
  %
  % BGTrainMonitor does NOT know how to probe the detailed state of the
  % train eg on disk. That is BGTrainWorkerObj's domain.
  %
  % So BGTrainMonitor is a connector/manager obj that runs the worker 
  % (knows how to poll the filesystem in detail) in the background and 
  % connects it with a Monitor.
  %
  % See also prepare() method comments for related info.
  
  methods
    
    function obj = BgTrainMonitor(varargin)
      obj@BgMonitor(varargin{:});
      obj.bgContCallInterval = 30; %secs
      obj.processName = 'train';
    end
    
  end
end