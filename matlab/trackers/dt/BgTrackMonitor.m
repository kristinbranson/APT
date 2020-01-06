classdef BgTrackMonitor < BgMonitor 
  %
  % A BgTrackMonitor:
  % 1. Is a BGClient/BGWorker pair comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of a track. For
  %     debugging/testing this can be done from the client machine.
  % 2. In general knows how to communicate with the bg tracking process.
  % 
  %
  % BgTrackMonitor is intended to be subclassed.
  %
  % BgTrackMonitor does NOT know how to spawn tracking jobs.
  %
  % See also prepare() method comments for related info.
  
  methods
    
    function obj = BgTrackMonitor(varargin)
      obj@BgMonitor(varargin{:});
      obj.bgContCallInterval = 20; %secs
      obj.processName = 'track';      
    end
    
  end
end