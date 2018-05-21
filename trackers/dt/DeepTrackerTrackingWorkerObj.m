classdef DeepTrackerTrackingWorkerObj < handle
  % Object deep copied onto BG Tracking worker. To be used with
  % BGWorkerContinuous
  %
  % Responsibilities:
  % - Poll filesystem for tracking output file
  
  %
  % TODO: multiview
  
  properties
    % We could keep track of mIdx/movfile purely on the client side.
    % There are bookkeeping issues to worry about -- what if the user
    % changes a movie name, reorders/deletes movies etc while tracking is
    % running. Rather than keep track of this, we store mIdx/movfile in the
    % worker. When tracking is done, if the metadata doesn''t match what
    % the client has, the client can decide what to do.
    
    mIdx % Movie index
    iview
    movfile % full path movie being tracked    
    trkfile % full path trkfile to be generated/output
  end
  
  methods
    function obj = DeepTrackerTrackingWorkerObj(mIdx,iview,movfile,outfile)
      obj.mIdx = mIdx;
      obj.iview = iview;
      obj.movfile = movfile;
      obj.trkfile = outfile;
    end    
    function sRes = compute(obj)
      sRes = struct(...
        'tfcomplete',exist(obj.trkfile,'file')>0,...
        'mIdx',obj.mIdx,...
        'iview',obj.iview,...
        'movfile',obj.movfile,...
        'trkfile',obj.trkfile);
    end
  end
  
end