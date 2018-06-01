classdef DeepTrackerTrackingWorkerObj < handle
  % Object deep copied onto BG Tracking worker. To be used with
  % BGWorkerContinuous
  %
  % Responsibilities:
  % - Poll filesystem for tracking output files
  
  properties
    % We could keep track of mIdx/movfile purely on the client side.
    % There are bookkeeping issues to worry about -- what if the user
    % changes a movie name, reorders/deletes movies etc while tracking is
    % running. Rather than keep track of this, we store mIdx/movfile in the
    % worker. When tracking is done, if the metadata doesn''t match what
    % the client has, the client can decide what to do.
    
    mIdx % Movie index
    nview % number of views
    movfiles % full paths movie being tracked    
    trkfiles % full paths trkfile to be generated/output
  end
  
  methods
    function obj = DeepTrackerTrackingWorkerObj(mIdx,nvw,movfiles,outfiles)
      assert(isequal(nvw,numel(movfiles),numel(outfiles)));
      obj.mIdx = mIdx;
      obj.nview = nvw;
      obj.movfiles = movfiles(:);
      obj.trkfiles = outfiles(:);
    end    
    function sRes = compute(obj)
      % sRes: [nview] struct array
      
      sRes = struct(...
        'tfcomplete',cellfun(@(x)exist(x,'file')>0,obj.trkfiles,'uni',0),...
        'mIdx',obj.mIdx,...
        'iview',num2cell((1:obj.nview)'),...
        'movfile',obj.movfiles,...
        'trkfile',obj.trkfiles);
    end
  end
  
end