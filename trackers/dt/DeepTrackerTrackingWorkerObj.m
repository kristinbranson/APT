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
    artfctTrkfiles % [nview] full paths trkfile to be generated/output
    artfctBsubLogs % [nview] cellstr of fullpaths to bsub logs
    artfctErrFile % char fullpath to DL errfile    
  end
  
  methods
    function obj = DeepTrackerTrackingWorkerObj(mIdx,nvw,movfiles,...
        outfiles,bsublogfiles,dlerrfile)
      
      assert(isequal(nvw,numel(movfiles),numel(outfiles),...
        numel(bsublogfiles))); 
      
      obj.mIdx = mIdx;
      obj.nview = nvw;
      obj.movfiles = movfiles(:);
      obj.artfctTrkfiles = outfiles(:);
      obj.artfctBsubLogs = bsublogfiles(:);
      obj.artfctErrFile = dlerrfile;
    end    
    function sRes = compute(obj)
      % sRes: [nview] struct array      
      
      tfErrFileErr = ...
        DeepTrackerTrainingWorkerObj.errFileExistsNonZeroSize(obj.artfctErrFile);
      
      bsuberrlikely = cellfun(@(x)exist(x,'file')>0 && ...
        DeepTrackerTrainingWorkerObj.parseBsubLogFile(x),obj.artfctBsubLogs,'uni',0);
      
      sRes = struct(...
        'tfcomplete',cellfun(@(x)exist(x,'file')>0,obj.artfctTrkfiles,'uni',0),...
        'errFile',obj.artfctErrFile,... % char, full path to DL err file
        'errFileExists',tfErrFileErr,... % true of errFile exists and has size>0
        'bsubLogFile',obj.artfctBsubLogs,... % char, full path to Bsub logfile
        'bsubLogFileErrLikely',bsuberrlikely,... % true if bsub logfile looks like err
        'mIdx',obj.mIdx,...
        'iview',num2cell((1:obj.nview)'),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles);
    end
  end
  
end