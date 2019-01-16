classdef BgTrackWorkerObj < handle
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
    movfiles % [nview] full paths movie being tracked
    artfctTrkfiles % [nview] full paths trkfile to be generated/output
    artfctLogfiles % [nview] cellstr of fullpaths to bsub logs
    artfctErrFiles % [nview] char fullpath to DL errfile    
    artfctPartTrkfiles % [nview] full paths to partial trkfile to be generated/output
  end
  
  methods (Abstract)
    % Same as in BgTrainWorkerObj could mixin
    tf = fileExists(obj,file)
    tf = errFileExistsNonZeroSize(obj,errFile)
    s = fileContents(obj,file)
    %[tf,warnings] = killProcess(obj)
  end
  
  methods
    function obj = BgTrackWorkerObj(mIdx,nvw,movfiles,outfiles,logfiles,dlerrfiles,partfiles)  
      assert(isequal(nvw,numel(movfiles),numel(outfiles),...
        numel(logfiles),numel(dlerrfiles)));
      
      % KB 20190115: also include locations of part track files
      if ~exist('partfiles','var'),
        partfiles = {};
      end
      
      obj.mIdx = mIdx;
      obj.nview = nvw;
      obj.movfiles = movfiles(:);
      obj.artfctTrkfiles = outfiles(:);
      obj.artfctLogfiles = logfiles(:);
      obj.artfctErrFiles = dlerrfiles(:);
      obj.artfctPartTrkfiles = partfiles(:);
      
    end    
    function sRes = compute(obj)
      % sRes: [nview] struct array      
      
      tfErrFileErr = cellfun(@obj.errFileExistsNonZeroSize,obj.artfctErrFiles,'uni',0);
      bsuberrlikely = cellfun(@obj.logFileErrLikely,obj.artfctLogfiles,'uni',0);
      
      % KB 20190115: also get locations of part track files and timestamps
      % of last modification
      partTrkFileTimestamps = nan(size(obj.artfctPartTrkfiles));
      for i = 1:numel(obj.artfctPartTrkfiles),
        tmp = dir(obj.artfctPartTrkfiles{i});
        if ~isempty(tmp),
          partTrkFileTimestamps(i) = tmp.datenum;
        end
      end
        
      sRes = struct(...
        'tfcomplete',cellfun(@obj.fileExists,obj.artfctTrkfiles,'uni',0),...
        'errFile',obj.artfctErrFiles,... % char, full path to DL err file
        'errFileExists',tfErrFileErr,... % true of errFile exists and has size>0
        'logFile',obj.artfctLogfiles,... % char, full path to Bsub logfile
        'logFileErrLikely',bsuberrlikely,... % true if bsub logfile looks like err
        'mIdx',obj.mIdx,...
        'iview',num2cell((1:obj.nview)'),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',partTrkFileTimestamps);
    end
  
    % Same as in BgTrainWorkerObj could mixin
    function printLogfiles(obj)
      logFiles = obj.artfctLogfiles;
      logFileContents = cellfun(@obj.fileContents,logFiles,'uni',0);
      BgTrainWorkerObj.printLogfilesStc(logFiles,logFileContents)
    end
    
    % Same as in BgTrainWorkerObj could mixin
    function tfLogErrLikely = logFileErrLikely(obj,file)
      tfLogErrLikely = obj.fileExists(file);
      if tfLogErrLikely
        logContents = obj.fileContents(file);
        tfLogErrLikely = ~isempty(regexpi(logContents,'exception','once'));
      end
    end

  end
  
end