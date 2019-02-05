classdef BgTrackWorkerObj < BgWorkerObj
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
    movfiles % [nview] full paths movie being tracked
    artfctTrkfiles % [nviews] full paths trkfile to be generated/output
    artfctLogfiles % [nviews] cellstr of fullpaths to bsub logs
    artfctErrFiles % [nviews] char fullpath to DL errfile    
    artfctPartTrkfiles % [nviews] full paths to partial trkfile to be generated/output
    killFiles % [nviews]
  end
    
  methods
    function obj = BgTrackWorkerObj(varargin)
      obj@BgWorkerObj(varargin{:});
    end
    function initFiles(obj,mIdx,movfiles,outfiles,logfiles,dlerrfiles,partfiles)
      obj.mIdx = mIdx;
      obj.movfiles = movfiles(:);
      obj.artfctTrkfiles = outfiles(:);
      obj.artfctLogfiles = logfiles(:);
      obj.artfctErrFiles = dlerrfiles(:);
      obj.artfctPartTrkfiles = partfiles(:);
      for ivw = 1:obj.nviews,
        obj.killFiles{ivw} = sprintf('%s.%d.KILLED',logfiles{ivw},ivw);
      end
      obj.killFiles = obj.killFiles(:);
    end
    function sRes = compute(obj)
      % sRes: [nviews] struct array      
      
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
        
      killFileExists = false(1,obj.nviews);
      for ivw = 1:obj.nviews,
        killFileExists(ivw) = obj.fileExists(obj.killFiles{ivw});
      end
      
      if iscell(obj.mIdx),
        mIdx = {obj.mIdx};
      else
        mIdx = obj.mIdx;
      end
      sRes = struct(...
        'tfComplete',cellfun(@obj.fileExists,obj.artfctTrkfiles,'uni',0),...
        'errFile',obj.artfctErrFiles,... % char, full path to DL err file
        'errFileExists',tfErrFileErr,... % true of errFile exists and has size>0
        'logFile',obj.artfctLogfiles,... % char, full path to Bsub logfile
        'logFileErrLikely',bsuberrlikely,... % true if bsub logfile looks like err
        'mIdx',mIdx,...
        'iview',num2cell((1:obj.nviews)'),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',num2cell(partTrkFileTimestamps),...
        'killFile',obj.killFiles(:),...
        'killFileExists',num2cell(killFileExists(:)));
    end
    function logFiles = getLogFiles(obj)
      logFiles = unique(obj.artfctLogfiles);
    end
    function killFiles = getKillFiles(obj)
      killFiles = obj.killFiles;
    end

  end
  
end