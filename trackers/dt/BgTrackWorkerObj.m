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
    
    nMovies = 1 % number of movies being tracked
    mIdx % Movie index
    movfiles % [nMovies x nview] full paths movie being tracked
    artfctTrkfiles % [nMovies x nviews] full paths trkfile to be generated/output
    artfctLogfiles % [nMovies x nviews] cellstr of fullpaths to bsub logs
    artfctErrFiles % [nMovies x nviews] char fullpath to DL errfile    
    artfctPartTrkfiles % [nMovies x nviews] full paths to partial trkfile to be generated/output
    killFiles % [nMovies x nviews]
  end
    
  methods
    function obj = BgTrackWorkerObj(varargin)
      obj@BgWorkerObj(varargin{:});
    end
    function initFiles(obj,mIdx,movfiles,outfiles,logfiles,dlerrfiles,partfiles)
      obj.mIdx = mIdx;
      obj.nMovies = size(movfiles,1);
      assert(size(movfiles,2)==obj.nviews);
      obj.movfiles = movfiles;
      obj.artfctTrkfiles = outfiles;
      obj.artfctLogfiles = logfiles;
      obj.artfctErrFiles = dlerrfiles;
      obj.artfctPartTrkfiles = partfiles;
      % number of jobs views are split into. this will either be 1 or nviews
      nViewJobs = size(logfiles,2);
      obj.killFiles = cell(obj.nMovies,nViewJobs);
      for imov = 1:obj.nMovies,
        for ivw = 1:nViewJobs,
          if obj.nMovies > 1,
            obj.killFiles{imov,ivw} = sprintf('%s.mov%d_vw%d.KILLED',logfiles{imov,ivw},imov,ivw);
          else
            obj.killFiles{ivw} = sprintf('%s.%d.KILLED',logfiles{ivw},ivw);
          end
        end
      end
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
        
      killFileExists = false(size(obj.killFiles));
      for i = 1:numel(obj.killFiles),
        killFileExists(i) = obj.fileExists(obj.killFiles{i});
      end
      
      if ~iscell(obj.mIdx),
        mIdx = {obj.mIdx};
      else
        mIdx = obj.mIdx;
      end
      % number of jobs views are split up into
      nViewJobs = size(obj.killFiles,2);
      % number of views handled per job
      nViewsPerJob = obj.nviews / nViewJobs;
      
      sRes = struct(...
        'tfComplete',cellfun(@obj.fileExists,obj.artfctTrkfiles,'uni',0),...
        'errFile',repmat(obj.artfctErrFiles,[1,nViewsPerJob]),... % char, full path to DL err file
        'errFileExists',repmat(tfErrFileErr,[1,nViewsPerJob]),... % true of errFile exists and has size>0
        'logFile',repmat(obj.artfctLogfiles,[1,nViewsPerJob]),... % char, full path to Bsub logfile
        'logFileErrLikely',repmat(bsuberrlikely,[1,nViewsPerJob]),... % true if bsub logfile looks like err
        'mIdx',mIdx{1},...
        'iview',num2cell(repmat(1:obj.nviews,[obj.nMovies,1])),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',num2cell(partTrkFileTimestamps),...
        'killFile',repmat(obj.killFiles,[1,nViewsPerJob]),...
        'killFileExists',repmat(num2cell(killFileExists),[1,nViewsPerJob]));
    end
    function logFiles = getLogFiles(obj)
      logFiles = unique(obj.artfctLogfiles(:));
    end
    function errFiles = getErrFile(obj)
      errFiles = unique(obj.artfctErrFiles(:));
    end
    function killFiles = getKillFiles(obj)
      killFiles = obj.killFiles(:);
    end

  end
  
end