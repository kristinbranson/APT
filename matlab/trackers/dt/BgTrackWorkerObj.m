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
    nMovJobs = []; % number of jobs movies are broken into. Either 1 (for 
                   % serial multimov), or .nMovies.
    nViewJobs = 1 % number of jobs views are broken into
    
    mIdx = [] % Movie index
    isexternal = false % whether we are tracking movies that are part of the lbl project
    movfiles % [nMovies x nview] full paths movie being tracked
    artfctTrkfiles % [nMovies x nviews] full paths trkfile to be generated/output
    artfctLogfiles % [nMovJobs x nViewJobs] cellstr of fullpaths to bsub logs
    artfctErrFiles % [nMovJobs x nViewJobs] char fullpath to DL errfile    
    artfctPartTrkfiles % [nMovies x nviews] full paths to partial trkfile to be generated/output
    partFileIsTextStatus % logical scalar. If true, partfiles are a 
          % textfile containing a single line '<nfrmsdone>' 
          % indicating tracking status. Otherwise, partfiles are mat-files.
    killFiles % [nMovJobs x nViewJobs]
  end
    
  methods
    function obj = BgTrackWorkerObj(varargin)
      obj@BgWorkerObj(varargin{:});
    end
    
    function initFilesOld(obj,mIdx,movfiles,outfiles,logfiles,dlerrfiles,partfiles)
      %
      %
      % movfiles: [nMovs x obj.nviews] just stored metadata ". However, the
      %   size is meaningful, see above
      
      obj.mIdx = mIdx;
      obj.nMovies = size(movfiles,1);
      assert(size(movfiles,2)==obj.nviews);
      obj.movfiles = movfiles;
      obj.artfctTrkfiles = outfiles;
      obj.artfctLogfiles = logfiles;
      obj.artfctErrFiles = dlerrfiles;
      obj.artfctPartTrkfiles = partfiles;
      obj.partFileIsTextStatus = false;
      % number of jobs views are split into. this will either be 1 or nviews
      obj.nViewJobs = size(logfiles,2);
      obj.killFiles = cell(obj.nMovies,obj.nViewJobs);
      for imov = 1:obj.nMovies,
        for ivw = 1:obj.nViewJobs,
          if obj.nMovies > 1,
            obj.killFiles{imov,ivw} = sprintf('%s.mov%d_vw%d.KILLED',logfiles{imov,ivw},imov,ivw);
          else
            obj.killFiles{ivw} = sprintf('%s.%d.KILLED',logfiles{ivw},ivw);
          end
        end
      end
    end
    
    function initFiles(obj,movfiles,trkfiles,logfiles,dlerrfiles,partfiles,isexternal)
      % 
      %
      % movfiles: [nMovs x obj.nviews] just stored metadata ". However, the
      %   size is meaningful, see above
      % trkfiles: [nMovs x nviews]. Every movie has precisely one trkfile.
      % logfiles: [nMovJobs x nViewJobs]. Every DL job has one logfile;
      %   this could represent multiple movies (across moviesets or views)
      % dlerrfiles: like logfiles
      % partfiles: like trkfiles
      % 
      
      obj.nMovies = size(movfiles,1);
      assert(size(movfiles,2)==obj.nviews);
      obj.movfiles = movfiles;
      
      szassert(trkfiles,size(movfiles));
      obj.artfctTrkfiles = trkfiles;
      
      [obj.nMovJobs,obj.nViewJobs] = size(logfiles);      
      obj.artfctLogfiles = logfiles;
      assert(obj.nMovJobs==1 || obj.nMovJobs==obj.nMovies);
      
      szassert(dlerrfiles,size(logfiles));      
      obj.artfctErrFiles = dlerrfiles;
      
      szassert(partfiles,size(trkfiles));
      obj.artfctPartTrkfiles = partfiles;
      obj.partFileIsTextStatus = false;

      obj.killFiles = cell(obj.nMovJobs,obj.nViewJobs);
      for imovjb = 1:obj.nMovJobs,
        for ivwjb = 1:obj.nViewJobs,
          if obj.nMovJob > 1, % nMovJob==nMovies
            obj.killFiles{imovjb,ivwjb} = sprintf('%s.mov%d_vwjb%d.KILLED',...
              logfiles{imovjb,ivwjb},imovjb,ivwjb);
          else
            % Could be single-movie job or multimovieserial job
            obj.killFiles{ivwjb} = sprintf('%s.%d.KILLED',logfiles{ivwjb},ivwjb);
          end
        end
      end
      
      assert(isscalar(isexternal));
      obj.isexternal = isexternal;
    end
    
    function setPartfileIsTextStatus(obj,tf)
      obj.partFileIsTextStatus = tf;
    end
      
    function sRes = compute(obj)
      % sRes: [nMovies x nviews] struct array      

      % Order important, check if job is running first. If we check after
      % looking at artifacts, job may stop in between time artifacts and 
      % isRunning are probed.
      % isRunning is nMovJobs x nViewJobs
      isRunning = obj.getIsRunning();
      if isempty(isRunning)
        isRunning = true(size(obj.killFiles));
      else
        szassert(isRunning,size(obj.killFiles));
      end
      isRunning = num2cell(isRunning);
      
      tfErrFileErr = cellfun(@obj.errFileExistsNonZeroSize,obj.artfctErrFiles,'uni',0); % nMovJobs x nViewJobs
      logFilesExist = cellfun(@obj.errFileExistsNonZeroSize,obj.artfctLogfiles,'uni',0); % nMovJobs x nViewJobs
      bsuberrlikely = cellfun(@obj.logFileErrLikely,obj.artfctLogfiles,'uni',0); % nMovJobs x nViewJobs
      
      % KB 20190115: also get locations of part track files and timestamps
      % of last modification
      partTrkFileTimestamps = nan(size(obj.artfctPartTrkfiles)); % nMovies x nviews
      parttrkfileNfrmtracked = nan(size(obj.artfctPartTrkfiles)); % nMovies x nviews
      for i = 1:numel(obj.artfctPartTrkfiles),
        parttrkfile = obj.artfctPartTrkfiles{i};
        tmp = dir(parttrkfile);
        if ~isempty(tmp),
          partTrkFileTimestamps(i) = tmp.datenum;
          if obj.partFileIsTextStatus
            tmp = obj.fileContents(parttrkfile);
            PAT = '(?<numfrmstrked>[0-9]+)';
            toks = regexp(tmp,PAT,'names');
            if ~isempty(toks)
              parttrkfileNfrmtracked(i) = str2double(toks.numfrmstrked);
            end
          end
        end
      end
        
      killFileExists = false(size(obj.killFiles)); % nMovJobs x nViewJobs
      for i = 1:numel(obj.killFiles),
        killFileExists(i) = obj.fileExists(obj.killFiles{i});
      end
      
      if ~iscell(obj.mIdx),
        mIdx = {obj.mIdx};
      else
        mIdx = obj.mIdx;
      end
      % number of views handled per job
      nViewsPerJob = obj.nviews / obj.nViewJobs;
      nMovsetsPerJob = obj.nMovies / obj.nMovJobs;
      % Recall some modalities: 
      % 1. Single movieset, one job per view => .nMovJobs=1, .nViewJobs=.nviews
      % 2. Single movieset, one job for all views => .nMovJobs=1, nViewJobs=1
      % 3. Multiple movies, one job per movie, single view => .nMovJobs=.nMovies, .nViewJobs=1
      % 4. Multiple movies, one job per movie, multi view => not sure we do this?
      % 5. Multiple movies, one job for all movies, single view => .nMovJobs=1, .nViewJobs=1
      
      % nMovies x nviews
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      sRes = struct(...
        'tfComplete',cellfun(@obj.fileExists,obj.artfctTrkfiles,'uni',0),...
        'isRunning',repmat(isRunning,[nMovsetsPerJob,nViewsPerJob]),...
        'errFile',repmat(obj.artfctErrFiles,[nMovsetsPerJob,nViewsPerJob]),... % char, full path to DL err file
        'errFileExists',repmat(tfErrFileErr,[nMovsetsPerJob,nViewsPerJob]),... % true of errFile exists and has size>0
        'logFile',repmat(obj.artfctLogfiles,[nMovsetsPerJob,nViewsPerJob]),... % char, full path to Bsub logfile
        'logFileExists',repmat(logFilesExist,[nMovsetsPerJob,nViewsPerJob]),...
        'logFileErrLikely',repmat(bsuberrlikely,[nMovsetsPerJob,nViewsPerJob]),... % true if bsub logfile looks like err
        'mIdx',mIdx{1},...
        'iview',num2cell(repmat(1:obj.nviews,[obj.nMovies,1])),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',num2cell(partTrkFileTimestamps),...
        'killFile',repmat(obj.killFiles,[nMovsetsPerJob,nViewsPerJob]),...
        'killFileExists',repmat(num2cell(killFileExists),[nMovsetsPerJob,nViewsPerJob]),...
        'isexternal',obj.isexternal... % scalar expansion
        );
      if obj.partFileIsTextStatus
        parttrkfileNfrmtracked = num2cell(parttrkfileNfrmtracked);
        [sRes.parttrkfileNfrmtracked] = deal(parttrkfileNfrmtracked{:});
        [sRes.trkfileNfrmtracked] = deal(parttrkfileNfrmtracked{:});
      end
    end
    
    function reset(obj)
      % AL: TODO This seems strange/dangerous, the files may not be on a 
      % local filesys...
      
      killFiles = obj.getKillFiles(); %#ok<PROP>
      for i = 1:numel(killFiles), %#ok<PROP>
        if exist(killFiles{i},'file'), %#ok<PROP>
          delete(killFiles{i}); %#ok<PROP>
        end
      end
      
      logFiles = obj.getLogFiles();
      for i = 1:numel(logFiles),
        if exist(logFiles{i},'file'),
          delete(logFiles{i});
        end
      end
      
      errFiles = obj.getErrFile();
      for i = 1:numel(errFiles),
        if exist(errFiles{i},'file'),
          delete(errFiles{i});
        end
      end
      
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