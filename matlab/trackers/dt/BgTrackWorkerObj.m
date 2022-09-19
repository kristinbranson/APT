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
    % worker. When tracking is done, if the metadata doesn't match what
    % the client has, the client can decide what to do.

    totrackinfos = [];
    
    nMovies = 1 % number of movies being tracked    
    nMovJobs = []; % number of jobs movies are broken into. Either 1 (for 
                   % serial multimov), or .nMovies.
    nviews = 1; % number of views being tracked                   
    nViewJobs = 1 % number of jobs views are broken into
    % nStages (see Dependent, below) % number of stages. Equals 2 for
                                   % top-down tracking

    % 2-stage note. Currently, 2-stage tracking always runs serially and 
    % for both stages. The output of stage1 is required for stage2;
    % tracking just stage1, or just tracking stage2 (with a pre-generated 
    % stage1 output) is possible but not a crucial workflow.                                   
    
    mIdx = [] % Movie index
    isexternal = false % whether we are tracking movies that are part of the lbl project
    movfiles % [nMovies x nview] full paths movie being tracked
    partFileIsTextStatus % logical scalar. If true, partfiles are a 
          % textfile containing a single line '<nfrmsdone>' 
          % indicating tracking status. Otherwise, partfiles are mat-files.
  end
  properties (Dependent)
    views
    nStages
    artfctTrkfiles % [nMovies x nviews x nStages] full paths trkfile to be generated/output
    artfctLogfiles % [nMovJobs x nViewJobs] cellstr of fullpaths to bsub logs
    artfctErrFiles % [nMovJobs x nViewJobs] char fullpath to DL errfile    
    artfctPartTrkfiles % [nMovies x nviews x nStages] full paths to partial trkfile to be generated/output
    killFiles % [nMovJobs x nViewJobs]


  end  
  methods
    function v = get.views(obj)
      v = unique(cat(2,obj.totrackinfo.views));
    end
    function v = get.nStages(obj)
      v = size(obj.artfctTrkfiles,3);
    end
  end
    
    
  methods
    function obj = BgTrackWorkerObj(nviews,varargin)
      obj@BgWorkerObj(varargin{:});
      if nargin >= 1,
        obj.nviews = nviews;
      end
    end

    function initFiles(obj,totrackinfos)
      obj.totrackinfos = totrackinfos;
    end
    
%     function initFiles(obj,movfiles,trkfiles,logfiles,dlerrfiles,...
%                             partfiles,isexternal)
%       % 
%       %
%       % movfiles: [nMovs x obj.nviews] just stored metadata ". However, the
%       %   size is meaningful, see above
%       % trkfiles: [nMovs x nviews x nsets].
%       % logfiles: [nMovJobs x nViewJobs]. Every DL job has one logfile;
%       %   this could represent multiple movies (across moviesets or views)
%       % dlerrfiles: like logfiles
%       % partfiles: like trkfiles
%       % 
%       
%       obj.nMovies = size(movfiles,1);
%       %assert(size(movfiles,2)==obj.nviews);
%       obj.movfiles = movfiles;
%             
%       %szassert(trkfiles,size(movfiles));
%       sztrk = size(trkfiles);
%       assert(isequal(sztrk(1:2),size(movfiles)));
%       obj.artfctTrkfiles = trkfiles;
%       
%       [obj.nMovJobs,obj.nViewJobs] = size(logfiles);      
%       obj.artfctLogfiles = logfiles;
%       assert(obj.nMovJobs==1 || obj.nMovJobs==obj.nMovies);
%       
%       szassert(dlerrfiles,size(logfiles));      
%       obj.artfctErrFiles = dlerrfiles;
%       
%       szassert(partfiles,size(trkfiles));
%       obj.artfctPartTrkfiles = partfiles;
%       obj.partFileIsTextStatus = false;
% 
%       obj.killFiles = cell(obj.nMovJobs,obj.nViewJobs);
%       for imovjb = 1:obj.nMovJobs,
%         for ivwjb = 1:obj.nViewJobs,
%           if obj.nMovJobs > 1, % nMovJob==nMovies
%             obj.killFiles{imovjb,ivwjb} = sprintf('%s.mov%d_vwjb%d.KILLED',...
%               logfiles{imovjb,ivwjb},imovjb,ivwjb);
%           else
%             % Could be single-movie job or multimovieserial job
%             obj.killFiles{ivwjb} = sprintf('%s.%d.KILLED',logfiles{ivwjb},ivwjb);
%           end
%         end
%       end
%       
%       assert(isscalar(isexternal));
%       obj.isexternal = isexternal;
%     end
    
    function setPartfileIsTextStatus(obj,tf)
      obj.partFileIsTextStatus = tf;
    end
      
    function sRes = compute(obj)
      % sRes: [nMovies x nviews x nStages] struct array      

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
      partTrkFileTimestamps = nan(size(obj.artfctPartTrkfiles)); % nMovies x nviews x nStages
      parttrkfileNfrmtracked = nan(size(obj.artfctPartTrkfiles)); % nMovies x nviews x nStages
      for i = 1:numel(obj.artfctPartTrkfiles),
        parttrkfile = obj.artfctPartTrkfiles{i};
        tmp = dir(parttrkfile);
        if ~isempty(tmp),
          partTrkFileTimestamps(i) = tmp.datenum;
          if obj.partFileIsTextStatus
            tmp = obj.fileContents(parttrkfile);
            PAT = '(?<numfrmstrked>[0-9]+)';
            toks = regexp(tmp,PAT,'names','once');
            if ~isempty(toks)
              if iscell(toks),
                toks = toks{1};
              end
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
        midx = {obj.mIdx};
      else
        midx = obj.mIdx;
      end
      % number of views handled per job
      nVwPerJob = obj.nViewsPerJob;
      nMovsetsPerJob = obj.nMovies / obj.nMovJobs;
      % Recall some modalities: 
      % 1. Single movieset, one job per view => .nMovJobs=1, .nViewJobs=.nviews
      % 2. Single movieset, one job for all views => .nMovJobs=1, nViewJobs=1
      % 3. Multiple movies, one job per movie, single view => .nMovJobs=.nMovies, .nViewJobs=1
      % 4. Multiple movies, one job per movie, multi view => not sure we do this?
      % 5. Multiple movies, one job for all movies, single view => .nMovJobs=1, .nViewJobs=1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      nStg = obj.nStages;
      sRes = struct(...
        'tfComplete',cellfun(@obj.fileExists,obj.artfctTrkfiles,'uni',0),...
        'isRunning',repmat(isRunning,[nMovsetsPerJob,nVwPerJob,nStg]),...
        'errFile',repmat(obj.artfctErrFiles,[nMovsetsPerJob,nVwPerJob,nStg]),... % char, full path to DL err file
        'errFileExists',repmat(tfErrFileErr,[nMovsetsPerJob,nVwPerJob,nStg]),... % true of errFile exists and has size>0
        'logFile',repmat(obj.artfctLogfiles,[nMovsetsPerJob,nVwPerJob,nStg]),... % char, full path to Bsub logfile
        'logFileExists',repmat(logFilesExist,[nMovsetsPerJob,nVwPerJob,nStg]),...
        'logFileErrLikely',repmat(bsuberrlikely,[nMovsetsPerJob,nVwPerJob,nStg]),... % true if bsub logfile looks like err
        'mIdx',midx{1},...
        'iview',num2cell(repmat(1:obj.nviews,[obj.nMovies,1,nStg])),...
        'movfile',repmat(obj.movfiles,[1,1,nStg]),...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',num2cell(partTrkFileTimestamps),...
        'killFile',repmat(obj.killFiles,[nMovsetsPerJob,nVwPerJob,nStg]),...
        'killFileExists',repmat(num2cell(killFileExists),[nMovsetsPerJob,nVwPerJob,nStg]),...
        'isexternal',obj.isexternal... % scalar expansion
        );
      if obj.partFileIsTextStatus
        parttrkfileNfrmtracked = num2cell(parttrkfileNfrmtracked);
        [sRes.parttrkfileNfrmtracked] = deal(parttrkfileNfrmtracked{:});
        [sRes.trkfileNfrmtracked] = deal(parttrkfileNfrmtracked{:});
      end
    end
    
    function reset(obj)
      % For BgTrackWorkerObjs, this appears to only be called at 
      % BgWorkerObj-construction time. So not a useful meth currently.
      
      killFiles = obj.getKillFiles(); %#ok<PROP>
      assert(isempty(killFiles));
%       for i = 1:numel(killFiles), %#ok<PROP>
%         if exist(killFiles{i},'file'), %#ok<PROP>
%           delete(killFiles{i}); %#ok<PROP>
%         end
%       end
      
      logFiles = obj.getLogFiles();
      assert(isempty(logFiles));
%       for i = 1:numel(logFiles),
%         if exist(logFiles{i},'file'),
%           delete(logFiles{i});
%         end
%       end
      
      errFiles = obj.getErrFile();
      assert(isempty(errFiles));
%       for i = 1:numel(errFiles),
%         if exist(errFiles{i},'file'),
%           delete(errFiles{i});
%         end
%       end
      
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