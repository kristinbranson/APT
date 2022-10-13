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

    nviews = 0;
    totrackinfos = [];
    
    isexternal = false % whether we are tracking movies that are part of the lbl project
    partFileIsTextStatus % logical scalar. If true, partfiles are a 
          % textfile containing a single line '<nfrmsdone>' 
          % indicating tracking status. Otherwise, partfiles are mat-files.
  end
  properties (Dependent)
    views % unique views being tracked
    movfiles % nmovies x nviews - all movies being tracked simultaneously
    stages % unique stages being tracked
    nStages % number of unique stages being tracked
    njobs % number of jobs being tracked simultaneously
    nMovies % number of movies being tracked    
%     artfctTrkfiles % [nMovies x nviews x nStages] full paths trkfile to be generated/output
%     artfctLogfiles % [nMovJobs x nViewJobs] cellstr of fullpaths to bsub logs
%     artfctErrFiles % [nMovJobs x nViewJobs] char fullpath to DL errfile    
%     artfctPartTrkfiles % [nMovies x nviews x nStages] full paths to partial trkfile to be generated/output
%     killFiles % [nMovJobs x nViewJobs]


  end  
  methods
    function v = get.views(obj)
      if isempty(obj.totrackinfos),
        v = [];
      else
        v = unique(cat(2,obj.totrackinfos.views));
      end
    end
    function v = get.stages(obj)
      if isempty(obj.totrackinfos),
        v = [];
      else
        v = unique(cat(2,obj.totrackinfos.stages));
      end
    end
    function v = get.nStages(obj)
      v = numel(obj.stages);
    end
    function v = get.njobs(obj)
      v = numel(obj.totrackinfos);
    end
    function v = get.movfiles(obj)
      v = ToTrackInfo.mergeGetMovfiles(obj.totrackinfos);
    end
    function v = get.nMovies(obj)
      v = size(obj.movfiles,1);
    end
  end
    
    
  methods
    function obj = BgTrackWorkerObj(nviews,varargin)
      obj@BgWorkerObj(varargin{:});
      if nargin > 1,
        obj.nviews = nviews;
      end
    end

    function initFiles(obj,totrackinfos)
      obj.totrackinfos = totrackinfos;
      obj.nviews = numel(obj.totrackinfos);
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

    function vout = replicateJobs(obj,vin)

      vout = repmat(vin(1),[obj.nMovies,obj.nviews,obj.nStages]);
      for i = 1:numel(obj.totrackinfos),
        movidxcurr = obj.totrackinfos(i).getMovidx();
        viewscurr = obj.totrackinfos(i).views();
        stagescurr = obj.totrackinfos(i).stages();
        vout(movidxcurr,viewscurr,stagescurr) = vin(i,:);        
      end

    end
      
    function sRes = compute(obj)
      % sRes: [nMovies x nviews x nStages] struct array      

      % Order important, check if job is running first. If we check after
      % looking at artifacts, job may stop in between time artifacts and 
      % isRunning are probed.
      % isRunning does not seem to do anything right now!!
      isRunning = obj.getIsRunning();
      if isempty(isRunning)
        isRunning = true(obj.njobs,1);
      else
        szassert(isRunning,[obj.njobs,1]);
      end
      
      errfiles = obj.getErrFile(); % njobs x 1
      logfiles = obj.getErrFile(); % njobs x 1
      killfiles = obj.getKillFiles(); % njobs x 1
      parttrkfiles = obj.getPartTrkFile(); % nmovies x nviews x nstages
      trkfiles = obj.getTrkFile(); % nmovies x nviews x nstages
      
      % KB 20190115: also get locations of part track files and timestamps
      % of last modification
      partTrkFileTimestamps = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      parttrkfileNfrmtracked = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      for i = 1:numel(parttrkfiles),
        parttrkfile = parttrkfiles{i};
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



      isRunning = obj.replicateJobs(isRunning);
      killFileExists = cellfun(@obj.fileExists,killfiles);
      tfComplete = cellfun(@obj.fileExists,trkfiles);
      tfErrFileErr = cellfun(@obj.errFileExistsNonZeroSize,errfiles); % njobs x 1
      logFilesExist = cellfun(@obj.errFileExistsNonZeroSize,logfiles); % njobs x 1
      bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      sRes = struct(...
        'tfComplete',num2cell(obj.replicateJobs(tfComplete)),...
        'isRunning',num2cell(obj.replicateJobs(isRunning)),...
        'errFile',obj.replicateJobs(errfiles),... % char, full path to DL err file
        'errFileExists',num2cell(obj.replicateJobs(tfErrFileErr)),... % true of errFile exists and has size>0
        'logFile',obj.replicateJobs(logfiles),... % char, full path to Bsub logfile
        'logFileExists',num2cell(obj.replicateJobs(logFilesExist)),...
        'logFileErrLikely',num2cell(obj.replicateJobs(bsuberrlikely)),... % true if bsub logfile looks like err
        'iview',num2cell(repmat(1:obj.nviews,[obj.nMovies,1,obj.nStages])),...
        'movfile',repmat(obj.movfiles,[1,1,obj.nStages]),...
        'trkfile',trkfiles,...
        'parttrkfile',parttrkfiles,...
        'parttrkfileTimestamp',num2cell(partTrkFileTimestamps),...
        'killFile',obj.replicateJobs(killfiles),...
        'killFileExists',num2cell(obj.replicateJobs(killFileExists)),...
        'isexternal',obj.isexternal... % scalar expansion
        );
      if obj.partFileIsTextStatus
        parttrkfileNfrmtracked = num2cell(parttrkfileNfrmtracked);
        [sRes.parttrkfileNfrmtracked] = deal(parttrkfileNfrmtracked{:});
        [sRes.trkfileNfrmtracked] = deal(parttrkfileNfrmtracked{:});
      end
    end
    
    function reset(obj) %#ok<MANU> 
      % For BgTrackWorkerObjs, this appears to only be called at 
      % BgWorkerObj-construction time. So not a useful meth currently.
      
%       killFiles = obj.getKillFiles(); 
%       assert(isempty(killFiles));
%       for i = 1:numel(killFiles), %#ok<PROP>
%         if exist(killFiles{i},'file'), %#ok<PROP>
%           delete(killFiles{i}); %#ok<PROP>
%         end
%       end
      
%       logFiles = obj.getLogFiles();
%       assert(isempty(logFiles));
%       for i = 1:numel(logFiles),
%         if exist(logFiles{i},'file'),
%           delete(logFiles{i});
%         end
%       end
      
%       errFiles = obj.getErrFile();
%       assert(isempty(errFiles));
%       for i = 1:numel(errFiles),
%         if exist(errFiles{i},'file'),
%           delete(errFiles{i});
%         end
%       end
      
    end
    
    function logFiles = getLogFiles(obj)
      logFiles = cell(numel(obj.totrackinfos),1);
      for i = 1:numel(obj.totrackinfos),
        logFiles{i} = obj.totrackinfos(i).getLogfile();
      end
    end
    function errFiles = getErrFile(obj)
      errFiles = cell(numel(obj.totrackinfos),1);
      for i = 1:numel(obj.totrackinfos),
        errFiles{i} = obj.totrackinfos(i).getErrfile();
      end
    end
    function killFiles = getKillFiles(obj)
      killFiles = cell(numel(obj.totrackinfos),1);
      for i = 1:numel(obj.totrackinfos),
        killFiles{i} = obj.totrackinfos(i).getKillfile();
      end
    end
    function v = getTrkFile(obj)
      v = ToTrackInfo.mergeGetTrkfiles(obj.totrackinfos);
    end
    function v = getPartTrkFile(obj)
      v = ToTrackInfo.mergeGetParttrkfiles(obj.totrackinfos);
    end
  end
  
end