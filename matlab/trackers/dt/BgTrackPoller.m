classdef BgTrackPoller < BgPoller
  % Object parfeval-copied to runPollingLoop().  It implements the poll()
  % method, which polls files (and eventually processes) on the backend to check
  % on the progress of tracking.
  
  properties
    % We could keep track of mIdx/movfile purely on the client side.
    % There are bookkeeping issues to worry about -- what if the user
    % changes a movie name, reorders/deletes movies etc while tracking is
    % running. Rather than keep track of this, we store mIdx/movfile in the
    % poller. When tracking is done, if the metadata doesn't match what
    % the client has, the client can decide what to do.

    dmcs_  % [nviews] DeepModelChainOnDisk array
    backend_  % A scalar DlBackEndClass, owned by someone else

    %nViews_ = 0
    trackType_ = 'movie'
    toTrackInfos_ = []
    
    % isexternal_ = false 
    %   % whether we are tracking movies that are part of the .lbl project
    % partFileIsTextStatus_
    %   % logical scalar. If true, partfiles are a textfile containing a single line
    %   % '<nfrmsdone>' indicating tracking status. Otherwise, partfiles are
    %   % mat-files.
  end

  properties (Dependent)
    views % unique views being tracked
    nViews  % number of unique views being tracked
    movfiles % nmovies x nviews - all movies being tracked simultaneously
    stages % unique stages being tracked
    nStages % number of unique stages being tracked
    njobs % number of jobs being tracked simultaneously
    nMovies % number of movies being tracked    
  end  

  methods
    function v = get.views(obj)
      v = obj.toTrackInfos_.views;
    end
    function v = get.nViews(obj)
      v = numel(obj.views) ;
    end
    function v = get.stages(obj)
      v = obj.toTrackInfos_.stages;
    end
    function v = get.nStages(obj)
      v = numel(obj.stages);
    end
    function v = get.njobs(obj)
      v = obj.toTrackInfos_.n;
    end
    function v = get.movfiles(obj)
      v = obj.toTrackInfos_.getMovfiles();
    end
    function v = get.nMovies(obj)
      v = size(obj.movfiles,1);
    end
  end    
    
  methods
    function obj = BgTrackPoller(trackType, dmc, backend, toTrackInfos)
      obj.trackType_ = trackType ;
      obj.dmcs_ = dmc ;
      obj.backend_ = backend ;
      obj.toTrackInfos_ = toTrackInfos ;
    end

    function sRes = poll(obj, logger)
      % Function that calls either compute() or computeList(), depending on
      % value of obj.track_type
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end
      if strcmp(obj.trackType_,'movie')
        sRes = obj.pollForMovie(logger) ;
      elseif strcmp(obj.trackType_,'list')
        sRes = obj.pollForList(logger) ;
      else
        error('Unknown track_type: %s', obj.trackType_) ;
      end
    end

    function sRes = pollForMovie(obj, logger)
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end
      
      logger.log('in bg track poller\n');
      errfiles = obj.toTrackInfos_.getErrFiles(); % njobs x 1
      logfiles = obj.toTrackInfos_.getLogFiles(); % njobs x 1
      %killfiles = obj.toTrackInfos_.getKillFiles(); % njobs x 1
      parttrkfiles = obj.toTrackInfos_.getPartTrkFiles(); % nmovies x nviews x nstages
      trkfiles = obj.toTrackInfos_.getTrkFiles(); % nmovies x nviews x nstages
      
      % KB 20190115: also get locations of part track files and timestamps
      % of last modification
      partTrkFileTimestamps = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      parttrkfileNfrmtracked = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      for i = 1:numel(parttrkfiles),
        trkfilecurr = parttrkfiles{i};
        tmp = dir(trkfilecurr);
        if isempty(tmp),
          trkfilecurr = trkfiles{i};
          tmp = dir(trkfiles{i});
        end
        if ~isempty(tmp),
          partTrkFileTimestamps(i) = tmp.datenum;
          parttrkfileNfrmtracked(i) = obj.backend_.readTrkFileStatus(trkfilecurr) ;
          logger.log('Read %d frames tracked from %s\n',parttrkfileNfrmtracked(i),trkfilecurr);
          assert(~isnan(parttrkfileNfrmtracked(i)));
        else
          logger.log('Part trk file %s and trk file %s do not exist\n',parttrkfiles{i},trkfiles{i});
        end
      end

      % nJobs = obj.toTrackInfos_.n ;
      % isRunningFromJobIndex = true([nJobs, 1]) ;  % TODO: Make this actually check if the spawned jobs are running  
      isRunningFromJobIndex = obj.backend_.isAliveFromRegisteredJobIndex('track') ;
      isRunning = obj.replicateJobs_(isRunningFromJobIndex);
      %killFileExists = cellfun(@obj.backend_.fileExists,killfiles);
      tfComplete = cellfun(@(fileName)(obj.backend_.fileExists(fileName)),trkfiles); % nmovies x njobs x nstages
      %logger.log('tfComplete = %s\n',mat2str(tfComplete(:)'));
      tfErrFileErr = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),errfiles); % njobs x 1
      logFilesExist = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),logfiles); % njobs x 1
      % bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      sRes = struct(...
        'tfComplete',num2cell(tfComplete),...
        'isRunning',num2cell(isRunning),...
        'errFile',obj.replicateJobs_(errfiles),... % char, full path to DL err file
        'errFileExists',num2cell(obj.replicateJobs_(tfErrFileErr)),... % true of errFile exists and has size>0
        'logFile',obj.replicateJobs_(logfiles),... % char, full path to Bsub logfile
        'logFileExists',num2cell(obj.replicateJobs_(logFilesExist)),...
        'iview',num2cell(repmat(1:obj.nViews,[obj.nMovies,1,obj.nStages])),...
        'movfile',repmat(obj.movfiles,[1,1,obj.nStages]),...
        'trkfile',trkfiles,...
        'parttrkfile',parttrkfiles,...
        'parttrkfileTimestamp',num2cell(partTrkFileTimestamps),...
        'parttrkfileNfrmtracked',num2cell(parttrkfileNfrmtracked),...
        'trkfileNfrmtracked',num2cell(parttrkfileNfrmtracked) ) ;
        % 'killFile',obj.replicateJobs_(killfiles),...
        % 'killFileExists',num2cell(obj.replicateJobs_(killFileExists)) );
        % 'logFileErrLikely',num2cell(obj.replicateJobs(bsuberrlikely)),... % true if bsub logfile looks like err
        % 'isexternal',obj.isexternal_... % scalar expansion
    end  % function

    function sRes = pollForList(obj, logger)
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end

      logger.log('in BgTrackPoller::pollForList()\n');
      errfiles = obj.toTrackInfos_.getErrFiles() ; % njobs x 1
      logfiles = obj.toTrackInfos_.getLogFiles() ; % njobs x 1
      % killfiles = obj.getKillFiles(); % njobs x 1
      outfiles = obj.toTrackInfos_.getListOutfiles() ; % nmovies x nviews x nstages
      
      outTrkFileTimestamps = nan(size(outfiles)); % nmovies x nviews x nstages
      for i = 1:numel(outfiles),
        trkfilecurr = outfiles{i};
        tmp = dir(trkfilecurr);
        if ~isempty(tmp) ,
          outTrkFileTimestamps(i) = tmp.datenum;
        else
          logger.log('Output file %s does not exist\n',outfiles{i});
        end
      end

      isRunning = true(obj.njobs,1) ;  % TODO: Make this actually check if the spawned jobs are running
      %killFileExists = cellfun(@obj.backend_.fileExists, killfiles) ;
      tfComplete = cellfun(@(fileName)(obj.backend_.fileExists(fileName)),outfiles); % nmovies x njobs x nstages
      logger.log('tfComplete = %s\n',mat2str(tfComplete(:)'));
      tfErrFileErr = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),errfiles); % njobs x 1
      logFilesExist = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),logfiles); % njobs x 1
      %bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      sRes = struct(...
        'tfComplete',num2cell(tfComplete),...
        'isRunning',num2cell(isRunning),...
        'errFile',errfiles,... % char, full path to DL err file
        'errFileExists',num2cell(tfErrFileErr),... % true of errFile exists and has size>0
        'logFile',logfiles,... % char, full path to Bsub logfile
        'logFileExists',num2cell(logFilesExist),...
        'iview',num2cell(1:obj.nViews),...
        'movfile','',...
        'outfile',outfiles,...
        'outfileTimestamp',num2cell(outTrkFileTimestamps) );
        % 'isexternal',obj.isexternal_... % scalar expansion
        % 'killFile',killfiles,...
        % 'killFileExists',num2cell(killFileExists),...
        %'logFileErrLikely',num2cell(bsuberrlikely),... % true if bsub logfile looks like err
    end  % function
    
    function suitcase = packParfevalSuitcase(obj)
      suitcase = obj.backend_.packParfevalSuitcase() ;
    end
   
    function restoreAfterParfeval(obj, suitcase)
      obj.backend_.restoreAfterParfeval(suitcase) ;
    end
   
    function result = replicateJobs_(obj, valueFromJobIndex)  % const method
      % Convert valueFromJobIndex, an array with nJobs elements, to result, a
      % nMovies x nViews x nStages array.  Each job has a movie index, view index,
      % and stage index associated with it.  Only the elements of result which
      % correspond to (movieIndex, viewIndex, stageIndex) triples that match those
      % of some job will be populated with meaningful values.
      movfiles = obj.movfiles ;
      nMovies = size(movfiles,1) ;
      nViews = obj.nViews ;
      stages = obj.stages ;
      nStages = numel(stages) ;
      if isempty(valueFromJobIndex) ,
        if islogical(valueFromJobIndex) , 
          result = false([nMovies, nViews, nStages]) ;
        elseif iscell(valueFromJobIndex) ,
          result = repmat({''}, [nMovies, nViews, nStages]) ;
        else
          error('BgTrackPoller:emptyArgumentOfUnhandledType', ...
                'BgTrackPoller::replicateJobs_() given empty argument of unhandled type %s', class(valueFromJobIndex)) ;
        end
      else
        % valueFromJobIndex is nonempty
        result = repmat(valueFromJobIndex(1), [nMovies, nViews, nStages]);
      end
      nJobs = obj.toTrackInfos_.n ;
      if nJobs > 0 ,
        ttis = obj.toTrackInfos_.ttis ;
        for jobIndex = 1:nJobs ,
          movidxcurr = ttis(jobIndex).getMovidx() ;
          viewscurr = ttis(jobIndex).views() ;
          stagescurr = ttis(jobIndex).stages() ;
          result(movidxcurr,viewscurr,stagescurr) = valueFromJobIndex(jobIndex) ;
        end
      end
    end  % function
      
    % function logFiles = getLogFiles(obj)
    %   logFiles = obj.toTrackInfos_.getLogFiles();
    % end
    % function errFiles = getErrFiles(obj)
    %   errFiles = obj.toTrackInfos_.getErrFiles();
    % end
    % function killFiles = getKillFiles(obj)
    %   killFiles = obj.toTrackInfos_.getKillfiles();
    % end
    % function v = getTrkFiles(obj)
    %   v = obj.toTrackInfos_.getTrkFiles();
    % end
    % function v = getPartTrkFiles(obj)
    %   v = obj.toTrackInfos_.getPartTrkFiles();
    % end
    % function v = getListOutFile(obj)
    %   v = obj.toTrackInfos_.getListOutfiles();
    % end
  end  % methods
  
end  % classdef
