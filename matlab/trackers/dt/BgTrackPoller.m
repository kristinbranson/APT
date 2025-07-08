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
    resultSize % number of results
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
    function sz = get.resultSize(obj)
      if strcmp(obj.trackType_,'list'),
        sz = [obj.njobs,1,1];
      elseif strcmp(obj.trackType_,'movie'),
        sz = [obj.nMovies,obj.nViews,obj.nStages];
      else
        sz = [];
      end
    end
  end    
    
  methods
    function obj = BgTrackPoller(trackType, dmc, backend, toTrackInfos)
      assert(strcmp(trackType,'movie') || strcmp(trackType,'list')) ;
      assert(isa(dmc, 'DeepModelChainOnDisk')) ;
      assert(isa(backend, 'DLBackEndClass') && isscalar(backend)) ;
      assert(isa(toTrackInfos, 'ToTrackInfoSet')) ;

      obj.trackType_ = trackType ;
      obj.dmcs_ = dmc ;
      obj.backend_ = backend ;
      obj.toTrackInfos_ = toTrackInfos ;
    end

    function result = poll(obj, logger)
      % Function that calls either compute() or computeList(), depending on
      % value of obj.track_type
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end
      if strcmp(obj.trackType_,'movie')
        result = obj.pollForMovie(logger) ;
      elseif strcmp(obj.trackType_,'list')
        result = obj.pollForList(logger) ;
      else
        error('Unknown track_type: %s', obj.trackType_) ;
      end
      assert(isstruct(result) && isscalar(result)) ;
    end

    function result = pollForMovie(obj, logger)
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end
      
      logger.log('in bg track poller\n');
      errfiles = obj.toTrackInfos_.getErrFiles(); % njobs x 1
      logfiles = obj.toTrackInfos_.getLogFiles(); % njobs x 1
      %killfiles = obj.toTrackInfos_.getKillFiles(); % njobs x 1
      parttrkfiles = obj.toTrackInfos_.getPartTrkFiles(); % nmovies x nviews x nstages, local file names
      trkfiles = obj.toTrackInfos_.getTrkFiles(); % nmovies x nviews x nstages, local file names
      
      % KB 20190115: also get locations of part track files and timestamps
      % of last modification
      partTrkFileTimestamps = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      parttrkfileNfrmtracked = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      trackedFrameCountSource = nan(size(parttrkfiles)); % nmovies x nviews x nstages      
      for i = 1:numel(parttrkfiles),
        parttrkfilecurr = parttrkfiles{i};
        if obj.backend_.fileExists(parttrkfilecurr) ,
          partTrkFileTimestamps(i) = obj.backend_.fileModTime(parttrkfilecurr) ;
          parttrkfileNfrmtracked(i) = obj.backend_.readTrkFileStatus(parttrkfilecurr) ;
          trackedFrameCountSource(i) = 0.5 ;  % got it from the partial file
          logger.log('Read %d frames tracked from %s\n',parttrkfileNfrmtracked(i),parttrkfilecurr);
        else
          % If the partial trk file does not exist, try to get info from the full trk file.
          trkfilecurr = trkfiles{i} ;
          if obj.backend_.fileExists(trkfilecurr) ,
            partTrkFileTimestamps(i) = obj.backend_.fileModTime(trkfilecurr) ;
            parttrkfileNfrmtracked(i) = obj.backend_.readTrkFileStatus(trkfilecurr) ;
            trackedFrameCountSource(i) = 1 ;  % got it from the final trk file
            logger.log('Read %d frames tracked from %s\n',parttrkfileNfrmtracked(i),trkfilecurr);
          else
            logger.log('Part trk file %s and trk file %s do not exist\n',parttrkfilecurr,trkfilecurr);
          end
        end
      end

      njobs = obj.njobs ;
      try
        % isRunningFromJobIndex = true([nJobs, 1]) ;  % TODO: Make this actually check if the spawned jobs are running  
        isRunningFromJobIndex = obj.backend_.isAliveFromRegisteredJobIndex('track') ;  % njobs x 1
        isRunningFromTripleIndex = obj.replicateJobs_(isRunningFromJobIndex) ;
        % isRunning = obj.replicateJobs_(isRunningFromJobIndex);  % nMovies x nViews x nStages
        %killFileExists = cellfun(@obj.backend_.fileExists,killfiles);
        doesOutputTrkFileExistFromTripleIndex = cellfun(@(fileName)(obj.backend_.fileExists(fileName)),trkfiles); % nmovies x nviews x nstages
        tfComplete = doesOutputTrkFileExistFromTripleIndex & ~isRunningFromTripleIndex ;
        %logger.log('tfComplete = %s\n',mat2str(tfComplete(:)'));
        tfErrFileErrFromJobIndex = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),errfiles); % njobs x 1
        logFilesExistFromJobIndex = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),logfiles); % njobs x 1
        pollsuccess = true ;
      catch me
        % Likely a filesystem error checking for the files
        isRunningFromJobIndex = false(njobs,1) ;
        isRunningFromTripleIndex = obj.replicateJobs_(isRunningFromJobIndex) ;
        tfComplete = false(size(trkfiles)) ;
        tfErrFileErrFromJobIndex = true(size(errfiles)) ;
        logFilesExistFromJobIndex = true(size(logfiles)) ;
        pollsuccess = false ;
      end
      % bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      result = struct(...
        'pollsuccess',{pollsuccess}, ...
        'isPopulated',{obj.replicateJobs_(true(njobs,1))}, ...
        'tfComplete',{tfComplete},...
        'isRunning',isRunningFromTripleIndex,...
        'errFile',{obj.replicateJobs_(errfiles)},... % char, full path to DL err file
        'errFileExists',{obj.replicateJobs_(tfErrFileErrFromJobIndex)},... % true if errFile exists and has size>0
        'logFile',{obj.replicateJobs_(logfiles)},... % char, full path to Bsub logfile
        'logFileExists',{obj.replicateJobs_(logFilesExistFromJobIndex)},...
        'iview',{repmat(1:obj.nViews,[obj.nMovies,1,obj.nStages])},...
        'movfile',{repmat(obj.movfiles,[1 1 obj.nStages])},...
        'trkfile',{trkfiles},...
        'parttrkfile',{parttrkfiles},...
        'parttrkfileTimestamp',{partTrkFileTimestamps},...
        'parttrkfileNfrmtracked',{parttrkfileNfrmtracked}, ...
        'trackedFrameCountSource',{trackedFrameCountSource}) ;
      assert(isscalar(result)) ;
    end  % function

    function result = pollForList(obj, logger)
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end

      logger.log('in BgTrackPoller::pollForList()\n');
      errfiles = obj.toTrackInfos_.getErrFiles() ; % njobs x 1
      logfiles = obj.toTrackInfos_.getLogFiles() ; % njobs x 1
      % killfiles = obj.getKillFiles(); % njobs x 1
      outfiles = col(obj.toTrackInfos_.getListOutfiles()) ; % njobs x 1

      outTrkFileTimestamps = nan(size(outfiles)) ;  % njobs x 1
      for i = 1:numel(outfiles),
        trkfilecurr = outfiles{i};
        tmp = dir(trkfilecurr);
        if ~isempty(tmp) ,
          outTrkFileTimestamps(i) = tmp.datenum;
        else
          logger.log('Output file %s does not exist\n',outfiles{i});
        end
      end

      njobs = obj.njobs ;
      try
        isRunningFromJobIndex = obj.backend_.isAliveFromRegisteredJobIndex('track') ;  % njobs x 1
        doesOutputTrkFileExistFromJobIndex = cellfun(@(fileName)(obj.backend_.fileExists(fileName)),outfiles); % njobs x 1
        tfCompleteFromJobIndex = doesOutputTrkFileExistFromJobIndex & ~isRunningFromJobIndex ; % njobs x 1
        tfErrFileErrFromJobIndex = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),errfiles); % njobs x 1
        logFilesExistFromJobIndex = cellfun(@(fileName)(obj.backend_.fileExistsAndIsNonempty(fileName)),logfiles); % njobs x 1
        pollsuccess = true ;
      catch me
        % Likely a filesystem error checking for the files
        isRunningFromJobIndex = false(njobs,1) ;
        tfCompleteFromJobIndex = false(size(outfiles)) ;
        tfErrFileErrFromJobIndex = true(size(errfiles)) ;
        logFilesExistFromJobIndex = true(size(logfiles)) ;
        pollsuccess = false ;
      end
      logger.log('tfComplete = %s\n',mat2str(tfCompleteFromJobIndex(:)'));
      
      %bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      result = struct(...
        'pollsuccess',{pollsuccess}, ...
        'isPopulated',{true(njobs,1)}, ...        
        'tfComplete',{tfCompleteFromJobIndex},...
        'isRunning',{isRunningFromJobIndex},...
        'errFile',{errfiles},... % char, full path to DL err file
        'errFileExists',{tfErrFileErrFromJobIndex},... % true of errFile exists and has size>0
        'logFile',{logfiles},... % char, full path to Bsub logfile
        'logFileExists',{logFilesExistFromJobIndex},...
        'iview',{1:obj.nViews},...
        'movfile',{''},...
        'outfile',{outfiles},...
        'outfileTimestamp',{outTrkFileTimestamps} );
      assert(isscalar(result)) ;      
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
