classdef BgTrackWorkerObjAWS < BgWorkerObjAWS & BgTrackWorkerObj 
  methods    
    function obj = BgTrackWorkerObjAWS(nviews, track_type, varargin)
      obj@BgWorkerObjAWS(varargin{:});
      obj.nviews = nviews ;
      obj.track_type = track_type ;
    end
    
    function sRes = workOnMovie(obj, logger)
      % sRes: [nMovies x nviews x nStages] struct array      
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger() ;
      end

      % Order important, check if job is running first. If we check after
      % looking at artifacts, job may stop in between time artifacts and 
      % isRunning are probed.
      % isRunning does not seem to do anything right now!!
      %logger.log('Inside BgTrackWorkerObjAWS::compute()\n') ;
      isRunning = obj.getIsRunning();
      isRunning = isRunning(:);
      if isempty(isRunning)
        isRunning = true(obj.njobs,1);
      else
        assert(numel(isRunning)==obj.njobs);
      end
      
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 1\n') ;
      errfiles = obj.getErrFile(); % njobs x 1
      logfiles = obj.getErrFile(); % njobs x 1
      killfiles = obj.getKillFiles(); % njobs x 1
      parttrkfiles = obj.getPartTrkFile(); % nmovies x nviews x nstages
      trkfiles = obj.getTrkFile(); % nmovies x nviews x nstages
      
      % KB 20190115: also get locations of part track files and timestamps
      % of last modification
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 2\n') ;
      partTrkFileTimestamps = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      parttrkfileNfrmtracked = nan(size(parttrkfiles)); % nmovies x nviews x nstages
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.1\n') ;
      for i = 1:numel(parttrkfiles),
        %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.2\n') ;
        parttrkfilecurr = parttrkfiles{i};
        %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.3\n') ;
        does_file_exist = obj.fileExists(parttrkfilecurr) ;
        %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.4\n') ;
        if does_file_exist ,
          %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.5\n') ;
          partTrkFileTimestamps(i) = obj.fileModTime(parttrkfilecurr)  ;
            % Non-AWS backends use the Matlab datenum of the mtime to compare partial trk file
            % modification times.  This is different (seconds since Epoch), but they're
            % only ever checked for nan and compared to each other, so as long as they're
            % internally consistent it's fine.
          %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.6\n') ;
          parttrkfileNfrmtracked(i) = obj.backend.readTrkFileStatus(parttrkfilecurr, obj.partFileIsTextStatus, logger) ;
          %logger.log('Read %d frames tracked from %s\n',parttrkfileNfrmtracked(i),parttrkfilecurr);
          assert(~isnan(parttrkfileNfrmtracked(i)));
          %logger.log('BgTrackWorkerObjAWS::compute() milestone 2.7\n') ;
        else
          %logger.log('Part trk file %s does not exist\n',parttrkfiles{i});
        end
      end

      % Get the number of tracked frames from the completed trk files
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 3\n') ;
      tfComplete = false(size(trkfiles)) ;
      trkfileNfrmtracked = nan(size(trkfiles)); % nmovies x nviews x nstages
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 3.1\n') ;
      for i = 1:numel(trkfileNfrmtracked),
        %logger.log('BgTrackWorkerObjAWS::compute() milestone 3.2\n') ;
        trkfilecurr = trkfiles{i};
        %logger.log('BgTrackWorkerObjAWS::compute() milestone 3.3\n') ;
        does_file_exist = obj.fileExists(trkfilecurr) ;
        %logger.log('BgTrackWorkerObjAWS::compute() milestone 3.4\n') ;
        tfComplete(i) = does_file_exist ;
        if does_file_exist ,
          %logger.log('BgTrackWorkerObjAWS::compute() milestone 3.5\n') ;
          trkfileNfrmtracked(i) = obj.backend.readTrkFileStatus(trkfilecurr, false, logger) ;
          %logger.log('Read %d frames tracked from %s\n',trkfileNfrmtracked(i),trkfilecurr);
        else
          %logger.log('Trk file %s does not exist\n',trkfiles{i});
        end
      end
      
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 4\n') ;
      isRunning = obj.replicateJobs(isRunning);
      killFileExists = cellfun(@obj.fileExists,killfiles);
      %tfComplete = cellfun(@obj.fileExists,trkfiles); % nmovies x njobs x nstages
      %logger.log('tfComplete = %s\n',mat2str(tfComplete(:)'));
      tfErrFileErr = cellfun(@obj.fileExistsAndIsNonempty,errfiles); % njobs x 1
      logFilesExist = cellfun(@obj.fileExistsAndIsNonempty,logfiles); % njobs x 1
      bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
      %logger.log('BgTrackWorkerObjAWS::compute() milestone 5\n') ;
      sRes = struct(...
        'tfComplete',num2cell(tfComplete),...
        'isRunning',num2cell(isRunning),...
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
        'parttrkfileNfrmtracked',num2cell(parttrkfileNfrmtracked),...
        'trkfileNfrmtracked',num2cell(trkfileNfrmtracked),...
        'killFile',obj.replicateJobs(killfiles),...
        'killFileExists',num2cell(obj.replicateJobs(killFileExists)),...
        'isexternal',obj.isexternal... % scalar expansion
        );
      %logger.log('About to exit BgTrackWorkerObjAWS::compute().') ;
    end  % function   
  end  % methods
end  % classdef
