classdef BgTrackWorkerObjAWS < BgWorkerObjAWS & BgTrackWorkerObj 
  methods    
    function obj = BgTrackWorkerObjAWS(nviews,varargin)
      obj@BgWorkerObjAWS(varargin{:});
      obj.nviews = nviews;
    end
    
    function sRes = compute(obj, logger)
      % sRes: [nMovies x nviews x nStages] struct array      
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger(1, 'BgTrackWorkerObjAWS::compute()') ;
      end

      % Order important, check if job is running first. If we check after
      % looking at artifacts, job may stop in between time artifacts and 
      % isRunning are probed.
      % isRunning does not seem to do anything right now!!
      logger.log('Inside BgTrackWorkerObjAWS::compute()\n') ;
      isRunning = obj.getIsRunning();
      isRunning = isRunning(:);
      if isempty(isRunning)
        isRunning = true(obj.njobs,1);
      else
        assert(numel(isRunning)==obj.njobs);
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
        parttrkfilecurr = parttrkfiles{i};
        does_file_exist = obj.fileExists(parttrkfilecurr) ;
        if does_file_exist ,
          partTrkFileTimestamps(i) = obj.fileModTime(parttrkfilecurr)  ;
            % Non-AWS backend use the Matlab datenum of the mtime to compare partial trk file
            % modification times.  This is different (seconds since Epoch), but they're
            % only ever checked for nan and compared to each other, so as long as they're
            % internally consistent it's fine.
          parttrkfileNfrmtracked(i) = obj.readTrkFileStatus(parttrkfilecurr, obj.partFileIsTextStatus, logger) ;
          logger.log('Read %d frames tracked from %s\n',parttrkfileNfrmtracked(i),parttrkfilecurr);
          assert(~isnan(parttrkfileNfrmtracked(i)));
        else
          logger.log('Part trk file %s does not exist\n',parttrkfiles{i});
        end
      end

      % Get the number of tracked frames from the completed trk files
      tfComplete = false(size(trkfiles)) ;
      trkfileNfrmtracked = nan(size(trkfiles)); % nmovies x nviews x nstages
      for i = 1:numel(trkfileNfrmtracked),
        trkfilecurr = trkfiles{i};
        does_file_exist = obj.fileExists(trkfilecurr) ;
        tfComplete(i) = does_file_exist ;
        if does_file_exist ,
          trkfileNfrmtracked(i) = obj.readTrkFileStatus(trkfilecurr, false, logger) ;
          logger.log('Read %d frames tracked from %s\n',trkfileNfrmtracked(i),trkfilecurr);
        else
          logger.log('Trk file %s does not exist\n',trkfiles{i});
        end
      end
      
      isRunning = obj.replicateJobs(isRunning);
      killFileExists = cellfun(@obj.fileExists,killfiles);
      %tfComplete = cellfun(@obj.fileExists,trkfiles); % nmovies x njobs x nstages
      logger.log('tfComplete = %s\n',mat2str(tfComplete(:)'));
      tfErrFileErr = cellfun(@obj.errFileExistsNonZeroSize,errfiles); % njobs x 1
      logFilesExist = cellfun(@obj.errFileExistsNonZeroSize,logfiles); % njobs x 1
      bsuberrlikely = cellfun(@obj.logFileErrLikely,logfiles); % njobs x 1
      
      % nMovies x nviews x nStages
      % We return/report a results structure for every movie/trkfile, even
      % if views/movs are tracked serially (nMovJobs>1 or nViewJobs>1). In
      % this way the monitor can track/viz the progress of each movie/view.
      
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
    end  % function
   
    function nframes = readTrkFileStatus(obj, filename, partFileIsTextStatus, logger)
      % Read the number of frames remaining according to the remote file at location
      % filename.  If partFileIsTextStatus is true, this file is assumed to be a
      % text file.  Otherwise, it is assumed to be a .mat file.      
      if ~exist('partFileIsTextStatus', 'var') || isempty(partFileIsTextStatus) ,
        partFileIsTextStatus = false;
      end
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger(1, 'BgTrackWorkerObjAWS::readTrkFileStatus()') ;
      end

      logger.log('partFileIsTextStatus: %d', double(partFileIsTextStatus)) ;
      nframes = 0;
      if ~obj.fileExists(filename) ,
        return
      end
      if partFileIsTextStatus,
        str = obj.fileContents(filename) ;
        nframes = TrkFile.getNFramesTrackedString(str) ;
      else
        local_filename = strcat(tempname(), '.mat') ;  % Has to have an extension or matfile() will add '.mat' to the filename
        did_succeed = obj.awsEc2.scpDownloadOrVerify(filename, local_filename) ;
        if did_succeed ,
          logger.log('Successfully downloaded remote tracking file %s\n', filename) ;
          try
            nframes = TrkFile.getNFramesTrackedMatFile(local_filename) ;
          catch me
            logger.log('Could not read tracking progress from remote file %s: %s\n', filename, me.message) ;
          end

          logger.log('Read that nframes = %d\n', nframes) ;
        else
          logger.log('Could not download tracking progress from remote file %s\n', filename) ;
        end
      end
    end  % function

%     function sRes = compute(obj)
%       % sRes: [nviews] struct array
%       %
%       % Production/external tracking not handled yet
% 
%       aws = obj.awsEc2;
%       
%       % Handle err/log files etc; there are nViewJobs of these
% 
%       errFileExists = num2cell(false(size(obj.artfctErrFiles)));
%       logFileExists = num2cell(false(size(obj.artfctLogfiles)));
%       logFileErrLikely = num2cell(false(size(obj.artfctLogfiles)));
%       killFileExists = num2cell(false(size(obj.killFiles)));
%       for ivwjb=1:obj.nViewJobs
%         errFile = obj.artfctErrFiles{ivwjb};
%         logFile = obj.artfctLogfiles{ivwjb};
%         killFile = obj.killFiles{ivwjb};
%         fspollargs = ...
%           sprintf('existsNE %s existsNE %s existsNEerr %s exists %s',...
%             errFile,logFile,logFile,killFile);
%         cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollargs);
%         
%         fprintf('The time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
%         [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
%         if tfpollsucc
%           reslines = regexp(res,'\n','split');
%           tfpollsucc = iscell(reslines) && numel(reslines)==4+1; % last cell is {0x0 char}
%         end        
%         if tfpollsucc
%           errFileExists{ivwjb} = strcmp(reslines{1},'y');
%           logFileExists{ivwjb} = strcmp(reslines{2},'y');
%           logFileErrLikely{ivwjb} = strcmp(reslines{3},'y');
%           killFileExists{ivwjb} = strcmp(reslines{4},'y');          
%           fprintf('The poll succeeded. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
%         else
%           % none; errFileExists, etc initted to false
%         end
%       end
% 
%       nVwPerJob = obj.nViewsPerJob;
% 
%       sRes = struct(...
%         'tfComplete',[],...
%         'errFile',repmat(obj.artfctErrFiles,[1 nVwPerJob]),... % char, full path to DL err file
%         'errFileExists',repmat(errFileExists,[1 nVwPerJob]),... % true of errFile exists and has size>0
%         'logFile',repmat(obj.artfctLogfiles,[1 nVwPerJob]),... % char, full path to Bsub logfile
%         'logFileExists',repmat(logFileExists,[1 nVwPerJob]),...
%         'logFileErrLikely',repmat(logFileErrLikely,[1 nVwPerJob]),... % true if bsub logfile looks like err
%         'iview',num2cell((1:obj.nviews)),...
%         'movfile',obj.movfiles,...
%         'trkfile',obj.artfctTrkfiles,...
%         'parttrkfile',obj.artfctPartTrkfiles,...
%         'parttrkfileTimestamp',[],...
%         'parttrkfileNfrmtracked',[],... % 'extra' in BgTrackWorkerObjAWS 
%         'trkfileNfrmtracked',[],... % 'extra' in BgTrackWorkerObjAWS 
%         'killFile',repmat(obj.killFiles,[1 nVwPerJob]),...
%         'killFileExists',repmat(killFileExists,[1 nVwPerJob]),...
%         'isexternal',obj.isexternal);
%       
%       assert(numel(sRes)==obj.nviews);
%       
%       % Handle trk/part files etc; these are one per view
%       for ivw=1:obj.nviews
%         trkfile = obj.artfctTrkfiles{ivw};
%         partFile = obj.artfctPartTrkfiles{ivw};
%                 
%         % See AWSEC2 convenience meth
%         fspollargs = ...
%           sprintf('exists %s lastmodified %s nfrmtracked %s nfrmtracked %s',...
%             trkfile,partFile,partFile,trkfile);
%         cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollargs);
% 
%         fprintf('The time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
%         [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
%         if tfpollsucc
%           reslines = regexp(res,'\n','split');
%           tfpollsucc = iscell(reslines) && numel(reslines)==4+1; % last cell is {0x0 char}
%         end
%         
%         if tfpollsucc          
%           sRes(ivw).tfComplete = strcmp(reslines{1},'y');
%           sRes(ivw).parttrkfileTimestamp = str2double(reslines{2}); % includes nan for 'DNE'
%           sRes(ivw).parttrkfileNfrmtracked = str2double(reslines{3}); % includes nan for 'DNE'
%           sRes(ivw).trkfileNfrmtracked = str2double(reslines{4}); % includes nan for 'DNE'
%           
%           fprintf('The poll succeeded. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
%         else
%           % trackWorkerObj results don't have a 'pollsuccess' for some
%           % reason
%           sRes(ivw).tfComplete = false;
%           sRes(ivw).parttrkfileTimestamp = nan;
%           sRes(ivw).parttrkfileNfrmtracked = nan;
%           sRes(ivw).trkfileNfrmtracked = nan;
%           
%           fprintf('The poll failed. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
%         end  % if
%       end  % for ivw
%       
%     end  % function
  end  % methods
end  % classdef
