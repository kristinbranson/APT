classdef BgWorkerObjLocalFilesys < BgWorkerObj
  %
  % 
  % 1. Artifacts written to local filesys
  % 2. Process killed by sending message, polling to confirm, and touching
  % filesystem tok
  %
  
  properties
    % Each element of jobID corresponds to a DL process. Since serial
    % tracking across movies and views is possible, a single DL process may 
    % track across multiple moviesets or views.
    jobID % [nmovjob x nviewJobs] bsub jobID; or docker cellstr containerID
    
    killPollIterWaitTime = 1; % sec
    killPollMaxWaitTime = 12; % sec
  end
  
  methods (Abstract)
    killJob(obj,jID) % kill a single job. jID is scalar jobID
    fcn = makeJobKilledPollFcn(obj,jID) % create function that returns true when job is confirmed killed. jID is scalar jobID
    createKillToken(obj,killtoken) % create/touch filesystem KILL token. killtoken is full linux path
    %queryClusterJobs(obj)
  end
  
  methods
    
    function obj = BgWorkerObjLocalFilesys(varargin)
      obj@BgWorkerObj(varargin{:});
    end
    
    function tf = fileExists(~,file)
      tf = exist(file,'file')>0;
    end
    
    function tf = errFileExistsNonZeroSize(~,errFile)
      tf = BgWorkerObjLocalFilesys.errFileExistsNonZeroSizeStc(errFile);
    end
        
    function s = fileContents(~,file)
      if exist(file,'file')==0
        s = '<file does not exist>';
      else
        lines = readtxtfile(file);
        s = sprintf('%s\n',lines{:});
      end
    end

    function nframes = readTrkFileStatus(obj,f,partFileIsTextStatus)
      if nargin < 2,
        partFileIsTextStatus = false;
      end
      nframes = 0;
      if ~exist(f,'file'),
        return;
      end
      if partFileIsTextStatus,
        nframes = BgWorkerObj.readTrkFileStatus(obj,f,partFileIsTextStatus);
      else
        try
          nframes = TrkFile.getNFramesTrackedMatFile(f);
        catch
          fprintf('Could not read tracking progress from %s\n',f);
        end
      end
    end
    
    function [tfsucc,warnings] = killProcess(obj)
      tfsucc = false;
      warnings = {};
      
      killfiles = obj.getKillFiles();
      killfiles = unique(killfiles);
      jobids = obj.jobID;
      % AL 20210805:
      % for tracking, numel(jobids)==numel(killfiles)
      % for training, this need not hold due to jobs that spawn eg two-view
      % or two-stage tracking serially.
      %assert(isequal(numel(jobids),numel(killfiles)));
      tfJobsDMCsAlign = numel(jobids)==numel(killfiles);
      % if this is true, assume jobids correspond to killfiles which may
      % not be 100.00% true but if not it's an extreme corner
      
      for ijb=1:numel(jobids),
        obj.killJob(jobids(ijb));
      end

      iterWaitTime = obj.killPollIterWaitTime;
      maxWaitTime = obj.killPollMaxWaitTime;

      for ijb=1:numel(jobids),
        fcn = obj.makeJobKilledPollFcn(jobids(ijb));
        tfsucc = waitforPoll(fcn,iterWaitTime,maxWaitTime);
        
        if ~tfsucc
          warnings{end+1} = 'Could not confirm that job was killed.'; %#ok<AGROW>
          warningNoTrace('Could not confirm that job was killed.');
        elseif tfJobsDMCsAlign
          % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to
          % pick up
          kfile = killfiles{ijb};
          tfsucc = obj.createKillToken(kfile);
          if ~tfsucc,
            warnings{end+1} = sprintf('Failed to create KILLED token: %s',kfile); %#ok<AGROW>
          end
        elseif numel(jobids)==1 % single job, assume 'scalar expansion' across killfiles
          for ikfile=1:numel(killfiles)
            kfile = killfiles{ikfile};
            tfsucc = obj.createKillToken(kfile);
            if ~tfsucc,
              warnings{end+1} = sprintf('Failed to create KILLED token: %s',kfile); %#ok<AGROW>
            end
          end
        else
          % not sure if this can happen
          warningNoTrace('Not creating KILLED token; unexpected job/killfile correspondence.');
        end
        
        % bgTrnMonitor should pick up KILL tokens and stop bg trn monitoring
      end
    end
    
    function res = queryMyJobsStatus(obj)
      % returns cellstr array same size as .jobID [nmovjob x nviewJobs]
      if ~iscell(obj.jobID),
        jobID = num2cell(obj.jobID);
      else
        jobID = obj.jobID;
      end
      res = cellfun(@obj.queryJobStatus,jobID,'uni',0);
    end
    
    function tfIsRunning = getIsRunning(obj)
      % tfIsRunning: same size as .jobID
      
      ids = obj.jobID;
      tfIsRunning = true(size(ids));
      try
        for i = 1:numel(ids),
          if iscell(ids),
            id = ids{i};
            if isempty(id),
              continue;
            end
          else
            id = ids(i);
            if isnan(id),
              continue;
            end
          end
          % Guard against nan jobIDs for now. This can occur when a job
          % is started; the monitor is started before the job is spawned,
          % and the monitorviz calls this after receiving the first
          % result from the bg worker before the jID is set.
          tfIsRunning(i) = ~obj.isKilled(id);
        end
        %fprintf('isRunning = %s\n',mat2str(tfIsRunning));
      catch ME,
        fprintf('Error in updateIsRunning:\n%s\n',getReport(ME));
      end
    end
    
    function trnImgIfo = loadTrainingImages(obj)
      trnImgIfo = cell(1,obj.dmcs.n);
      necfields = {'idx','ims','locs'};
      for i=1:obj.dmcs.n,
        f = obj.dmcs.trainImagesNameLnx(i);
        f = f{1};
        if exist(f,'file')>0
          infocurr = load(f,'-mat');
          if ~all(isfield(infocurr,necfields)),
            warningNoTrace('Training image file ''%s'' exists, but not all fields (yet) saved in it.',f);
            continue;
          end
          trnImgIfo{i} = infocurr;          
          trnImgIfo{i}.name = obj.dmcs.getNetDescriptor(i);
          trnImgIfo{i}.name = trnImgIfo{i}.name{1};
        else
          warningNoTrace('Training image file ''%s'' does not exist yet.',f);
        end
      end
    end
        
  end
    
  methods (Static)
    function tfErrFileErr = errFileExistsNonZeroSizeStc(errFile)
      tfErrFileErr = exist(errFile,'file')>0;
      if tfErrFileErr
        direrrfile = dir(errFile);
        tfErrFileErr = direrrfile.bytes>0;
      end
    end
  end
  
end
